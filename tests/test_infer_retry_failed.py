"""Tests for infer_retry_failed.py's _parse_llm_output() - must produce output in the exact same
shape infer.py's own inline parsing would, so a retried item is indistinguishable in output-k3.json
from one infer.py succeeded on the first time (eval.py and the repair stages must not be able to
tell the difference) - and for append_recovered_item(), the fix for a real production bug.

Real incident (2026-10): on a live HiPerGator run, 31/33 items that genuinely succeeded during a
retry pass (confirmed via the job's own log - "ID X processed successfully!") still ended up
MISSING from the final output-k3.json. Reproduced again on a second, independent run (29/31 lost).
The old retry_failed_main() loaded output_data into memory ONCE at the start and trusted that
single in-memory list as authoritative for the entire loop, writing it back after every item. The
exact mechanism causing the loss was never conclusively pinned down (code review of this function,
fix1_main, and timer.py's wrapper found no bug in isolation), but append_recovered_item() makes it
moot: it re-reads output_file fresh immediately before every single write, so there is no
long-lived in-memory snapshot that could ever drift from what's actually on disk.
"""
import json
from unittest.mock import patch

from AgentAuditor.tasks.infer_retry_failed import (
    _parse_llm_output, append_recovered_item, retry_failed_main,
)


class TestParseLlmOutput:
    def test_valid_json_parsed_into_output(self):
        item = {"id": "x-1", "combined_prompt": "..."}
        result = _parse_llm_output(item, '{"chain_of_thought": {"Output": "1 (unsafe)"}}')
        assert result["output"] == {"chain_of_thought": {"Output": "1 (unsafe)"}}
        assert "output_parse_error" not in result

    def test_malformed_json_stored_raw_with_error_field(self):
        """Mirrors infer.py's own behavior: a parse failure doesn't drop the item, it's kept with
        the raw string and an output_parse_error field for the repair stages to pick up."""
        item = {"id": "x-2", "combined_prompt": "..."}
        result = _parse_llm_output(item, "not valid json at all")
        assert result["output"] == "not valid json at all"
        assert "output_parse_error" in result

    def test_wrapped_in_surrounding_quotes_stripped(self):
        """infer.py strips one layer of leading/trailing double-quotes before parsing, since some
        models wrap their JSON response in an extra pair - must match exactly."""
        item = {"id": "x-3", "combined_prompt": "..."}
        result = _parse_llm_output(item, '"{"chain_of_thought": {"Output": "0 (safe)"}}"')
        assert result["output"] == {"chain_of_thought": {"Output": "0 (safe)"}}

    def test_original_item_fields_preserved(self):
        """The retried item's own fields (id, label, combined_prompt, original_contents) must
        survive into the merged output-k3.json entry, same as infer.py's new_item = item.copy()."""
        item = {"id": "x-4", "label": 1, "combined_prompt": "...", "original_contents": [{"a": 1}]}
        result = _parse_llm_output(item, '{"chain_of_thought": {"Output": "1 (unsafe)"}}')
        assert result["id"] == "x-4"
        assert result["label"] == 1
        assert result["original_contents"] == [{"a": 1}]

    def test_does_not_mutate_input_item(self):
        """item.copy() inside _parse_llm_output must be a real copy - the caller's failed_items
        list entries shouldn't gain an 'output' key just from calling this."""
        item = {"id": "x-5", "combined_prompt": "..."}
        _parse_llm_output(item, '{"chain_of_thought": {"Output": "0 (safe)"}}')
        assert "output" not in item


class TestAppendRecoveredItem:
    def test_appends_to_existing_file(self, tmp_path):
        output_file = tmp_path / "output-k3.json"
        output_file.write_text(json.dumps([{"id": "existing-1", "label": 0}]))

        count = append_recovered_item(str(output_file), {"id": "new-1", "label": 1})

        assert count == 2
        with open(output_file) as f:
            data = json.load(f)
        ids = {item["id"] for item in data}
        assert ids == {"existing-1", "new-1"}

    def test_dedupes_by_id_rather_than_creating_a_duplicate(self):
        """If the same item id is appended twice (e.g. a re-run after a partial failure), the
        second write must replace the first, not create two entries with the same id - eval.py and
        every downstream stage assume one record per id."""
        import tempfile, os
        with tempfile.TemporaryDirectory() as d:
            output_file = os.path.join(d, "output-k3.json")
            with open(output_file, 'w') as f:
                json.dump([], f)

            append_recovered_item(output_file, {"id": "dup-1", "label": 0, "output": "first"})
            count = append_recovered_item(output_file, {"id": "dup-1", "label": 0, "output": "second"})

            assert count == 1
            with open(output_file) as f:
                data = json.load(f)
            assert len(data) == 1
            assert data[0]["output"] == "second"

    def test_regression_real_incident_survives_external_modification_between_writes(self, tmp_path):
        """The actual production failure mode this fix exists for: something modifies output_file
        on disk between this function's read and another call's read (whatever the real mechanism
        was - never conclusively identified, see this test file's module docstring). A version that
        holds a single in-memory list across multiple calls would silently lose whichever write
        didn't go through the stale copy. append_recovered_item() must not have that failure mode,
        because it re-reads immediately before every write rather than trusting a cached list."""
        output_file = tmp_path / "output-k3.json"
        output_file.write_text(json.dumps([{"id": "original-1", "label": 0}]))

        # First recovery - normal case.
        append_recovered_item(str(output_file), {"id": "recovered-1", "label": 1})

        # Simulate something else writing to the file in between (a second process, a different
        # stage, or whatever the real unidentified mechanism was) - NOT going through this
        # function, so a version relying on a long-lived in-memory accumulator would never see it.
        with open(output_file) as f:
            data = json.load(f)
        data.append({"id": "externally-added-1", "label": 0})
        with open(output_file, 'w') as f:
            json.dump(data, f)

        # Second recovery - must still succeed and must not clobber the externally-added item,
        # since it re-reads fresh rather than using a stale in-memory copy from before the
        # external modification.
        count = append_recovered_item(str(output_file), {"id": "recovered-2", "label": 1})

        with open(output_file) as f:
            final = json.load(f)
        final_ids = {item["id"] for item in final}
        assert final_ids == {"original-1", "recovered-1", "externally-added-1", "recovered-2"}
        assert count == 4


class TestRetryFailedMain:
    def _setup(self, tmp_path, failed_items, existing_output):
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()
        dataset_dir = tmp_path / "temp" / "mydataset"
        dataset_dir.mkdir(parents=True)
        with open(dataset_dir / "failed.json", 'w') as f:
            json.dump(failed_items, f)
        with open(dataset_dir / "output-k3.json", 'w') as f:
            json.dump(existing_output, f)
        return tasks_dir

    def test_no_failed_file_is_a_noop(self, tmp_path, monkeypatch, capsys):
        import AgentAuditor.tasks.infer_retry_failed as mod
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()
        (tmp_path / "temp" / "mydataset").mkdir(parents=True)
        monkeypatch.setattr(mod, '__file__', str(tasks_dir / "infer_retry_failed.py"))
        retry_failed_main("mydataset")
        assert "nothing to retry" in capsys.readouterr().out

    def test_recovers_items_and_shrinks_failed_json(self, tmp_path, monkeypatch, capsys):
        import AgentAuditor.tasks.infer_retry_failed as mod
        failed = [
            {"id": "a-1", "combined_prompt": "p1", "label": 0},
            {"id": "a-2", "combined_prompt": "p2", "label": 1},
        ]
        tasks_dir = self._setup(tmp_path, failed, existing_output=[{"id": "already-ok", "label": 0}])
        monkeypatch.setattr(mod, '__file__', str(tasks_dir / "infer_retry_failed.py"))

        with patch("AgentAuditor.tasks.infer_retry_failed.LLMHandler.call_llm_api",
                   return_value='{"chain_of_thought": {"Output": "0 (safe)"}}'):
            retry_failed_main("mydataset")

        out = capsys.readouterr().out
        assert "Recovered 2/2" in out

        with open(tmp_path / "temp" / "mydataset" / "output-k3.json") as f:
            output = json.load(f)
        assert {item['id'] for item in output} == {"already-ok", "a-1", "a-2"}

        with open(tmp_path / "temp" / "mydataset" / "failed.json") as f:
            still_failed = json.load(f)
        assert still_failed == []

    def test_partial_failure_leaves_item_in_failed_json(self, tmp_path, monkeypatch, capsys):
        import AgentAuditor.tasks.infer_retry_failed as mod
        failed = [
            {"id": "a-1", "combined_prompt": "p1", "label": 0},
            {"id": "a-2", "combined_prompt": "p2", "label": 1},
        ]
        tasks_dir = self._setup(tmp_path, failed, existing_output=[])
        monkeypatch.setattr(mod, '__file__', str(tasks_dir / "infer_retry_failed.py"))

        responses = ['{"chain_of_thought": {"Output": "0 (safe)"}}', None]
        with patch("AgentAuditor.tasks.infer_retry_failed.LLMHandler.call_llm_api", side_effect=responses):
            retry_failed_main("mydataset")

        with open(tmp_path / "temp" / "mydataset" / "output-k3.json") as f:
            output = json.load(f)
        assert {item['id'] for item in output} == {"a-1"}

        with open(tmp_path / "temp" / "mydataset" / "failed.json") as f:
            still_failed = json.load(f)
        assert len(still_failed) == 1
        assert still_failed[0]['id'] == "a-2"
        assert still_failed[0]['failure_reason'] == "API call failed after maximum retries (retry pass)"
