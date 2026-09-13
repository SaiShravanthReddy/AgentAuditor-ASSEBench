"""Tests for infer.py's process_json_file() completion messaging.

Regression guard for a real incident: this function used to print "Processing complete! Final
output saved to: <output_file>" unconditionally, even when every item's API call failed and
output_file was never actually written (that write only happens inside the per-item loop, for
items that got *some* response). A dead/invalid API key made every call fail for both us and a
teammate independently; the misleading message claimed success right before the next pipeline
stage (infer_json_repair.py) failed with "input file not found" on a file that was never created.
"""
import json
from unittest.mock import patch

from AgentAuditor.tasks.infer import process_json_file


def _write_input(path, n=2):
    data = [
        {"id": f"item-{i}", "contents": [{"role": "user", "content": "hi"}], "label": 0,
         "fewshot_demos": []}
        for i in range(n)
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


class TestProcessJsonFileCompletionMessage:
    def test_all_items_fail_prints_error_not_false_success(self, tmp_path, capsys):
        input_file = tmp_path / "k3.json"
        intermediate_file = tmp_path / "intermediate.json"
        output_file = tmp_path / "output-k3.json"
        failed_items_file = tmp_path / "failed.json"
        _write_input(input_file)

        with patch("AgentAuditor.tasks.infer.LLMHandler.call_llm_api", return_value=None):
            process_json_file(str(input_file), str(intermediate_file), str(output_file),
                               str(failed_items_file))

        captured = capsys.readouterr()
        assert "Processing complete! Final output saved to" not in captured.out
        assert "ERROR: Every item failed its API call" in captured.out
        assert str(output_file) in captured.out
        # The actual bug: the file must not exist, and the message must not claim it does.
        assert not output_file.exists()
        assert failed_items_file.exists()

    def test_some_items_succeed_prints_real_success(self, tmp_path, capsys):
        input_file = tmp_path / "k3.json"
        intermediate_file = tmp_path / "intermediate.json"
        output_file = tmp_path / "output-k3.json"
        failed_items_file = tmp_path / "failed.json"
        _write_input(input_file, n=2)

        with patch("AgentAuditor.tasks.infer.LLMHandler.call_llm_api", return_value='{"result": 0}'):
            process_json_file(str(input_file), str(intermediate_file), str(output_file),
                               str(failed_items_file))

        captured = capsys.readouterr()
        assert "Processing complete! Final output saved to" in captured.out
        assert "ERROR: Every item failed" not in captured.out
        assert output_file.exists()
        with open(output_file) as f:
            assert len(json.load(f)) == 2

    def test_partial_failure_still_reports_real_success_for_the_survivors(self, tmp_path, capsys):
        input_file = tmp_path / "k3.json"
        intermediate_file = tmp_path / "intermediate.json"
        output_file = tmp_path / "output-k3.json"
        failed_items_file = tmp_path / "failed.json"
        _write_input(input_file, n=2)

        responses = [None, '{"result": 1}']
        with patch("AgentAuditor.tasks.infer.LLMHandler.call_llm_api", side_effect=responses):
            process_json_file(str(input_file), str(intermediate_file), str(output_file),
                               str(failed_items_file))

        captured = capsys.readouterr()
        assert "Processing complete! Final output saved to" in captured.out
        assert output_file.exists()
        with open(output_file) as f:
            assert len(json.load(f)) == 1
        with open(failed_items_file) as f:
            assert len(json.load(f)) == 1
