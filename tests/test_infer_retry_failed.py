"""Tests for infer_retry_failed.py's _parse_llm_output() - must produce output in the exact same
shape infer.py's own inline parsing would, so a retried item is indistinguishable in output-k3.json
from one infer.py succeeded on the first time (eval.py and the repair stages must not be able to
tell the difference).
"""
from AgentAuditor.tasks.infer_retry_failed import _parse_llm_output


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
