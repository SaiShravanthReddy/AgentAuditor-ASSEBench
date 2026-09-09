"""Tests for infer_llm_repair.py's find_malformed_ids() and validate_output_format().

Regression guard for a real bug: fix2_main() previously used a hardcoded list of bare integers
([83, 102, 163, ...], leftover from some earlier dev/test dataset) as its correction target IDs.
Every real dataset here uses string IDs (e.g. "harmless-MT_App-1"), so that list never matched
anything - this LLM-based repair fallback was silently a no-op on every run. find_malformed_ids()
replaces it by computing the target IDs from what's actually still broken.
"""
import json

from AgentAuditor.tasks.infer_llm_repair import find_malformed_ids, validate_output_format


class TestValidateOutputFormat:
    def test_valid_structure_passes(self):
        assert validate_output_format({"chain_of_thought": {"Output": "1 (unsafe)"}}) is True
        assert validate_output_format({"chain_of_thought": {"Output": "0 (safe)"}}) is True

    def test_non_dict_fails(self):
        assert validate_output_format("not a dict") is False
        assert validate_output_format(None) is False

    def test_missing_chain_of_thought_fails(self):
        assert validate_output_format({"Output": "1 (unsafe)"}) is False

    def test_wrong_output_value_fails(self):
        assert validate_output_format({"chain_of_thought": {"Output": "unsafe"}}) is False
        assert validate_output_format({"chain_of_thought": {"Output": 1}}) is False


class TestFindMalformedIds:
    def test_finds_only_malformed_items(self, tmp_path):
        data = [
            {"id": "good-1", "output": {"chain_of_thought": {"Output": "0 (safe)"}}},
            {"id": "bad-1", "output": "not parsed json"},
            {"id": "good-2", "output": {"chain_of_thought": {"Output": "1 (unsafe)"}}},
            {"id": "bad-2", "output": {"chain_of_thought": {"Output": "invalid value"}}},
        ]
        f = tmp_path / "output-k3_fixed.json"
        f.write_text(json.dumps(data))

        assert find_malformed_ids(str(f)) == ["bad-1", "bad-2"]

    def test_uses_real_string_ids_not_bare_integers(self, tmp_path):
        """The actual production bug this guards against: real dataset IDs are strings like
        'harmless-MT_App-1', not bare integers - the old hardcoded [83, 102, ...] list would never
        match any of these."""
        data = [{"id": "harmless-MT_App-16", "output": "unparseable"}]
        f = tmp_path / "output-k3_fixed.json"
        f.write_text(json.dumps(data))

        result = find_malformed_ids(str(f))
        assert result == ["harmless-MT_App-16"]
        assert all(isinstance(i, str) for i in result)

    def test_empty_file_returns_empty_list(self, tmp_path):
        f = tmp_path / "output-k3_fixed.json"
        f.write_text(json.dumps([]))
        assert find_malformed_ids(str(f)) == []

    def test_all_valid_returns_empty_list(self, tmp_path):
        data = [{"id": "x", "output": {"chain_of_thought": {"Output": "0 (safe)"}}}]
        f = tmp_path / "output-k3_fixed.json"
        f.write_text(json.dumps(data))
        assert find_malformed_ids(str(f)) == []
