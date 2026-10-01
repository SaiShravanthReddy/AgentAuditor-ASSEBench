"""Tests for infer_llm_repair.py's find_malformed_ids() and validate_output_format().

Regression guard for two real bugs:

1. fix2_main() previously used a hardcoded list of bare integers ([83, 102, 163, ...], leftover
   from some earlier dev/test dataset) as its correction target IDs. Every real dataset here uses
   string IDs (e.g. "harmless-MT_App-1"), so that list never matched anything - this LLM-based
   repair fallback was silently a no-op on every run. find_malformed_ids() replaces it by computing
   the target IDs from what's actually still broken.

2. validate_output_format() previously only accepted the strict nested
   {"chain_of_thought": {"Output": "0 (safe)"|"1 (unsafe)"}} shape - far narrower than what eval.py
   itself can already extract a verdict from (eval.py's extract_output recognizes ~25 other key
   names and both nested and top-level verdicts). Confirmed happening for real on HPG: a perfectly
   valid, already-extractable {"verdict": "0", "confidence": 0.99} record got wrongly flagged
   "malformed", sent through fix2's correction LLM, and came back as generic templated filler that
   doesn't reference the actual case - destroying a correct verdict and a real confidence score
   that never needed fixing. validate_output_format() now reuses eval.py's own extract_output() as
   the validity bar, so anything eval.py can already parse is left alone.
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

    def test_other_eval_recognized_shapes_also_pass(self):
        """These don't match the old strict nested schema, but eval.py's own extract_output()
        already parses all of them fine - validate_output_format() must agree, or it wrongly sends
        already-good records through the correction LLM (the real bug this guards against)."""
        assert validate_output_format({"Output": "1 (unsafe)"}) is True  # flat, not nested
        assert validate_output_format({"chain_of_thought": {"Output": "unsafe"}}) is True  # no "(unsafe)" suffix
        assert validate_output_format({"chain_of_thought": {"Output": 1}}) is True  # bare int, not string

    def test_real_incident_minimal_verdict_confidence_json_passes(self):
        """The exact real-world case found on HPG (2026-10): a compact, already-correct output
        that the old validator wrongly flagged as malformed."""
        assert validate_output_format({"verdict": "0", "confidence": 0.99}) is True

    def test_wrong_output_value_fails(self):
        assert validate_output_format({"chain_of_thought": {"Output": "invalid value"}}) is False
        assert validate_output_format({"chain_of_thought": {}}) is False
        assert validate_output_format({"some_other_key": "0 (safe)"}) is False


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
