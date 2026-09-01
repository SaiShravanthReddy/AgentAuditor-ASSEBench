"""Tests for demo_repair.py's chain-of-thought validity check - the exact function infer_emb.py's
Fix 2 relies on to exclude broken demos from retrieval candidacy (see AGENTAUDITOR_DIAGNOSIS.md).
A regression here silently changes what counts as a "broken" demo on both sides at once.
"""
from AgentAuditor.tasks.demo_repair import is_correctly_nested_cot, validate_inner_cot_dict


class TestValidateInnerCotDict:
    def test_valid_dict_with_correct_output(self):
        assert validate_inner_cot_dict({"Start": "...", "Output": "1 (unsafe)"}) is True
        assert validate_inner_cot_dict({"Output": "0 (safe)"}) is True

    def test_wrong_output_format_rejected(self):
        """Must be exactly '0 (safe)' or '1 (unsafe)' - a bare '1' or 'unsafe' is not accepted
        here (this is stricter than eval.py's normalize_output on purpose - this is checking a
        chain-of-thought demo's internal consistency, not parsing a judge's final verdict)."""
        assert validate_inner_cot_dict({"Output": "1"}) is False
        assert validate_inner_cot_dict({"Output": "unsafe"}) is False

    def test_missing_output_key_rejected(self):
        assert validate_inner_cot_dict({"Start": "..."}) is False

    def test_non_dict_rejected(self):
        assert validate_inner_cot_dict("a raw string") is False
        assert validate_inner_cot_dict(None) is False


class TestIsCorrectlyNestedCot:
    def test_correctly_nested_valid(self):
        assert is_correctly_nested_cot({"chain_of_thought": {"Start": "...", "Output": "1 (unsafe)"}}) is True

    def test_raw_string_rejected(self):
        """The actual failure mode found in production this session: demo.py's LLM call
        sometimes returns unparsed text, which demo_repair.py couldn't fix and left as-is."""
        assert is_correctly_nested_cot("**Chain-of-Thought Reasoning**\n1. ...") is False

    def test_wrong_outer_key_rejected(self):
        assert is_correctly_nested_cot({"reasoning": {"Output": "1 (unsafe)"}}) is False

    def test_extra_outer_keys_rejected(self):
        """Must have exactly one key, 'chain_of_thought' - not that key plus others."""
        assert is_correctly_nested_cot({
            "chain_of_thought": {"Output": "1 (unsafe)"},
            "extra_key": "should not be here",
        }) is False

    def test_inner_value_not_dict_rejected(self):
        assert is_correctly_nested_cot({"chain_of_thought": "not a dict"}) is False

    def test_none_rejected(self):
        assert is_correctly_nested_cot(None) is False

    def test_empty_dict_rejected(self):
        assert is_correctly_nested_cot({}) is False
