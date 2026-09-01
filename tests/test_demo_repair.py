"""Tests for demo_repair.py's chain-of-thought validity check - the exact function infer_emb.py's
Fix 2 relies on to exclude broken demos from retrieval candidacy (see AGENTAUDITOR_DIAGNOSIS.md).
A regression here silently changes what counts as a "broken" demo on both sides at once.
"""
import json

import pytest

from AgentAuditor.tasks.demo_repair import (
    is_correctly_nested_cot,
    validate_inner_cot_dict,
    parse_json_with_brace_repair,
)


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


class TestParseJsonWithBraceRepair:
    def test_well_formed_json_parses_normally(self):
        text = '{"chain_of_thought": {"Output": "1 (unsafe)"}}'
        assert parse_json_with_brace_repair(text) == json.loads(text)

    def test_missing_one_closing_brace_is_repaired(self):
        """The actual production bug: 24/24 real repair-LLM failures in one run were exactly
        this pattern - otherwise-correct JSON missing its outermost closing brace."""
        broken = '{"chain_of_thought": {"Start": "...", "Output": "1 (unsafe)"}'
        result = parse_json_with_brace_repair(broken)
        assert result == {"chain_of_thought": {"Start": "...", "Output": "1 (unsafe)"}}

    def test_missing_two_closing_braces_is_repaired(self):
        broken = '{"chain_of_thought": {"nested": {"Output": "0 (safe)"}'
        result = parse_json_with_brace_repair(broken)
        assert result == {"chain_of_thought": {"nested": {"Output": "0 (safe)"}}}

    def test_genuinely_broken_json_still_raises(self):
        """Must not silently paper over real breakage - only the specific
        missing-closing-brace pattern should be recovered. An unterminated string value is a
        different kind of broken that appending braces can't fix - must still raise."""
        genuinely_broken = '{"chain_of_thought": "unterminated string'
        with pytest.raises(json.JSONDecodeError):
            parse_json_with_brace_repair(genuinely_broken)

    def test_balanced_but_invalid_json_raises(self):
        """Equal brace counts but still malformed (trailing comma) - the repair only ever adds
        braces, so this must fail exactly like plain json.loads would."""
        balanced_but_broken = '{"a": 1,}'
        with pytest.raises(json.JSONDecodeError):
            parse_json_with_brace_repair(balanced_but_broken)
