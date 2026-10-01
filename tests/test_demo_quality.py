"""Tests for demo_quality.py - the demo self-consistency metric flagged as missing in
results/PIPELINE_METRICS.md's `demo` section: "never whether the reasoning inside actually reaches
the correct conclusion for that record's known label."
"""
import json

from AgentAuditor.tasks.demo_quality import (
    compute_demo_consistency, extract_demo_verdict, write_demo_consistency,
)


def _valid_cot(output_str):
    return {"chain_of_thought": {"Start": "...", "Output": output_str}}


class TestExtractDemoVerdict:
    def test_safe_output_extracts_0(self):
        assert extract_demo_verdict(_valid_cot("0 (safe)")) == 0

    def test_unsafe_output_extracts_1(self):
        assert extract_demo_verdict(_valid_cot("1 (unsafe)")) == 1

    def test_malformed_structure_returns_none(self):
        assert extract_demo_verdict({"not_chain_of_thought": {}}) is None
        assert extract_demo_verdict("not even a dict") is None
        assert extract_demo_verdict(None) is None

    def test_unrecognized_output_string_returns_none(self):
        # is_correctly_nested_cot's own validator requires Output to be exactly one of the two
        # literals, so this case is already filtered upstream - but guard against it directly too.
        bad = {"chain_of_thought": {"Output": "yes, unsafe"}}
        assert extract_demo_verdict(bad) is None


class TestComputeDemoConsistency:
    def test_all_consistent(self):
        data = [
            {'id': 'a', 'label': 0, 'chain_of_thought': _valid_cot("0 (safe)")},
            {'id': 'b', 'label': 1, 'chain_of_thought': _valid_cot("1 (unsafe)")},
        ]
        result = compute_demo_consistency(data)
        assert result['total_records'] == 2
        assert result['num_validated'] == 2
        assert result['num_consistent'] == 2
        assert result['consistency_rate'] == 1.0
        assert result['inconsistent_ids'] == []

    def test_real_self_contradiction_flagged(self):
        """The actual defect this metric exists to catch: demo.py's prompt tells the model
        label=1 and says 'do not contradict this label', but the model's own generated Output
        field says '0 (safe)' anyway."""
        data = [
            {'id': 'contradicts-its-own-label', 'label': 1, 'chain_of_thought': _valid_cot("0 (safe)")},
        ]
        result = compute_demo_consistency(data)
        assert result['num_validated'] == 1
        assert result['num_consistent'] == 0
        assert result['consistency_rate'] == 0.0
        assert result['inconsistent_ids'] == ['contradicts-its-own-label']

    def test_unvalidated_demos_excluded_from_rate_not_counted_as_failures(self):
        """A demo still malformed after repair (failed_cot_processing_log.json territory) isn't a
        consistency failure - it's just not judgeable, and shouldn't silently deflate the rate."""
        data = [
            {'id': 'good', 'label': 0, 'chain_of_thought': _valid_cot("0 (safe)")},
            {'id': 'still-broken', 'label': 1, 'chain_of_thought': "not even nested"},
        ]
        result = compute_demo_consistency(data)
        assert result['total_records'] == 2
        assert result['num_validated'] == 1
        assert result['num_consistent'] == 1
        assert result['consistency_rate'] == 1.0

    def test_empty_input(self):
        result = compute_demo_consistency([])
        assert result == {'total_records': 0, 'num_validated': 0, 'num_consistent': 0,
                           'consistency_rate': None, 'inconsistent_ids': []}

    def test_no_validated_demos_rate_is_none_not_zero(self):
        data = [{'id': 'a', 'label': 0, 'chain_of_thought': "broken"}]
        result = compute_demo_consistency(data)
        assert result['num_validated'] == 0
        assert result['consistency_rate'] is None


class TestWriteDemoConsistency:
    def test_writes_sidecar_next_to_demo_fixed_json(self, tmp_path):
        output_path = str(tmp_path / "demo_fixed.json")
        metrics = {'total_records': 10, 'num_validated': 9, 'num_consistent': 8,
                   'consistency_rate': 0.889, 'inconsistent_ids': ['x']}
        quality_path = write_demo_consistency(output_path, metrics)
        assert quality_path == str(tmp_path / "demo_quality.json")
        with open(quality_path) as f:
            assert json.load(f) == metrics
