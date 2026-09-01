"""Tests for eval.py's output-parsing and metrics functions - the code with this session's own
concrete bug history (ROOT_KEYS missing 'verdict'/'manipulation_attempt' caused 1062/1064 and
1062/1064 items to be misreported as processing errors on real runs, see AGENTAUDITOR_DIAGNOSIS.md
and git history). These are pure functions - no I/O, no LLM calls - so they're fully testable.
"""
import pytest

from AgentAuditor.tasks.eval import (
    normalize_output,
    extract_output,
    extract_confidence,
    calculate_metrics,
    calculate_ranking_metrics,
    ROOT_KEYS,
)


class TestNormalizeOutput:
    @pytest.mark.parametrize("value,expected", [
        ("1", 1), ("0", 0),
        ("unsafe", 1), ("safe", 0),
        ("1 (unsafe)", 1), ("0 (safe)", 0),
        (1, 1), (0, 0),
        (1.0, 1), (0.0, 0),
        ("1.0", 1), ("0.0", 0),
        ("UNSAFE", 1), ("Safe", 0),  # case-insensitive
    ])
    def test_recognized_formats(self, value, expected):
        assert normalize_output(value) == expected

    def test_unrecognized_raises(self):
        with pytest.raises(ValueError):
            normalize_output("maybe")

    def test_out_of_range_numeric_raises(self):
        with pytest.raises(ValueError):
            normalize_output("2")


class TestExtractOutput:
    def test_plain_string_or_number(self):
        assert extract_output("1") == 1
        assert extract_output(0) == 0

    @pytest.mark.parametrize("key", [
        "verdict", "manipulation_attempt", "Output", "output", "final_output", "safe", "label",
    ])
    def test_recognized_top_level_keys(self, key):
        """Every key ROOT_KEYS is supposed to cover, tried individually - a regression here means
        a real judge output format silently stops being recognized (this session's actual bug)."""
        assert extract_output({key: 1}) == 1
        assert extract_output({key: 0}) == 0

    def test_key_matching_is_case_and_spacing_insensitive(self):
        assert extract_output({"Final Judgment": "1 (unsafe)"}) == 1
        assert extract_output({"FINAL_JUDGMENT": "0 (safe)"}) == 0

    def test_nested_under_chain_of_thought(self):
        """The other half of this session's real regression: gpt-oss-20b nests its verdict under
        chain_of_thought once few-shot demos are actually working, matching the demos' own style."""
        assert extract_output({"chain_of_thought": {"verdict": "1 (unsafe)"}}) == 1
        assert extract_output({"chain_of_thought": {"Final Judgment": "0 (safe)"}}) == 0

    def test_falls_through_to_next_candidate_on_bad_value(self):
        """A record can have both a working key and a non-numeric one - must not give up on the
        first match if it doesn't normalize."""
        assert extract_output({"final_decision": "not a number", "safety": "1"}) == 1

    def test_no_recognized_key_raises(self):
        with pytest.raises(ValueError):
            extract_output({"totally_unrecognized_key": "1"})

    def test_non_dict_non_scalar_raises(self):
        with pytest.raises(ValueError):
            extract_output([1, 2, 3])

    def test_new_key_regression_guard(self):
        """If this fails, a real key found in production (e.g. 'safe_or_unsafe', seen in
        AGENTAUDITOR_DIAGNOSIS.md's CNFinBench investigation) was removed from ROOT_KEYS without
        a replacement - not necessarily wrong, but should be a deliberate decision, not silent."""
        expected_keys = {
            'output', 'finaloutput', 'finalsafetyscore', 'prediction', 'finaldecision', 'result',
            'decision', 'final', 'safety', 'safe', 'unsafe', 'finalanswer', 'safetyissue',
            'safetystatus', 'safetyflag', 'label', 'finalresult', 'safetyevaluation',
            'safetydecision', 'safetyrating', 'overallsafety', 'conclusion', 'safetyassessment',
            'finaljudgment', 'verdict', 'manipulationattempt',
        }
        normalized = {k.lower().replace('_', '').replace(' ', '') for k in ROOT_KEYS}
        assert expected_keys <= normalized, f"missing: {expected_keys - normalized}"


class TestExtractConfidence:
    def test_predicted_positive_uses_confidence_directly(self):
        assert extract_confidence({"confidence": 0.9}, predicted_label=1) == 0.9

    def test_predicted_negative_inverts_confidence(self):
        """A record predicted 0 with stated confidence 0.9 means '90% sure it's 0' - i.e. ~10%
        likely to be 1, so the positive-class score must be (1 - confidence)."""
        assert extract_confidence({"confidence": 0.9}, predicted_label=0) == pytest.approx(0.1)

    def test_confidence_nested_under_chain_of_thought(self):
        result = extract_confidence({"chain_of_thought": {"confidence": 0.8}}, predicted_label=1)
        assert result == 0.8

    def test_missing_confidence_returns_none(self):
        assert extract_confidence({"verdict": 1}, predicted_label=1) is None

    def test_non_dict_returns_none(self):
        assert extract_confidence("not a dict", predicted_label=1) is None

    def test_out_of_range_confidence_ignored(self):
        assert extract_confidence({"confidence": 1.5}, predicted_label=1) is None


class TestCalculateMetrics:
    def test_perfect_predictions(self):
        acc, prec, rec, f1 = calculate_metrics([1, 0, 1, 0], [1, 0, 1, 0])
        assert (acc, prec, rec, f1) == (1.0, 1.0, 1.0, 1.0)

    def test_empty_input(self):
        assert calculate_metrics([], []) == (0.0, 0.0, 0.0, 0.0)

    def test_known_confusion_matrix(self):
        # TP=1, FP=1, FN=1, TN=1
        true = [1, 1, 0, 0]
        pred = [1, 0, 1, 0]
        acc, prec, rec, f1 = calculate_metrics(true, pred)
        assert acc == 0.5
        assert prec == 0.5
        assert rec == 0.5
        assert f1 == 0.5

    def test_never_predicts_positive(self):
        """Matches the real FinVault Q2 shape found this session: precision undefined-by-zero-
        division must default to 0.0, not crash."""
        acc, prec, rec, f1 = calculate_metrics([1, 1, 0], [0, 0, 0])
        assert prec == 0.0
        assert rec == 0.0
        assert f1 == 0.0


class TestCalculateRankingMetrics:
    def test_needs_both_classes(self):
        assert calculate_ranking_metrics([1, 1, 1], [0.9, 0.8, 0.7]) is None

    def test_needs_at_least_two_items(self):
        assert calculate_ranking_metrics([1], [0.9]) is None

    def test_perfect_ranking(self):
        result = calculate_ranking_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        assert result is not None
        assert result['auroc'] == 1.0
        assert result['auprc'] == 1.0

    def test_returns_auroc_and_auprc_keys(self):
        result = calculate_ranking_metrics([0, 1], [0.3, 0.7])
        assert set(result.keys()) == {'auroc', 'auprc'}
