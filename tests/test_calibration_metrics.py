"""Tests for calibration_metrics.py - Brier score and ECE, named in CLAUDE.md's "shared metrics
(AUROC/ECE/Brier)" for the TRACES comparison but never actually computed anywhere until now.
"""
import pytest

from AgentAuditor.tasks.calibration_metrics import compute_brier_score, compute_ece


class TestComputeBrierScore:
    def test_perfect_predictions_score_zero(self):
        true_labels = [0, 0, 1, 1]
        scores =      [0.0, 0.0, 1.0, 1.0]
        assert compute_brier_score(true_labels, scores) == 0.0

    def test_constant_half_guess_scores_quarter(self):
        true_labels = [0, 1]
        scores =      [0.5, 0.5]
        assert compute_brier_score(true_labels, scores) == 0.25

    def test_maximally_wrong_confident_predictions_score_one(self):
        true_labels = [0, 1]
        scores =      [1.0, 0.0]
        assert compute_brier_score(true_labels, scores) == 1.0

    def test_empty_input_returns_none(self):
        assert compute_brier_score([], []) is None

    def test_mismatched_lengths_returns_none(self):
        assert compute_brier_score([0, 1], [0.5]) is None


class TestComputeECE:
    def test_perfectly_calibrated_has_zero_ece(self):
        # Two clean groups, each bin's mean confidence exactly matches its own accuracy.
        confidences = [0.9] * 10 + [0.2] * 10
        correctness = [True] * 9 + [False] * 1 + [False] * 8 + [True] * 2
        ece, bins = compute_ece(confidences, correctness, num_bins=10)
        assert ece == pytest.approx(0.0, abs=1e-9)

    def test_overconfident_model_has_positive_ece(self):
        """Model always says 95% confident but is only right half the time - should be heavily
        penalized, this is the real failure mode ECE exists to catch."""
        confidences = [0.95] * 10
        correctness = [True, False] * 5  # 50% actual accuracy vs 95% stated confidence
        ece, bins = compute_ece(confidences, correctness, num_bins=10)
        assert ece > 0.4  # |0.95 - 0.5| = 0.45, single bin so ece == that gap exactly
        assert len(bins) == 1
        assert bins[0]['count'] == 10
        assert bins[0]['accuracy'] == 0.5

    def test_bin_details_only_include_nonempty_bins(self):
        confidences = [0.05, 0.95]
        correctness = [True, True]
        ece, bins = compute_ece(confidences, correctness, num_bins=10)
        assert len(bins) == 2  # only the two bins actually hit, not all 10

    def test_empty_input_returns_none(self):
        assert compute_ece([], []) is None

    def test_mismatched_lengths_returns_none(self):
        assert compute_ece([0.5], [True, False]) is None
