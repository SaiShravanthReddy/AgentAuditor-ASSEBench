"""Tests for threshold_recalibration.py - turns the "Q2's confidence signal is strong but the hard
verdict is miscalibrated" finding (results/MODEL_COMPARISON_SUMMARY.md) into a reusable tool.
"""
import json

from AgentAuditor.tasks.threshold_recalibration import (
    find_best_threshold, recalibrate, threshold_recalibrate_main,
)


def _record(label, output_verdict, confidence):
    return {
        'label': label,
        'output': {'Output': output_verdict, 'confidence': confidence},
    }


class TestFindBestThreshold:
    def test_finds_perfectly_separating_threshold(self):
        # Positive-class scores: label 1 items score high, label 0 items score low - a threshold
        # between the two groups should classify everything correctly.
        true_labels = [0, 0, 0, 1, 1, 1]
        scores =      [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
        result = find_best_threshold(true_labels, scores)
        assert result is not None
        threshold, accuracy = result
        assert 0.3 <= threshold <= 0.7
        assert accuracy == 1.0

    def test_returns_none_when_only_one_class_present(self):
        assert find_best_threshold([1, 1, 1], [0.5, 0.6, 0.7]) is None

    def test_returns_none_for_empty_scores(self):
        assert find_best_threshold([], []) is None


class TestRecalibrate:
    def test_hard_verdict_miscalibrated_but_confidence_score_separates_classes(self):
        """Reproduces the actual finding: the model's own 0/1 verdict is wrong on some items even
        though its confidence score (converted to a positive-class score by extract_confidence)
        would have separated the classes cleanly at a different threshold."""
        data = [
            # label=0, correctly verdicted 0, high confidence -> positive-class score low (~0.05)
            _record(0, 0, 0.95),
            _record(0, 0, 0.90),
            # label=1, but judge WRONGLY verdicted 0 (miscalibrated hard verdict) - yet its
            # confidence in that wrong call is low, so the positive-class score is still highish
            _record(1, 0, 0.40),
            _record(1, 0, 0.45),
            # label=1, correctly verdicted 1, high confidence -> positive-class score high (~0.9)
            _record(1, 1, 0.90),
            _record(1, 1, 0.85),
        ]
        result = recalibrate(data)

        assert result['num_items'] == 6
        assert result['num_with_confidence'] == 6

        # Hard verdict gets the 2 miscalibrated items wrong: 4/6 correct.
        assert result['hard_verdict']['accuracy'] == 4 / 6

        # Recalibrating on the confidence-derived score should do at least as well, since the
        # miscalibrated items' positive-class scores (0.60, 0.55) sit clearly above the two
        # true-negative items' scores (0.05, 0.10) and below the two true-positive items' (0.90,
        # 0.85) - a threshold around 0.6 recovers all 6.
        assert result['recalibrated'] is not None
        assert result['recalibrated']['accuracy'] >= result['hard_verdict']['accuracy']
        assert result['recalibrated']['accuracy'] == 1.0

    def test_no_confidence_field_anywhere_returns_none_recalibrated(self):
        """Matches a real scenario found this session: datasets from before the confidence-request
        prompt change have zero usable confidence values - recalibration must degrade gracefully,
        not crash."""
        data = [
            {'label': 0, 'output': {'Output': 0}},
            {'label': 1, 'output': {'Output': 1}},
            {'label': 1, 'output': {'Output': 0}},
        ]
        result = recalibrate(data)
        assert result['num_with_confidence'] == 0
        assert result['hard_verdict'] is not None
        assert result['recalibrated'] is None

    def test_empty_input(self):
        result = recalibrate([])
        assert result['num_items'] == 0
        assert result['hard_verdict'] is None
        assert result['recalibrated'] is None


class TestThresholdRecalibrateMain:
    def test_missing_file_does_not_raise(self, tmp_path, monkeypatch, capsys):
        import AgentAuditor.tasks.threshold_recalibration as mod
        monkeypatch.setattr(
            mod, '__file__', str(tmp_path / "tasks" / "threshold_recalibration.py")
        )
        (tmp_path / "tasks").mkdir()
        threshold_recalibrate_main("nonexistent-dataset")
        assert "not found" in capsys.readouterr().out

    def test_reads_real_output_file_and_prints_report(self, tmp_path, monkeypatch, capsys):
        import AgentAuditor.tasks.threshold_recalibration as mod
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()
        monkeypatch.setattr(mod, '__file__', str(tasks_dir / "threshold_recalibration.py"))

        dataset_dir = tmp_path / "temp" / "mydataset"
        dataset_dir.mkdir(parents=True)
        data = [
            _record(0, 0, 0.9), _record(0, 0, 0.85),
            _record(1, 1, 0.9), _record(1, 0, 0.4),
        ]
        with open(dataset_dir / "output-k3_corrected.json", "w") as f:
            json.dump(data, f)

        threshold_recalibrate_main("mydataset")
        out = capsys.readouterr().out
        assert "Threshold Recalibration" in out
        assert "Hard verdict" in out
        assert "Recalibrated" in out
