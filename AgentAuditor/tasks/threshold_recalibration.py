"""Optional stage, run manually after 'eval': sweeps the self-reported confidence score for the
hard-verdict threshold that would have scored best on this dataset, and reports how much accuracy
that recovers versus the judge's own raw 0/1 verdict.

Not wired into the main pipeline - run on demand:
    python -m AgentAuditor <dataset> threshold_recalibrate

Origin: results/MODEL_COMPARISON_SUMMARY.md found that CNFinBench Q2's confidence score is a
strong, genuine signal on gpt-oss-20b (mean confidence 0.78 for correct predictions vs. 0.05 for
wrong ones) but the model's own hard 0/1 verdict is badly miscalibrated - recalibrating the
decision threshold on data already on disk (zero new API cost, no rerun needed) recovered most of
a 46%->92% accuracy gap. That finding was analysis-only until now; this module makes it a reusable
tool for any dataset instead of a one-off script.

Caveat: the reported threshold is fit and evaluated on the SAME data (no held-out split), so its
accuracy is optimistic/in-sample, not a generalization estimate - useful for deciding whether the
confidence signal has more headroom than the raw verdict is capturing, not for picking a
production threshold blind.
"""
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from .eval import calculate_metrics, extract_confidence, extract_output


def _collect_scores(data: List[Dict[str, Any]]) -> Tuple[List[int], List[int], List[int], List[float]]:
    """Returns (true_labels, hard_predicted_labels, ranking_true_labels, ranking_scores) - the
    last two restricted to items with a usable confidence value, mirroring eval.py's own split
    between "all successfully processed items" and "items with a usable confidence score"."""
    true_labels: List[int] = []
    hard_predicted: List[int] = []
    ranking_true: List[int] = []
    ranking_scores: List[float] = []

    for item in data:
        if not isinstance(item, dict) or 'label' not in item or item['label'] not in (0, 1):
            continue
        if 'output' not in item:
            continue
        try:
            predicted_label = extract_output(item['output'])
        except (ValueError, KeyError, TypeError):
            continue

        true_labels.append(item['label'])
        hard_predicted.append(predicted_label)

        confidence_score = extract_confidence(item['output'], predicted_label)
        if confidence_score is not None:
            ranking_true.append(item['label'])
            ranking_scores.append(confidence_score)

    return true_labels, hard_predicted, ranking_true, ranking_scores


def find_best_threshold(true_labels: List[int], scores: List[float]) -> Optional[Tuple[float, float]]:
    """Sweeps every score value present as a candidate threshold (predict 1 if score >= threshold)
    and returns the (threshold, accuracy) pair with the highest accuracy. Returns None if both
    classes aren't present (accuracy at any threshold is meaningless/undefined otherwise, same
    guard eval.py's calculate_ranking_metrics uses for AUROC/AUPRC)."""
    if len(set(true_labels)) < 2 or not scores:
        return None

    best_threshold = 0.5
    best_accuracy = -1.0
    for candidate in sorted(set(scores)):
        predicted = [1 if s >= candidate else 0 for s in scores]
        accuracy, _, _, _ = calculate_metrics(true_labels, predicted)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = candidate
    return best_threshold, best_accuracy


def recalibrate(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes the original hard-verdict metrics and the best-achievable recalibrated metrics
    side by side. Returns a dict rather than printing directly so tests can assert on the numbers."""
    true_labels, hard_predicted, ranking_true, ranking_scores = _collect_scores(data)

    result: Dict[str, Any] = {
        'num_items': len(true_labels),
        'num_with_confidence': len(ranking_scores),
    }

    if true_labels:
        hard_accuracy, hard_precision, hard_recall, hard_f1 = calculate_metrics(true_labels, hard_predicted)
        result['hard_verdict'] = {
            'accuracy': hard_accuracy, 'precision': hard_precision, 'recall': hard_recall, 'f1': hard_f1,
        }
    else:
        result['hard_verdict'] = None

    best = find_best_threshold(ranking_true, ranking_scores)
    if best is None:
        result['recalibrated'] = None
    else:
        threshold, _ = best
        recal_predicted = [1 if s >= threshold else 0 for s in ranking_scores]
        recal_accuracy, recal_precision, recal_recall, recal_f1 = calculate_metrics(ranking_true, recal_predicted)
        result['recalibrated'] = {
            'threshold': threshold, 'accuracy': recal_accuracy, 'precision': recal_precision,
            'recall': recal_recall, 'f1': recal_f1, 'num_items': len(ranking_true),
        }

    return result


def print_recalibration_report(result: Dict[str, Any]) -> None:
    print(f"\n=== Threshold Recalibration ({result['num_items']} items, "
          f"{result['num_with_confidence']} with a usable confidence value) ===")

    if result['hard_verdict'] is None:
        print("No successfully processed items - nothing to recalibrate.")
        return

    hv = result['hard_verdict']
    print(f"\nHard verdict (judge's own 0/1 output):")
    print(f"  Accuracy: {hv['accuracy']:.4f}  Precision: {hv['precision']:.4f}  "
          f"Recall: {hv['recall']:.4f}  F1: {hv['f1']:.4f}")

    recal = result['recalibrated']
    if recal is None:
        print("\nRecalibration not possible - need both classes present among items with a "
              "usable confidence value (e.g. this dataset predates the confidence-request prompt "
              "change, or every confident item shares one true label).")
        return

    print(f"\nRecalibrated (best in-sample threshold on confidence score, "
          f"{recal['num_items']} items):")
    print(f"  Threshold: {recal['threshold']:.4f}")
    print(f"  Accuracy: {recal['accuracy']:.4f}  Precision: {recal['precision']:.4f}  "
          f"Recall: {recal['recall']:.4f}  F1: {recal['f1']:.4f}")
    print(f"\n  Delta vs. hard verdict (on the same {recal['num_items']}-item subset): "
          f"{recal['accuracy'] - hv['accuracy']:+.4f} accuracy")
    print("\nCaveat: this threshold is fit and evaluated on the same data (no held-out split) - "
          "it's an optimistic estimate of how much headroom the confidence signal has over the "
          "raw verdict, not a threshold to deploy blind.")


def threshold_recalibrate_main(dataset: str) -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, f"../temp/{dataset}/output-k3_corrected.json")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {file_path} not found - run 'infer' (and its repair steps) first.")
        return
    except json.JSONDecodeError:
        print(f"Error: {file_path} is not valid JSON.")
        return

    result = recalibrate(data)
    print_recalibration_report(result)
