"""Brier score and Expected Calibration Error (ECE) - the two calibration metrics named alongside
AUROC in CLAUDE.md's own description of what this baseline needs to share with the TRACES
comparison ("shared metrics (AUROC/ECE/Brier)"), but never actually computed anywhere in this repo
until now - eval.py only had AUROC/AUPRC.

Both answer a different question than AUROC does. AUROC asks "does the score rank positives above
negatives" (threshold-independent, ignores whether the actual numbers mean anything). Brier/ECE
ask "when the model says 90%, is it right about 90% of the time" - whether the stated confidence
is trustworthy as a probability, not just useful for ranking.

Pure functions only, no I/O - wired into eval.py's existing ranking-metrics section.
"""
from typing import List, Optional, Tuple


def compute_brier_score(true_labels: List[int], positive_class_scores: List[float]) -> Optional[float]:
    """Mean squared error between the predicted positive-class probability and the actual outcome
    (1 if the record is truly positive, else 0). Lower is better - 0 is a perfect prediction, 0.25
    is what a constant 0.5 guess scores, 1.0 is a maximally confident wrong prediction every time.

    Uses the SAME positive-class scores as AUROC/AUPRC (eval.py's extract_confidence), since Brier
    score is inherently a single-class-probability metric - unlike ECE, it doesn't need the raw
    "confidence in own prediction" value.
    """
    if not true_labels or len(true_labels) != len(positive_class_scores):
        return None
    n = len(true_labels)
    return sum((s - t) ** 2 for s, t in zip(positive_class_scores, true_labels)) / n


def compute_ece(confidences: List[float], correctness: List[bool], num_bins: int = 10) -> Optional[Tuple[float, List[dict]]]:
    """Expected Calibration Error: bins predictions by the model's stated confidence in its own
    verdict (NOT the positive-class score - use eval.py's extract_raw_confidence for this), and
    measures how far the average confidence in each bin is from the actual accuracy in that bin,
    weighted by bin size.

    confidences and correctness must be the SAME LENGTH and index-aligned: confidences[i] is how
    confident the model was in whatever it predicted for item i, correctness[i] is whether that
    prediction actually matched the true label.

    Returns (ece, bin_details) where bin_details lists each non-empty bin's range, count, mean
    confidence, and accuracy - useful for seeing WHERE calibration breaks down (e.g. overconfident
    in the 90-100% bin specifically), not just the single aggregate number.

    Returns None if there's nothing to bin (empty input) rather than a misleading 0.0.
    """
    if not confidences or len(confidences) != len(correctness):
        return None

    n = len(confidences)
    bin_width = 1.0 / num_bins
    bins: List[List[int]] = [[] for _ in range(num_bins)]
    for i, c in enumerate(confidences):
        bin_idx = min(int(c / bin_width), num_bins - 1)
        bins[bin_idx].append(i)

    ece = 0.0
    bin_details = []
    for bin_idx, indices in enumerate(bins):
        if not indices:
            continue
        bin_conf = sum(confidences[i] for i in indices) / len(indices)
        bin_acc = sum(1 for i in indices if correctness[i]) / len(indices)
        weight = len(indices) / n
        ece += weight * abs(bin_acc - bin_conf)
        bin_details.append({
            'range': [round(bin_idx * bin_width, 2), round((bin_idx + 1) * bin_width, 2)],
            'count': len(indices),
            'mean_confidence': bin_conf,
            'accuracy': bin_acc,
        })

    return ece, bin_details
