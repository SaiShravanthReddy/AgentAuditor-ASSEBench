"""Retrieval-quality metrics for infer_emb.py's few-shot demo retrieval:
- "% of retrieved demos sharing the query's true label" - flagged as missing in
  results/PIPELINE_METRICS.md (section 4, `infer_emb`): "no standing metric for retrieval quality
  itself."
- The actual content-similarity score of each retrieved demo - find_most_similar_two_stage()
  already computes this per retrieval (it's the ranking criterion), but the score was previously
  discarded immediately after use ("The score is not used here, only the ID to fetch raw data").
  Tracking its distribution catches a different failure mode than label agreement alone: high
  label agreement with LOW similarity scores would mean retrieval is coasting on class imbalance
  (most things share the majority label anyway), not genuinely finding similar content.

Pure functions only (no I/O beyond write_retrieval_quality's sidecar file) so this is testable
without the heavy embedding/GPU stack infer_emb.py itself needs - infer_emb.py collects
(query_label, demo_label) pairs and the raw similarity scores while it already has both in memory
during retrieval (zero extra embedding cost) and calls compute_label_agreement/
compute_similarity_stats/write_retrieval_quality once at the end.
"""
import json
import os
from typing import Any, Dict, List, Optional, Tuple


def compute_label_agreement(pairs: List[Tuple[int, int]]) -> Dict[str, Any]:
    """pairs: (query_true_label, retrieved_demo_true_label) for every demo slot actually filled.
    Agreement rate is the fraction of retrieved demos whose true label matches the query they were
    retrieved for - a demo pool with no label signal at all would land near the dataset's positive
    rate (not 50%), so compare against that, not a flat 0.5, when judging whether retrieval is
    doing better than chance.
    """
    total = len(pairs)
    if total == 0:
        return {'total_demo_slots': 0, 'agreement_rate': None, 'query_positive_rate': None}

    agreement = sum(1 for q, d in pairs if q == d) / total
    query_positive_rate = sum(1 for q, _ in pairs if q == 1) / total

    return {
        'total_demo_slots': total,
        'agreement_rate': agreement,
        'query_positive_rate': query_positive_rate,
    }


def compute_similarity_stats(scores: List[float]) -> Dict[str, Any]:
    """Summary stats for the raw content-similarity scores of every retrieved demo slot. Returns
    None-filled stats (not zeros) for empty input, since 0.0 would misleadingly read as "retrieval
    found nothing similar" rather than "no data available"."""
    if not scores:
        return {'count': 0, 'mean': None, 'min': None, 'max': None, 'median': None}

    n = len(scores)
    sorted_scores = sorted(scores)
    mid = n // 2
    median = sorted_scores[mid] if n % 2 == 1 else (sorted_scores[mid - 1] + sorted_scores[mid]) / 2

    return {
        'count': n,
        'mean': sum(scores) / n,
        'min': sorted_scores[0],
        'max': sorted_scores[-1],
        'median': median,
    }


def write_retrieval_quality(k3_output_path: str, metrics: Dict[str, Any]) -> str:
    """Writes the sidecar file next to k3.json (same directory infer_emb.py's own output_path
    already points at), so no extra dataset-name plumbing is needed through create_fewshot_dataset."""
    quality_path = os.path.join(os.path.dirname(k3_output_path), 'infer_emb_quality.json')
    with open(quality_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    return quality_path
