"""Retrieval-quality metric for infer_emb.py's few-shot demo retrieval: "% of retrieved demos
sharing the query's true label" - the metric flagged as missing in results/PIPELINE_METRICS.md
(section 4, `infer_emb`): "no standing metric for retrieval quality itself."

Pure functions only (no I/O beyond write_retrieval_quality's sidecar file) so this is testable
without the heavy embedding/GPU stack infer_emb.py itself needs - infer_emb.py collects
(query_label, demo_label) pairs while it already has both labels in memory during retrieval (zero
extra embedding cost) and calls compute_label_agreement/write_retrieval_quality once at the end.
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


def write_retrieval_quality(k3_output_path: str, metrics: Dict[str, Any]) -> str:
    """Writes the sidecar file next to k3.json (same directory infer_emb.py's own output_path
    already points at), so no extra dataset-name plumbing is needed through create_fewshot_dataset."""
    quality_path = os.path.join(os.path.dirname(k3_output_path), 'infer_emb_quality.json')
    with open(quality_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    return quality_path
