"""Quality metric for cluster.py's FINCH clustering: silhouette score + coverage, flagged as
missing in results/PIPELINE_METRICS.md (section 2, `cluster`): "no silhouette score, no coverage
check has been run. This is a real gap, not a zero score."

Pure function on already-computed embeddings/labels (compute_cluster_quality) so this is testable
without the heavy embedding/GPU stack cluster.py itself needs - cluster.py calls it once, right
after FINCH clustering finishes, using the same embeddings/labels array already in memory (zero
extra embedding-recomputation cost).
"""
import json
import os
from typing import Any, Dict, Optional

import numpy as np


def compute_cluster_quality(embeddings: np.ndarray, cluster_labels: np.ndarray) -> Dict[str, Any]:
    """coverage: fraction of records FINCH assigned to a real cluster (label >= 0) rather than
    left unclustered (label -1, FINCH's noise/unassigned marker).
    silhouette_score: sklearn's standard cluster-separation metric (-1 worst, +1 best), computed
    only over the covered records - undefined (and left None) with fewer than 2 clusters or fewer
    than 2 covered points, matching sklearn's own precondition.
    """
    cluster_labels = np.asarray(cluster_labels)
    total = len(cluster_labels)
    if total == 0:
        return {'total_records': 0, 'num_valid_records': 0, 'coverage': None,
                'num_clusters': 0, 'silhouette_score': None}

    valid_mask = cluster_labels >= 0
    num_valid = int(valid_mask.sum())
    coverage = num_valid / total
    num_clusters = len(set(cluster_labels[valid_mask].tolist())) if num_valid else 0

    silhouette = None
    if num_valid >= 2 and num_clusters >= 2:
        from sklearn.metrics import silhouette_score
        try:
            silhouette = float(silhouette_score(embeddings[valid_mask], cluster_labels[valid_mask]))
        except ValueError:
            # e.g. a cluster with only 1 member after masking - sklearn requires every cluster to
            # have at least 2 members for a well-defined silhouette score.
            silhouette = None

    return {
        'total_records': total,
        'num_valid_records': num_valid,
        'coverage': coverage,
        'num_clusters': num_clusters,
        'silhouette_score': silhouette,
    }


def write_cluster_quality(output_path: str, metrics: Dict[str, Any]) -> str:
    """Writes the sidecar file next to cluster.json (same directory cluster.py's own output_path
    already points at)."""
    quality_path = os.path.join(os.path.dirname(output_path), 'cluster_quality.json')
    with open(quality_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    return quality_path
