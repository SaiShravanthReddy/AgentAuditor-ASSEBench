"""Tests for cluster_quality.py - the silhouette/coverage metric flagged as missing in
results/PIPELINE_METRICS.md's `cluster` section.
"""
import json

import numpy as np
import pytest

from AgentAuditor.tasks.cluster_quality import compute_cluster_quality, write_cluster_quality


class TestComputeClusterQuality:
    def test_empty_input(self):
        result = compute_cluster_quality(np.array([]).reshape(0, 4), np.array([]))
        assert result == {'total_records': 0, 'num_valid_records': 0, 'coverage': None,
                           'num_clusters': 0, 'silhouette_score': None}

    def test_full_coverage_two_well_separated_clusters(self):
        # 4 points in 2 obviously-separated clusters -> coverage=1.0, silhouette close to 1.0
        embeddings = np.array([[0.0, 0.0], [0.1, 0.1], [10.0, 10.0], [10.1, 10.1]])
        labels = np.array([0, 0, 1, 1])
        result = compute_cluster_quality(embeddings, labels)
        assert result['total_records'] == 4
        assert result['num_valid_records'] == 4
        assert result['coverage'] == 1.0
        assert result['num_clusters'] == 2
        assert result['silhouette_score'] is not None
        assert result['silhouette_score'] > 0.9

    def test_partial_coverage_with_unassigned_noise_points(self):
        # FINCH marks unassigned points as -1 - coverage should exclude them from the denominator
        # of "assigned" but still count them in total_records.
        embeddings = np.array([[0.0, 0.0], [0.1, 0.1], [10.0, 10.0], [10.1, 10.1], [5.0, 5.0]])
        labels = np.array([0, 0, 1, 1, -1])
        result = compute_cluster_quality(embeddings, labels)
        assert result['total_records'] == 5
        assert result['num_valid_records'] == 4
        assert result['coverage'] == 0.8
        assert result['num_clusters'] == 2
        assert result['silhouette_score'] is not None

    def test_single_cluster_silhouette_undefined(self):
        embeddings = np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]])
        labels = np.array([0, 0, 0])
        result = compute_cluster_quality(embeddings, labels)
        assert result['coverage'] == 1.0
        assert result['num_clusters'] == 1
        assert result['silhouette_score'] is None

    def test_all_noise_no_clusters(self):
        embeddings = np.array([[0.0, 0.0], [1.0, 1.0]])
        labels = np.array([-1, -1])
        result = compute_cluster_quality(embeddings, labels)
        assert result['num_valid_records'] == 0
        assert result['coverage'] == 0.0
        assert result['num_clusters'] == 0
        assert result['silhouette_score'] is None


class TestWriteClusterQuality:
    def test_writes_sidecar_next_to_cluster_json(self, tmp_path):
        output_path = str(tmp_path / "cluster.json")
        metrics = {'total_records': 10, 'num_valid_records': 9, 'coverage': 0.9,
                   'num_clusters': 3, 'silhouette_score': 0.42}
        quality_path = write_cluster_quality(output_path, metrics)
        assert quality_path == str(tmp_path / "cluster_quality.json")
        with open(quality_path) as f:
            assert json.load(f) == metrics
