"""Tests for retrieval_quality.py - the "% of retrieved demos sharing the query's true label"
metric flagged as missing in results/PIPELINE_METRICS.md's `infer_emb` section, plus the raw
content-similarity score distribution (previously computed by find_most_similar_two_stage() and
discarded immediately after use).
"""
import json

from AgentAuditor.tasks.retrieval_quality import (
    compute_label_agreement, compute_similarity_stats, write_retrieval_quality,
)


class TestComputeLabelAgreement:
    def test_no_pairs(self):
        result = compute_label_agreement([])
        assert result == {'total_demo_slots': 0, 'agreement_rate': None, 'query_positive_rate': None}

    def test_perfect_agreement(self):
        pairs = [(0, 0), (0, 0), (1, 1), (1, 1)]
        result = compute_label_agreement(pairs)
        assert result['total_demo_slots'] == 4
        assert result['agreement_rate'] == 1.0
        assert result['query_positive_rate'] == 0.5

    def test_zero_agreement(self):
        pairs = [(0, 1), (0, 1), (1, 0)]
        result = compute_label_agreement(pairs)
        assert result['agreement_rate'] == 0.0

    def test_partial_agreement(self):
        pairs = [(0, 0), (0, 1), (1, 1), (1, 0)]
        result = compute_label_agreement(pairs)
        assert result['agreement_rate'] == 0.5
        assert result['query_positive_rate'] == 0.5

    def test_query_positive_rate_reflects_class_imbalance_not_just_agreement(self):
        # 9 negative queries, 1 positive - agreement rate alone can't distinguish "retrieval is
        # label-blind on an imbalanced set" from "retrieval genuinely tracks label" without this.
        pairs = [(0, 0)] * 9 + [(1, 1)]
        result = compute_label_agreement(pairs)
        assert result['agreement_rate'] == 1.0
        assert result['query_positive_rate'] == 0.1


class TestComputeSimilarityStats:
    def test_empty_scores_returns_none_fields_not_zeros(self):
        """0.0 would misleadingly read as 'nothing was similar' rather than 'no data' - must be
        distinguishable."""
        result = compute_similarity_stats([])
        assert result == {'count': 0, 'mean': None, 'min': None, 'max': None, 'median': None}

    def test_basic_stats(self):
        scores = [0.2, 0.4, 0.6, 0.8, 1.0]
        result = compute_similarity_stats(scores)
        assert result['count'] == 5
        assert result['mean'] == 0.6
        assert result['min'] == 0.2
        assert result['max'] == 1.0
        assert result['median'] == 0.6

    def test_median_even_count_averages_middle_two(self):
        scores = [0.1, 0.2, 0.3, 0.4]
        result = compute_similarity_stats(scores)
        assert result['median'] == 0.25

    def test_single_score(self):
        result = compute_similarity_stats([0.77])
        assert result['count'] == 1
        assert result['mean'] == result['min'] == result['max'] == result['median'] == 0.77


class TestWriteRetrievalQuality:
    def test_writes_sidecar_next_to_k3_json(self, tmp_path):
        k3_path = str(tmp_path / "k3.json")
        metrics = {'total_demo_slots': 30, 'agreement_rate': 0.73, 'query_positive_rate': 0.4}
        quality_path = write_retrieval_quality(k3_path, metrics)
        assert quality_path == str(tmp_path / "infer_emb_quality.json")
        with open(quality_path) as f:
            assert json.load(f) == metrics
