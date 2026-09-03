"""Tests for infer_emb.py's retrieval logic. Covers Fix 1 (goal-aware demo Q text) and Fix 3
(self-exclusion) from results/AGENTAUDITOR_DIAGNOSIS.md - both confirmed happening on real production data
this session (a CNFinBench query retrieved itself as its own demo; FinVault Q2's demos asked the
wrong question). Does NOT load the real embedding model (network/GPU-dependent) - a bare subclass
skips EmbeddingProcessor's model-loading __init__ entirely, since none of this logic touches the
model itself.
"""
import numpy as np
import pytest

from AgentAuditor.tasks.infer_emb import EmbeddingProcessor


class BareEmbeddingProcessor(EmbeddingProcessor):
    """Skips the real (network/GPU-dependent) model loading - nothing under test here needs it."""
    def __init__(self):
        pass


@pytest.fixture
def processor():
    return BareEmbeddingProcessor()


def vec(x, y):
    return np.array([x, y], dtype=np.float32)


class TestFindMostSimilarTwoStage:
    def test_returns_top_k_by_content_similarity(self, processor):
        query = {'Ec': vec(1.0, 0.0)}
        refs = {
            'closest': {'Ec': vec(1.0, 0.0)},
            'middle': {'Ec': vec(0.7, 0.3)},
            'farthest': {'Ec': vec(0.0, 1.0)},
        }
        results = processor.find_most_similar_two_stage(query, refs, k=2, top_n_content=2, params=[1, 1, 1])
        ids = [r[0] for r in results]
        assert ids[0] == 'closest'
        assert 'farthest' not in ids
        assert len(ids) == 2

    def test_exclude_id_prevents_self_retrieval(self, processor):
        """The exact bug found in production: a cluster representative retrieving itself as its
        own demo when it's also being judged as a query (near-perfect content similarity, since
        it IS the same record). exclude_id must keep it out of the candidate pool entirely."""
        query_id = 'item_A'
        query = {'Ec': vec(1.0, 0.0)}
        refs = {
            'item_A': {'Ec': vec(1.0, 0.0)},  # identical to query - would win without exclusion
            'item_B': {'Ec': vec(0.9, 0.1)},
            'item_C': {'Ec': vec(0.5, 0.5)},
        }
        results = processor.find_most_similar_two_stage(
            query, refs, k=2, top_n_content=2, params=[1, 1, 1], exclude_id=query_id
        )
        ids = [r[0] for r in results]
        assert 'item_A' not in ids
        assert 'item_B' in ids  # next-best candidate fills the freed slot

    def test_no_exclude_id_is_backward_compatible(self, processor):
        """Default behavior (exclude_id not passed) must be unchanged."""
        query = {'Ec': vec(1.0, 0.0)}
        refs = {'item_A': {'Ec': vec(1.0, 0.0)}, 'item_B': {'Ec': vec(0.0, 1.0)}}
        results = processor.find_most_similar_two_stage(query, refs, k=1, top_n_content=1, params=[1, 1, 1])
        assert results[0][0] == 'item_A'

    def test_empty_reference_embeddings_returns_empty(self, processor):
        assert processor.find_most_similar_two_stage({'Ec': vec(1, 0)}, {}, k=3, top_n_content=3, params=[1, 1, 1]) == []

    def test_missing_query_embedding_returns_empty(self, processor):
        refs = {'item_A': {'Ec': vec(1.0, 0.0)}}
        assert processor.find_most_similar_two_stage({}, refs, k=1, top_n_content=1, params=[1, 1, 1]) == []

    def test_k_zero_returns_empty(self, processor):
        refs = {'item_A': {'Ec': vec(1.0, 0.0)}}
        assert processor.find_most_similar_two_stage({'Ec': vec(1, 0)}, refs, k=0, top_n_content=3, params=[1, 1, 1]) == []


class TestGenerateFewshotDemo:
    def test_no_goal_uses_default_goal_text(self, processor):
        demo = processor.generate_fewshot_demo(content=[], chain_of_thought=None, goal=None)
        assert processor.DEFAULT_GOAL in demo['Q']

    def test_custom_goal_replaces_default(self, processor):
        """The other half of Fix 1: the demo's own Q text (shown to the judge alongside its
        answer) must ask the same question the actual judge prompt asks, not always the stock
        safe/unsafe framing."""
        custom_goal = "Was there a manipulation attempt? Print 0 or 1."
        demo = processor.generate_fewshot_demo(content=[], chain_of_thought=None, goal=custom_goal)
        assert custom_goal in demo['Q']
        assert processor.DEFAULT_GOAL not in demo['Q']

    def test_valid_chain_of_thought_used_as_answer(self, processor):
        cot = {"chain_of_thought": {"Start": "...", "Output": "1 (unsafe)"}}
        demo = processor.generate_fewshot_demo(content=[], chain_of_thought=cot, goal=None)
        assert "1 (unsafe)" in demo['A']

    def test_missing_chain_of_thought_produces_empty_structure(self, processor):
        """Not a crash - generate_fewshot_demo's own fallback for missing/invalid CoT. (Whether
        this record should have been a retrieval candidate at all is infer_emb's Fix 2 filtering,
        tested separately in test_demo_repair.py - this test is just about what this function
        does with whatever CoT it's handed.)"""
        demo = processor.generate_fewshot_demo(content=[], chain_of_thought=None, goal=None)
        assert '"chain_of_thought": {}' in demo['A']

    def test_content_serialized_into_question(self, processor):
        content = [{"role": "user", "content": "a distinctive marker string"}]
        demo = processor.generate_fewshot_demo(content=content, chain_of_thought=None, goal=None)
        assert "a distinctive marker string" in demo['Q']
