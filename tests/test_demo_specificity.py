"""Tests for demo_specificity.py - catching generic/templated demo reasoning that isn't actually
grounded in the real conversation, the same failure pattern found in infer_llm_repair.py's
correction-LLM bug (see that module's git history), applied here as a standing check.
"""
from AgentAuditor.tasks.demo_specificity import (
    compute_demo_specificity, compute_specificity_score,
)


class TestComputeSpecificityScore:
    def test_reasoning_grounded_in_conversation_scores_high(self):
        conversation = "The user PERSON-000016 requested a loan of 1.5 million yuan with credit score 650."
        reasoning = "PERSON-000016 requested a loan of 1.5 million yuan, which is a normal banking request."
        score = compute_specificity_score(reasoning, conversation)
        assert score > 0.5

    def test_real_templated_filler_incident_scores_low(self):
        """The exact real-world failure case: infer_llm_repair.py's correction LLM produced this
        text for a record about financial loan approval - generic, references nothing specific."""
        conversation = (
            "The user requested approval of a 1.5 million yuan loan for PERSON-000001, who has a "
            "monthly income of 15,000 yuan and a credit score of 650."
        )
        reasoning = (
            "Received input JSON with verdict and confidence. Verdict '0' indicates safe content. "
            "Confidence 0.99 suggests high certainty. No evidence of unsafe content."
        )
        score = compute_specificity_score(reasoning, conversation)
        assert score < 0.3

    def test_empty_reasoning_returns_none(self):
        assert compute_specificity_score("", "some conversation text") is None

    def test_pure_stopword_reasoning_returns_none(self):
        assert compute_specificity_score("the is a of in on at to for", "some text") is None


class TestComputeDemoSpecificity:
    def _valid_demo(self, item_id, reasoning_extra, contents_text):
        return {
            'id': item_id,
            'contents': [[{'role': 'user', 'content': contents_text}]],
            'chain_of_thought': {
                'chain_of_thought': {
                    'Start': reasoning_extra,
                    'Output': '0 (safe)',
                }
            },
        }

    def test_scores_only_validated_demos(self):
        data = [
            self._valid_demo('good-1', 'discussing loan approval for customer account', 'loan approval for customer account'),
            {'id': 'malformed', 'chain_of_thought': 'not nested', 'contents': [[]]},
        ]
        result = compute_demo_specificity(data)
        assert result['num_scored'] == 1

    def test_flags_low_specificity_demos_by_id(self):
        data = [
            self._valid_demo(
                'generic-filler',
                "Received input JSON with verdict and confidence. Verdict indicates safe content.",
                "The user PERSON-000016 requested a loan of 1.5 million yuan with credit score 650.",
            ),
        ]
        result = compute_demo_specificity(data)
        assert result['low_specificity_count'] == 1
        assert 'generic-filler' in result['low_specificity_ids']

    def test_empty_input(self):
        result = compute_demo_specificity([])
        assert result == {'num_scored': 0, 'mean_specificity': None, 'low_specificity_count': 0,
                           'low_specificity_ids': []}
