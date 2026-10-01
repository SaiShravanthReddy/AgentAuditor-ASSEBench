"""Tests for category_breakdown.py - per-scenario-type accuracy, previously only done ad-hoc by
manually reading records (e.g. the MT_App false-negative clustering in AGENTAUDITOR_DIAGNOSIS.md).
"""
from AgentAuditor.tasks.category_breakdown import (
    compute_category_breakdown, extract_category_from_id,
)


class TestExtractCategoryFromId:
    def test_cnfinbench_scenario_tokens(self):
        assert extract_category_from_id('harmless-MT_App-12') == 'MT_App'
        assert extract_category_from_id('harmful-MT_Cog-5') == 'MT_Cog'
        assert extract_category_from_id('harmless-unblocked-MT_Inter-22') == 'MT_Inter'

    def test_finvault_category_tokens(self):
        assert extract_category_from_id(
            'finvault-v5-fixed-01-attack-case_fixed_0089'
        ) == '01-attack-case'
        assert extract_category_from_id(
            'finvault-v3-fixed-08-attack-case_fixed_0353'
        ) == '08-attack-case'

    def test_unrecognized_pattern_returns_none(self):
        assert extract_category_from_id('rjudge-12345') is None
        assert extract_category_from_id('some-random-id') is None

    def test_non_string_returns_none(self):
        assert extract_category_from_id(12345) is None
        assert extract_category_from_id(None) is None


class TestComputeCategoryBreakdown:
    def test_groups_and_scores_per_category(self):
        ids =        ['harmless-MT_App-1', 'harmless-MT_App-2', 'harmless-MT_Cog-1']
        true =       [0, 1, 0]
        predicted =  [0, 0, 0]  # MT_App: 1/2 correct, MT_Cog: 1/1 correct
        breakdown = compute_category_breakdown(ids, true, predicted)

        assert set(breakdown.keys()) == {'MT_App', 'MT_Cog'}
        assert breakdown['MT_App']['count'] == 2
        assert breakdown['MT_App']['accuracy'] == 0.5
        assert breakdown['MT_Cog']['count'] == 1
        assert breakdown['MT_Cog']['accuracy'] == 1.0

    def test_unmatched_ids_grouped_as_uncategorized_not_dropped(self):
        ids = ['rjudge-1', 'rjudge-2']
        true = [0, 1]
        predicted = [0, 1]
        breakdown = compute_category_breakdown(ids, true, predicted)
        assert set(breakdown.keys()) == {'uncategorized'}
        assert breakdown['uncategorized']['count'] == 2

    def test_counts_sum_to_input_length(self):
        ids = ['harmless-MT_App-1', 'rjudge-1', 'harmful-MT_Inter-3']
        true = [0, 1, 1]
        predicted = [0, 1, 0]
        breakdown = compute_category_breakdown(ids, true, predicted)
        assert sum(b['count'] for b in breakdown.values()) == 3

    def test_empty_input(self):
        assert compute_category_breakdown([], [], []) == {}
