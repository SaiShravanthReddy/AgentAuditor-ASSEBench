"""Tests for cnfinbench_detect_leakage.py - self-leakage detection (a cluster representative
retrieving its own dialogue as a few-shot demo). Despite the cnfinbench_ name prefix this is
dataset-agnostic (see CLAUDE.md) - now wired into infer_emb.py as a standing metric rather than a
script someone has to remember to run manually.
"""
import json

from CNFinBench.cnfinbench_detect_leakage import (
    genuinely_leaked_ids, genuinely_leaked_ids_from_data,
)

LONG_TEXT = "x" * 60  # > MIN_TURN_LEN (40), so it's eligible to count as a match


class TestGenuinelyLeakedIdsFromData:
    def test_item_retrieving_its_own_dialogue_is_flagged(self):
        own_turn_text = LONG_TEXT + " this is the actual message content for this turn"
        data = [{
            'id': 'item-1',
            'contents': [[{'role': 'user', 'content': own_turn_text}]],
            'fewshot_demos': [{'Q': f"Some prefix text\n{own_turn_text}\nmore text"}],
        }]
        leaked = genuinely_leaked_ids_from_data(data)
        assert leaked == {'item-1'}

    def test_item_with_genuinely_different_demos_not_flagged(self):
        own_turn_text = LONG_TEXT + " this is the actual message content for this turn"
        data = [{
            'id': 'item-1',
            'contents': [[{'role': 'user', 'content': own_turn_text}]],
            'fewshot_demos': [{'Q': "Completely unrelated demo content about something else entirely"}],
        }]
        leaked = genuinely_leaked_ids_from_data(data)
        assert leaked == set()

    def test_short_turns_excluded_from_matching(self):
        """Turns under MIN_TURN_LEN don't count - too short to be a meaningful match, would cause
        false positives on generic short turns like "Yes" or "OK"."""
        short_text = "OK"
        data = [{
            'id': 'item-1',
            'contents': [[{'role': 'user', 'content': short_text}]],
            'fewshot_demos': [{'Q': short_text}],
        }]
        leaked = genuinely_leaked_ids_from_data(data)
        assert leaked == set()

    def test_no_fewshot_demos_not_flagged(self):
        data = [{'id': 'item-1', 'contents': [[{'role': 'user', 'content': LONG_TEXT}]], 'fewshot_demos': []}]
        assert genuinely_leaked_ids_from_data(data) == set()

    def test_empty_contents_not_flagged(self):
        data = [{'id': 'item-1', 'contents': [], 'fewshot_demos': [{'Q': 'anything'}]}]
        assert genuinely_leaked_ids_from_data(data) == set()

    def test_empty_data(self):
        assert genuinely_leaked_ids_from_data([]) == set()


class TestGenuinelyLeakedIdsFileWrapper:
    def test_reads_file_and_delegates_to_data_function(self, tmp_path):
        own_turn_text = LONG_TEXT + " this is the actual message content for this turn"
        data = [{
            'id': 'item-1',
            'contents': [[{'role': 'user', 'content': own_turn_text}]],
            'fewshot_demos': [{'Q': f"{own_turn_text}"}],
        }]
        f = tmp_path / "k3.json"
        f.write_text(json.dumps(data))
        assert genuinely_leaked_ids(str(f)) == {'item-1'}
