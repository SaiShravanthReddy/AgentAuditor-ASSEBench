"""Tests for infer.py's combine_to_prompt() - the judge-facing prompt builder. Covers Fix 1 from
results/AGENTAUDITOR_DIAGNOSIS.md: this previously ignored the record's `goal` field entirely, silently
using the stock safe/unsafe framing for every comparison including ones with a genuinely different
question (e.g. FinVault's "was there a manipulation attempt" comparisons).
"""
from AgentAuditor.tasks.infer import combine_to_prompt, DEFAULT_GOAL, CONFIDENCE_INSTRUCTION


class TestCombineToPrompt:
    def test_no_goal_uses_default(self):
        """Backward-compatible: datasets without a custom goal must be unaffected."""
        prompt = combine_to_prompt(contents=[{"role": "user", "content": "hi"}], fewshot_demos=[], goal=None)
        assert DEFAULT_GOAL in prompt

    def test_custom_goal_used_instead_of_default(self):
        custom_goal = "Was there a manipulation attempt? Print 0 or 1."
        prompt = combine_to_prompt(contents=[], fewshot_demos=[], goal=custom_goal)
        assert custom_goal in prompt
        assert DEFAULT_GOAL not in prompt

    def test_confidence_instruction_always_present(self):
        """Added per the team's confirmed self-reported-confidence decision - must be present
        regardless of which goal is used."""
        prompt_default = combine_to_prompt(contents=[], fewshot_demos=[], goal=None)
        prompt_custom = combine_to_prompt(contents=[], fewshot_demos=[], goal="custom question")
        assert CONFIDENCE_INSTRUCTION in prompt_default
        assert CONFIDENCE_INSTRUCTION in prompt_custom

    def test_fewshot_demos_included(self):
        demos = [{"Q": "example question", "A": "example answer"}]
        prompt = combine_to_prompt(contents=[], fewshot_demos=demos, goal=None)
        assert "example question" in prompt
        assert "example answer" in prompt

    def test_contents_serialized_into_prompt(self):
        contents = [{"role": "user", "content": "a distinctive marker string"}]
        prompt = combine_to_prompt(contents=contents, fewshot_demos=[], goal=None)
        assert "a distinctive marker string" in prompt

    def test_empty_string_goal_falls_back_to_default(self):
        """Falsy-but-present goal (empty string) should behave like None, not like a real goal -
        `goal or DEFAULT_GOAL` in the implementation relies on this."""
        prompt = combine_to_prompt(contents=[], fewshot_demos=[], goal="")
        assert DEFAULT_GOAL in prompt
