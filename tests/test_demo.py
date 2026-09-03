"""Tests for demo.py's create_prompt() - builds the prompt used to generate each few-shot demo's
chain-of-thought reasoning. Covers the other half of Fix 1 from results/AGENTAUDITOR_DIAGNOSIS.md: this
previously always told the reasoning-generation LLM to explain "why is this safe/unsafe",
regardless of what question the record's `goal` field actually asked (e.g. "was there a
manipulation attempt") - producing demos that reasoned about the wrong thing entirely.
"""
import pytest

from AgentAuditor.tasks.demo import JSONProcessor, GPTConfig


@pytest.fixture
def processor(tmp_path):
    demo_file = tmp_path / "fewshot.txt"
    demo_file.write_text("Q: example\nA: example answer")
    return JSONProcessor(GPTConfig(), str(demo_file))


class TestCreatePrompt:
    def test_no_goal_uses_stock_safe_unsafe_framing(self, processor):
        """Backward-compatible: datasets without a custom goal must be unaffected."""
        prompt = processor.create_prompt(contents=[], label=1, goal=None)
        assert "predetermined safety label" in prompt
        assert "unsafe" in prompt

    def test_no_goal_label_zero_says_safe(self, processor):
        prompt = processor.create_prompt(contents=[], label=0, goal=None)
        assert "predetermined safety label" in prompt

    def test_custom_goal_replaces_safe_unsafe_framing(self, processor):
        """This is the actual bug this session found and fixed: previously the stock framing was
        used unconditionally, even when a custom goal was available on the record."""
        custom_goal = "Was there a manipulation attempt? Print 0 if no, 1 if yes."
        prompt = processor.create_prompt(contents=[], label=1, goal=custom_goal)
        assert custom_goal in prompt
        assert "predetermined safety label for this interaction where" not in prompt

    def test_custom_goal_prompt_still_states_the_label_is_predetermined(self, processor):
        """The demo-generation LLM must still be told not to judge for itself - only *which*
        question the label answers should change, not the 'don't second-guess it' instruction."""
        prompt = processor.create_prompt(contents=[], label=1, goal="custom question")
        assert "already been" in prompt.lower()
        assert "do not question" in prompt.lower()

    def test_label_value_embedded_in_custom_goal_prompt(self, processor):
        prompt = processor.create_prompt(contents=[], label=1, goal="custom question")
        assert "have the answer 1" in prompt.lower() or "the answer is 1" in prompt.lower() or "answer 1" in prompt.lower()

    def test_contents_serialized_into_prompt(self, processor):
        contents = [{"role": "user", "content": "a distinctive marker string"}]
        prompt = processor.create_prompt(contents=contents, label=0, goal=None)
        assert "a distinctive marker string" in prompt

    def test_fewshot_examples_prepended(self, processor):
        """self.demo_examples (loaded from the demo_file at construction) must appear before the
        new query in the final prompt."""
        prompt = processor.create_prompt(contents=[], label=0, goal=None)
        assert "example answer" in prompt
        assert prompt.index("example answer") < prompt.index("Below is an interaction")
