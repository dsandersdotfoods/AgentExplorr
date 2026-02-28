"""
Tests for Few-Shot Learning
=============================

Test example management, selection strategies, and prompt construction.
"""

from __future__ import annotations

from agentexplorr.prompt_engineering.few_shot import Example, FewShotSelector


class TestFewShotSelector:
    """Tests for FewShotSelector."""

    def setup_method(self) -> None:
        """Set up test examples."""
        self.selector = FewShotSelector()
        self.selector.add_examples([
            Example("I love this!", "Positive", {"type": "sentiment"}),
            Example("Terrible.", "Negative", {"type": "sentiment"}),
            Example("It's okay.", "Neutral", {"type": "sentiment"}),
            Example("Great product!", "Positive", {"type": "sentiment"}),
            Example("Awful service.", "Negative", {"type": "sentiment"}),
        ])

    def test_add_example(self) -> None:
        """Test adding examples to the pool."""
        assert self.selector.size == 5

    def test_select_random(self) -> None:
        """Test random selection returns correct count."""
        selected = self.selector.select_random(3, seed=42)
        assert len(selected) == 3

    def test_select_random_reproducible(self) -> None:
        """Test that random selection with seed is reproducible."""
        a = self.selector.select_random(3, seed=42)
        b = self.selector.select_random(3, seed=42)
        assert [e.input_text for e in a] == [e.input_text for e in b]

    def test_select_by_metadata(self) -> None:
        """Test metadata-based selection."""
        matches = self.selector.select_by_metadata("type", "sentiment")
        assert len(matches) == 5

    def test_format_examples(self) -> None:
        """Test formatting examples into prompt text."""
        examples = [
            Example("I love this!", "Positive"),
            Example("Terrible.", "Negative"),
        ]
        formatted = self.selector.format_examples(examples=examples)
        assert "Input: I love this!" in formatted
        assert "Output: Positive" in formatted

    def test_build_prompt(self) -> None:
        """Test building a complete few-shot prompt."""
        prompt = self.selector.build_prompt(
            instruction="Classify the sentiment:",
            input_text="Pretty good overall",
            n=2,
        )
        assert "Classify the sentiment:" in prompt
        assert "Pretty good overall" in prompt
        assert "Output:" in prompt


class TestChainOfThought:
    """Tests for CoT prompt construction."""

    def test_zero_shot_cot(self) -> None:
        """Test zero-shot CoT prompt."""
        from agentexplorr.prompt_engineering.chain_of_thought import ChainOfThoughtPrompt

        cot = ChainOfThoughtPrompt(task="Solve the math problem.")
        prompt = cot.build_zero_shot("What is 15% of 80?")

        assert "Solve the math problem" in prompt
        assert "15% of 80" in prompt
        assert "step by step" in prompt.lower()

    def test_self_consistency(self) -> None:
        """Test self-consistency generates N identical prompts."""
        from agentexplorr.prompt_engineering.chain_of_thought import ChainOfThoughtPrompt

        cot = ChainOfThoughtPrompt(task="Solve.")
        prompts = cot.build_self_consistency("2+2?", n_paths=5)

        assert len(prompts) == 5
        assert all(p == prompts[0] for p in prompts)
