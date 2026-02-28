"""
Tests for Chain of Thought Prompting
======================================

Tests the ChainOfThoughtPrompt builder for zero-shot, few-shot,
and self-consistency prompt generation.
"""

from __future__ import annotations

from agentexplorr.prompt_engineering.chain_of_thought import (
    ChainOfThoughtPrompt,
    ReasoningStep,
)


class TestReasoningStep:
    def test_create(self) -> None:
        step = ReasoningStep(step_number=1, description="Convert", content="15% = 0.15")
        assert step.step_number == 1
        assert step.description == "Convert"
        assert step.content == "15% = 0.15"


class TestChainOfThoughtPrompt:
    def test_zero_shot_basic(self) -> None:
        cot = ChainOfThoughtPrompt(task="Solve the math problem.")
        prompt = cot.build_zero_shot("What is 2 + 2?")

        assert "Solve the math problem." in prompt
        assert "What is 2 + 2?" in prompt
        assert "Let's think step by step:" in prompt

    def test_zero_shot_custom_trigger(self) -> None:
        cot = ChainOfThoughtPrompt(
            task="Reason carefully.",
            cot_trigger="Let me work through this:",
        )
        prompt = cot.build_zero_shot("Why is the sky blue?")
        assert "Let me work through this:" in prompt

    def test_few_shot_with_examples(self) -> None:
        cot = ChainOfThoughtPrompt(task="Solve the math problem.")
        cot.add_example(
            question="What is 15% of 80?",
            reasoning=[
                ReasoningStep(1, "Convert percentage", "15% = 0.15"),
                ReasoningStep(2, "Multiply", "0.15 * 80 = 12"),
            ],
            answer="12",
        )

        prompt = cot.build_few_shot("What is 20% of 150?")

        # Check that the example is included
        assert "15% of 80" in prompt
        assert "0.15" in prompt
        assert "Answer: 12" in prompt
        # Check the new question is at the end
        assert "20% of 150?" in prompt

    def test_few_shot_multiple_examples(self) -> None:
        cot = ChainOfThoughtPrompt(task="Math solver")
        cot.add_example(
            question="Q1",
            reasoning=[ReasoningStep(1, "step", "content")],
            answer="A1",
        )
        cot.add_example(
            question="Q2",
            reasoning=[ReasoningStep(1, "step", "content")],
            answer="A2",
        )

        prompt = cot.build_few_shot("Q3")
        assert "Q1" in prompt
        assert "Q2" in prompt
        assert "A1" in prompt
        assert "A2" in prompt
        assert prompt.endswith("Let's think step by step:")

    def test_self_consistency_returns_n_prompts(self) -> None:
        cot = ChainOfThoughtPrompt(task="Solve this.")
        prompts = cot.build_self_consistency("What is 7 * 8?", n_paths=5)

        assert len(prompts) == 5
        # All prompts should be identical (diversity from temperature)
        assert all(p == prompts[0] for p in prompts)

    def test_self_consistency_uses_few_shot_if_examples(self) -> None:
        cot = ChainOfThoughtPrompt(task="Math")
        cot.add_example(
            question="2+2?",
            reasoning=[ReasoningStep(1, "add", "2+2=4")],
            answer="4",
        )
        prompts = cot.build_self_consistency("3+3?", n_paths=3)

        # Should include the example
        assert "2+2?" in prompts[0]
        assert "Answer: 4" in prompts[0]

    def test_self_consistency_zero_shot_when_no_examples(self) -> None:
        cot = ChainOfThoughtPrompt(task="Reason carefully.")
        prompts = cot.build_self_consistency("Why?", n_paths=2)

        # Should not have "Answer:" since there are no examples
        assert "Answer:" not in prompts[0]
        assert "Let's think step by step:" in prompts[0]

    def test_add_example_stores_correctly(self) -> None:
        cot = ChainOfThoughtPrompt(task="Test")
        assert len(cot.examples) == 0

        cot.add_example(
            question="Q",
            reasoning=[ReasoningStep(1, "desc", "content")],
            answer="A",
        )
        assert len(cot.examples) == 1
        assert cot.examples[0]["question"] == "Q"
        assert cot.examples[0]["answer"] == "A"
