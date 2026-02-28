"""
Chain of Thought (CoT) Prompting
=================================

WHAT IS CHAIN OF THOUGHT?
  CoT prompting asks the model to "think step by step" before giving
  a final answer. This dramatically improves performance on tasks
  requiring reasoning: math, logic, multi-step analysis, etc.

THE BREAKTHROUGH:
  Google's 2022 paper showed that simply adding "Let's think step by step"
  to a prompt improved GPT-3's accuracy on math problems from ~18% to ~79%.
  This was one of the most impactful discoveries in prompt engineering.

CoT VARIANTS:
  1. **Zero-shot CoT** — Just add "Let's think step by step"
  2. **Few-shot CoT** — Include examples WITH reasoning chains
  3. **Self-consistency** — Generate multiple reasoning paths, take majority vote
  4. **Tree of Thought** — Explore multiple reasoning branches (advanced)

WHEN TO USE CoT:
  ✅ Math and arithmetic
  ✅ Multi-step reasoning
  ✅ Complex classification with explanations
  ✅ Code generation (plan before coding)
  ❌ Simple factual questions ("What is the capital of France?")
  ❌ Creative writing (may over-constrain the output)

LEARNING RESOURCES:
  - Original CoT paper: https://arxiv.org/abs/2201.11903
  - Zero-shot CoT: https://arxiv.org/abs/2205.11916
  - Self-consistency: https://arxiv.org/abs/2203.11171
  - Prompting Guide: https://www.promptingguide.ai/techniques/cot
  - VIDEO: "Chain of Thought Explained" — https://www.youtube.com/watch?v=H4gZd4BCrDQ
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ReasoningStep:
    """A single step in a chain of thought.

    Attributes:
        step_number: Position in the reasoning chain (1-indexed).
        description: What this step does.
        content: The actual reasoning for this step.
    """

    step_number: int
    description: str
    content: str


@dataclass
class ChainOfThoughtPrompt:
    """Construct Chain of Thought prompts for step-by-step reasoning.

    This class helps you build CoT prompts with optional examples
    that include reasoning chains.

    Example (Zero-shot CoT):
        >>> cot = ChainOfThoughtPrompt(
        ...     task="Solve the following math problem.",
        ...     cot_trigger="Let's solve this step by step:"
        ... )
        >>> prompt = cot.build_zero_shot("If a train travels 60 mph for 2.5 hours, how far does it go?")
        >>> print(prompt)

    Example (Few-shot CoT):
        >>> cot = ChainOfThoughtPrompt(task="Solve the math problem.")
        >>> cot.add_example(
        ...     question="What is 15% of 80?",
        ...     reasoning=[
        ...         ReasoningStep(1, "Convert percentage", "15% = 0.15"),
        ...         ReasoningStep(2, "Multiply", "0.15 × 80 = 12"),
        ...     ],
        ...     answer="12"
        ... )
        >>> prompt = cot.build_few_shot("What is 20% of 150?")
    """

    task: str
    cot_trigger: str = "Let's think step by step:"
    examples: list[dict[str, object]] = field(default_factory=list)

    def add_example(
        self,
        question: str,
        reasoning: list[ReasoningStep],
        answer: str,
    ) -> None:
        """Add a few-shot example with explicit reasoning chain.

        WHY INCLUDE REASONING?
          When the model sees examples with step-by-step reasoning,
          it learns to produce similar reasoning for new questions.
          This is the key insight of few-shot CoT.

        Args:
            question: The example question.
            reasoning: List of reasoning steps.
            answer: The final answer.
        """
        self.examples.append(
            {
                "question": question,
                "reasoning": reasoning,
                "answer": answer,
            }
        )

    def build_zero_shot(self, question: str) -> str:
        """Build a zero-shot CoT prompt.

        THE SIMPLEST COT TECHNIQUE:
          Just append "Let's think step by step" to your prompt.
          Despite its simplicity, this works remarkably well because
          it forces the model to generate intermediate reasoning tokens
          before the final answer, reducing errors.

        Args:
            question: The question to reason about.

        Returns:
            A prompt with CoT trigger appended.
        """
        return f"""{self.task}

Question: {question}

{self.cot_trigger}"""

    def build_few_shot(self, question: str) -> str:
        """Build a few-shot CoT prompt with reasoning examples.

        FEW-SHOT COT STRUCTURE:
          [Task description]
          [Example 1 with reasoning chain]
          [Example 2 with reasoning chain]
          [New question — model continues the pattern]

        This is more powerful than zero-shot CoT because the examples
        teach the model the STYLE of reasoning you want.

        Args:
            question: The new question to answer.

        Returns:
            Complete few-shot CoT prompt.
        """
        parts = [self.task, ""]

        for ex in self.examples:
            parts.append(f"Question: {ex['question']}")
            parts.append(self.cot_trigger)
            reasoning_steps = ex["reasoning"]
            assert isinstance(reasoning_steps, list)
            for step in reasoning_steps:
                assert isinstance(step, ReasoningStep)
                parts.append(f"  Step {step.step_number} ({step.description}): {step.content}")
            parts.append(f"Answer: {ex['answer']}")
            parts.append("")

        parts.append(f"Question: {question}")
        parts.append(self.cot_trigger)

        return "\n".join(parts)

    def build_self_consistency(
        self,
        question: str,
        n_paths: int = 5,
    ) -> list[str]:
        """Generate N prompts for self-consistency voting.

        SELF-CONSISTENCY (Wang et al., 2022):
          1. Generate N different reasoning paths (using temperature > 0)
          2. Extract the final answer from each path
          3. Take the majority vote

          This is like asking 5 different people the same question and
          going with the most common answer. It's remarkably effective
          at correcting individual reasoning errors.

        HOW TO USE:
          1. Call this method to get N prompts
          2. Send each prompt to the LLM with temperature=0.7
          3. Extract the final answer from each response
          4. Vote on the most common answer

        Args:
            question: The question to answer.
            n_paths: Number of reasoning paths to generate.

        Returns:
            List of prompts (all identical — diversity comes from temperature).
        """
        # All prompts are the same — diversity comes from the LLM's
        # stochastic sampling (temperature > 0). The identical prompt
        # ensures each path tackles the same question independently.
        if self.examples:
            base_prompt = self.build_few_shot(question)
        else:
            base_prompt = self.build_zero_shot(question)

        return [base_prompt] * n_paths
