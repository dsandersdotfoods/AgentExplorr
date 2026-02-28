"""
Few-Shot Learning — Teaching LLMs by Example
=============================================

WHAT IS FEW-SHOT LEARNING?
  Instead of fine-tuning a model on thousands of examples, you include
  a few examples directly in the prompt. The model learns the pattern
  from these examples and applies it to new inputs.

  Zero-shot:  "Classify this sentiment: 'Great product!'"
  One-shot:   "Example: 'Love it!' → Positive. Now classify: 'Great product!'"
  Few-shot:   3-5 examples → then the actual task

WHY IT WORKS:
  LLMs perform "in-context learning" — they recognize patterns in the prompt
  and extrapolate. This is one of the most surprising capabilities of large
  language models, first demonstrated in the GPT-3 paper.

KEY INSIGHT:
  The QUALITY of examples matters more than the QUANTITY.
  Choose examples that:
  - Cover edge cases
  - Are diverse (not all the same pattern)
  - Are similar to the expected input (semantic similarity selection)

LEARNING RESOURCES:
  - Few-Shot Prompting Guide: https://www.promptingguide.ai/techniques/fewshot
  - GPT-3 Paper ("Language Models are Few-Shot Learners"): https://arxiv.org/abs/2005.14165
  - VIDEO: "Few-Shot Learning Explained" — https://www.youtube.com/watch?v=hE7eGew4eeg
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Example:
    """A single input-output example for few-shot learning.

    Attributes:
        input_text: The example input (what you'd ask the model).
        output_text: The expected output (what you want the model to produce).
        metadata: Optional tags for filtering/selection (e.g., {"category": "positive"}).
    """

    input_text: str
    output_text: str
    metadata: dict[str, str] = field(default_factory=dict)


class FewShotSelector:
    """Select and format few-shot examples for prompt construction.

    This class manages a pool of examples and provides methods to select
    the best ones for a given input.

    SELECTION STRATEGIES:
      1. **Random** — Pick N random examples. Simple but effective baseline.
      2. **Category-based** — Pick examples from specific categories.
         Useful when you have labeled examples and want diversity.
      3. **Semantic similarity** — (Advanced) Pick examples most similar to
         the input using embeddings. This is the best strategy but requires
         a vector store. See the RAG module for embedding-based retrieval.

    Example:
        >>> selector = FewShotSelector()
        >>> selector.add_example(Example(
        ...     input_text="I love this!",
        ...     output_text="Positive",
        ...     metadata={"category": "sentiment"}
        ... ))
        >>> selector.add_example(Example(
        ...     input_text="Terrible experience.",
        ...     output_text="Negative",
        ...     metadata={"category": "sentiment"}
        ... ))
        >>> prompt = selector.format_examples(n=2)
        >>> print(prompt)
    """

    def __init__(self) -> None:
        self._examples: list[Example] = []

    def add_example(self, example: Example) -> None:
        """Add an example to the pool."""
        self._examples.append(example)

    def add_examples(self, examples: list[Example]) -> None:
        """Add multiple examples at once."""
        self._examples.extend(examples)

    @property
    def size(self) -> int:
        """Number of examples in the pool."""
        return len(self._examples)

    def select_random(self, n: int, seed: int | None = None) -> list[Example]:
        """Select N random examples from the pool.

        WHY RANDOM?
          Random selection is a surprisingly strong baseline. It provides
          diversity automatically. Use this when you don't have a clear
          selection criterion.

        Args:
            n: Number of examples to select.
            seed: Random seed for reproducibility.

        Returns:
            List of selected examples.
        """
        import random

        rng = random.Random(seed)
        n = min(n, len(self._examples))
        return rng.sample(self._examples, n)

    def select_by_metadata(
        self, key: str, value: str, n: int | None = None
    ) -> list[Example]:
        """Select examples matching a metadata filter.

        WHEN TO USE:
          When you have categorized examples and want to pick from a
          specific category. For example, selecting only "positive"
          sentiment examples.

        Args:
            key: Metadata key to filter on.
            value: Metadata value to match.
            n: Max examples to return (None = all matching).

        Returns:
            List of matching examples.
        """
        matches = [ex for ex in self._examples if ex.metadata.get(key) == value]
        if n is not None:
            matches = matches[:n]
        return matches

    def format_examples(
        self,
        examples: list[Example] | None = None,
        n: int = 3,
        input_label: str = "Input",
        output_label: str = "Output",
        separator: str = "\n\n",
    ) -> str:
        """Format examples into a prompt string.

        HOW FORMATTING WORKS:
          Each example becomes:
            Input: <input_text>
            Output: <output_text>

          Examples are separated by blank lines for readability.

        Args:
            examples: Specific examples to format. If None, selects random.
            n: Number of examples (used only if examples is None).
            input_label: Label for input (e.g., "Question", "Text").
            output_label: Label for output (e.g., "Answer", "Category").
            separator: String between examples.

        Returns:
            Formatted string ready to insert into a prompt.
        """
        if examples is None:
            examples = self.select_random(n)

        formatted = []
        for ex in examples:
            formatted.append(f"{input_label}: {ex.input_text}\n{output_label}: {ex.output_text}")

        return separator.join(formatted)

    def build_prompt(
        self,
        instruction: str,
        input_text: str,
        examples: list[Example] | None = None,
        n: int = 3,
        input_label: str = "Input",
        output_label: str = "Output",
    ) -> str:
        """Build a complete few-shot prompt with instruction, examples, and input.

        This combines:
          1. An instruction (what the model should do)
          2. Few-shot examples (showing how to do it)
          3. The actual input (what the model should process)

        THE FEW-SHOT PROMPT STRUCTURE:
          ┌──────────────────────────────┐
          │ Instruction                  │  ← What to do
          │                              │
          │ Input: example_1_input       │  ← Example 1
          │ Output: example_1_output     │
          │                              │
          │ Input: example_2_input       │  ← Example 2
          │ Output: example_2_output     │
          │                              │
          │ Input: actual_input          │  ← Your input
          │ Output:                      │  ← Model completes this
          └──────────────────────────────┘

        Args:
            instruction: Task description for the model.
            input_text: The actual input to process.
            examples: Specific examples (None = random selection).
            n: Number of examples if none specified.
            input_label: Label for inputs.
            output_label: Label for outputs.

        Returns:
            Complete few-shot prompt string.
        """
        example_str = self.format_examples(
            examples=examples,
            n=n,
            input_label=input_label,
            output_label=output_label,
        )

        return f"""{instruction}

{example_str}

{input_label}: {input_text}
{output_label}:"""
