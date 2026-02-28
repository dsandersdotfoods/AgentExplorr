"""
Prompt Engineering Module
=========================

Master the art and science of crafting effective prompts for LLMs.

This module covers:
  1. **Templates** — Reusable Jinja2 prompt templates with variable injection
  2. **Few-Shot Learning** — Teaching LLMs by example (in-context learning)
  3. **Chain of Thought** — Step-by-step reasoning for complex tasks
  4. **Structured Output** — Parsing LLM responses into Pydantic models

WHY PROMPT ENGINEERING MATTERS:
  The same model can give wildly different results depending on the prompt.
  A well-crafted prompt can turn a mediocre response into an expert-level one.
  This is the most accessible and impactful AI skill you can develop.

LEARNING RESOURCES:
  - Prompt Engineering Guide: https://www.promptingguide.ai/
  - OpenAI Prompt Engineering: https://platform.openai.com/docs/guides/prompt-engineering
  - Anthropic Prompt Engineering: https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering
  - VIDEO: "Prompt Engineering Full Course" — https://www.youtube.com/watch?v=_ZvnD96hOZo
"""

from agentexplorr.prompt_engineering.chain_of_thought import ChainOfThoughtPrompt
from agentexplorr.prompt_engineering.few_shot import FewShotSelector
from agentexplorr.prompt_engineering.structured_output import OutputParser
from agentexplorr.prompt_engineering.templates import PromptTemplate

__all__ = [
    "ChainOfThoughtPrompt",
    "FewShotSelector",
    "OutputParser",
    "PromptTemplate",
]
