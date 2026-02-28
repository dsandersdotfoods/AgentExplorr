"""
Agent Evaluation Module -- Measuring Agent Performance
========================================================

WHY EVALUATE AGENTS?
  "If you can't measure it, you can't improve it." -- Peter Drucker

  Agent evaluation is HARD because agents are non-deterministic:
    - The same query might get different answers each time
    - Tool calls may return different results (search results change)
    - Intermediate reasoning varies between runs

  Despite this, we need principled ways to measure:
    - **Accuracy** -- Does the agent get the right answer?
    - **Latency** -- How long does it take?
    - **Tool efficiency** -- Does it use tools wisely (not too many, not too few)?
    - **Robustness** -- Does it handle edge cases and errors gracefully?

EVALUATION APPROACHES:
  1. **Ground-truth benchmarks** -- Questions with known correct answers.
     Run the agent on each question, check if the answer is correct.
     Simple but limited (only works for factual questions).

  2. **LLM-as-judge** -- Use another LLM to grade the agent's answer.
     More flexible (can evaluate open-ended responses) but introduces
     bias from the judge model.

  3. **Human evaluation** -- The gold standard but expensive and slow.
     Reserve for final validation, not development iteration.

  4. **Tool usage analysis** -- Track which tools were called, how many
     times, and whether the calls were appropriate. Good for debugging.

THIS MODULE PROVIDES:
  - ``AgentBenchmark`` -- Run agents on question sets, measure accuracy
    and latency. Supports both exact-match and fuzzy matching.

LEARNING RESOURCES:
  - PAPER: "Evaluating LLM Agent Group Performances" -- https://arxiv.org/abs/2401.04531
  - PAPER: "AgentBench: Evaluating LLMs as Agents" -- https://arxiv.org/abs/2308.03688
  - LangSmith evaluation: https://docs.smith.langchain.com/evaluation
  - VIDEO: "How to Evaluate AI Agents" -- https://www.youtube.com/watch?v=2e_7VCnAzCQ
  - VIDEO: "LLM Evaluation Explained" -- https://www.youtube.com/watch?v=E3BGBIBYB0Y
"""

from __future__ import annotations

from agentexplorr.agents.evaluation.benchmarks import (
    AgentBenchmark,
    BenchmarkQuestion,
    BenchmarkResult,
    BenchmarkSummary,
)

__all__ = [
    "AgentBenchmark",
    "BenchmarkQuestion",
    "BenchmarkResult",
    "BenchmarkSummary",
]
