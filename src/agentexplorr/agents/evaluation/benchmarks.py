"""
Agent Benchmarking -- Systematic Performance Evaluation
========================================================

WHY BENCHMARK AGENTS?
  Building an agent is easy. Building a GOOD agent is hard. Without
  systematic evaluation, you're flying blind -- you don't know if your
  changes are making the agent better or worse.

  Benchmarking gives you:
    - **Baseline metrics** -- Know where you stand before optimizing
    - **Regression detection** -- Catch regressions when you change prompts/models
    - **Model comparison** -- Compare llama3.2 vs mistral vs command-r
    - **Architecture comparison** -- ReAct vs ToolAgent vs MultiAgent

HOW THIS BENCHMARK WORKS:
  1. Define a set of questions with known correct answers
  2. Run the agent on each question
  3. Compare the agent's answer to the expected answer
  4. Calculate accuracy, latency, and tool usage statistics
  5. Return a structured report

MATCHING STRATEGIES:
  - **Exact match** -- Agent's answer must contain the exact expected string
  - **Fuzzy match** -- Uses sequence matching to allow minor differences
    (typos, rephrasing). A score of 0.8+ is typically "correct enough."
  - **Keyword match** -- Answer must contain ALL specified keywords
  - **Numeric match** -- For math questions, extract numbers and compare
    with a small tolerance (0.01)

BUILT-IN QUESTION SETS:
  We include starter question sets for different capabilities:
    - ``math_questions`` -- Arithmetic and math function questions
    - ``search_questions`` -- Factual questions requiring web search
    - ``mixed_questions`` -- Combination of both types

  These are intentionally simple so you can verify your setup works
  before creating domain-specific benchmarks.

LEARNING RESOURCES:
  - PAPER: "AgentBench" (Liu et al., 2023) -- https://arxiv.org/abs/2308.03688
  - PAPER: "Measuring Massive Multitask Language Understanding" -- https://arxiv.org/abs/2009.03300
  - HuggingFace Evaluate library: https://huggingface.co/docs/evaluate/
  - VIDEO: "How to Evaluate LLM Applications" -- https://www.youtube.com/watch?v=2e_7VCnAzCQ
  - VIDEO: "Building LLM Benchmarks" -- https://www.youtube.com/watch?v=E3BGBIBYB0Y
  - LangSmith Evaluation: https://docs.smith.langchain.com/evaluation
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Callable, Protocol, runtime_checkable
import re
import statistics

from agentexplorr.core import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Protocols -- define what an "agent" looks like to the benchmark
# ---------------------------------------------------------------------------
# Using Protocol (structural typing) means any object with a `run()` method
# that returns an object with a `final_answer` or `answer` attribute can
# be benchmarked. This works with ReActAgent, ToolAgent, and custom agents.

@runtime_checkable
class AgentProtocol(Protocol):
    """Protocol defining what the benchmark expects from an agent.

    Any agent class that has a ``run(query: str)`` method returning an
    object with a ``final_answer`` or ``answer`` attribute will work.

    This uses Python's structural typing (Protocol) rather than
    inheritance. Your agent doesn't need to inherit from anything --
    it just needs to have the right methods.

    LEARNING RESOURCE:
      - Python Protocols: https://docs.python.org/3/library/typing.html#typing.Protocol
      - VIDEO: "Protocols in Python" -- https://www.youtube.com/watch?v=xvb5hGLoK0A
    """

    def run(self, query: str) -> Any:
        """Run the agent on a query and return a result object."""
        ...


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkQuestion:
    """A single benchmark question with its expected answer.

    Attributes:
        question: The question to ask the agent.
        expected_answer: The correct answer (or key part of it).
        category: Optional category for grouping (e.g., "math", "search").
        keywords: Optional keywords that must appear in a correct answer.
        match_type: How to compare the agent's answer to expected.
                   "contains" (default), "fuzzy", "keywords", "numeric".
        difficulty: Optional difficulty level ("easy", "medium", "hard").
    """

    question: str
    expected_answer: str
    category: str = "general"
    keywords: list[str] = field(default_factory=list)
    match_type: str = "contains"
    difficulty: str = "medium"


@dataclass
class BenchmarkResult:
    """Result from running a single benchmark question.

    Attributes:
        question: The benchmark question that was run.
        agent_answer: The answer the agent produced.
        is_correct: Whether the answer matched the expected answer.
        match_score: Numeric similarity score (0.0 to 1.0).
        latency_seconds: How long the agent took to answer.
        tools_used: Which tools the agent called (if available).
        error: Error message if the agent failed, else None.
    """

    question: BenchmarkQuestion
    agent_answer: str
    is_correct: bool
    match_score: float
    latency_seconds: float
    tools_used: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class BenchmarkSummary:
    """Summary statistics from a complete benchmark run.

    Attributes:
        total_questions: Total number of questions attempted.
        correct: Number of questions answered correctly.
        incorrect: Number of questions answered incorrectly.
        errors: Number of questions that caused agent errors.
        accuracy: Fraction of correct answers (0.0 to 1.0).
        avg_latency: Average time per question (seconds).
        median_latency: Median time per question (seconds).
        p95_latency: 95th percentile latency (seconds).
        total_time: Total benchmark execution time (seconds).
        tool_usage: Dict mapping tool names to call counts.
        per_category: Accuracy broken down by question category.
        results: Full list of individual question results.
    """

    total_questions: int
    correct: int
    incorrect: int
    errors: int
    accuracy: float
    avg_latency: float
    median_latency: float
    p95_latency: float
    total_time: float
    tool_usage: dict[str, int]
    per_category: dict[str, dict[str, Any]]
    results: list[BenchmarkResult]


# ---------------------------------------------------------------------------
# Answer matching functions
# ---------------------------------------------------------------------------

def _contains_match(agent_answer: str, expected: str) -> tuple[bool, float]:
    """Check if the agent's answer contains the expected answer.

    This is the simplest matching strategy. It's case-insensitive and
    checks if the expected string appears anywhere in the agent's answer.

    Good for: factual questions where the answer is a specific phrase.

    Args:
        agent_answer: The agent's full response.
        expected: The expected answer string.

    Returns:
        Tuple of (is_correct, match_score).
    """
    answer_lower = agent_answer.lower().strip()
    expected_lower = expected.lower().strip()

    if expected_lower in answer_lower:
        return True, 1.0

    # Partial credit: use sequence matching to see how close we are
    score = SequenceMatcher(None, answer_lower, expected_lower).ratio()
    return score >= 0.8, score


def _fuzzy_match(agent_answer: str, expected: str) -> tuple[bool, float]:
    """Fuzzy match using SequenceMatcher.

    More forgiving than exact matching. Uses Python's difflib to compute
    a similarity ratio between the two strings. A score of 0.8+ is
    typically "close enough" for a correct answer.

    HOW SEQUENCEMATCHER WORKS:
      It finds the longest contiguous matching subsequence, then
      recursively matches the parts before and after. The ratio
      is 2 * M / T where M = matching characters and T = total characters.

    Args:
        agent_answer: The agent's full response.
        expected: The expected answer string.

    Returns:
        Tuple of (is_correct, match_score).
    """
    # Normalize: lowercase, collapse whitespace
    answer_normalized = " ".join(agent_answer.lower().split())
    expected_normalized = " ".join(expected.lower().split())

    score = SequenceMatcher(None, answer_normalized, expected_normalized).ratio()

    # Also check if the expected answer is contained in the response
    # (the agent might produce a longer answer that includes the expected text)
    if expected_normalized in answer_normalized:
        score = max(score, 1.0)

    return score >= 0.8, score


def _keyword_match(
    agent_answer: str, expected: str, keywords: list[str]
) -> tuple[bool, float]:
    """Check if the answer contains all required keywords.

    This is useful for open-ended questions where the exact phrasing
    doesn't matter, but certain concepts must be mentioned.

    Example:
      Question: "What are the benefits of exercise?"
      Keywords: ["health", "fitness", "mental"]
      Answer: "Exercise improves health, fitness levels, and mental well-being."
      -> Correct (all keywords present)

    Args:
        agent_answer: The agent's response.
        expected: The expected answer (used as fallback).
        keywords: List of keywords that must all be present.

    Returns:
        Tuple of (is_correct, match_score).
    """
    answer_lower = agent_answer.lower()

    if not keywords:
        # Fall back to contains match if no keywords specified
        return _contains_match(agent_answer, expected)

    matches = sum(1 for kw in keywords if kw.lower() in answer_lower)
    score = matches / len(keywords)

    return score >= 0.8, score


def _numeric_match(agent_answer: str, expected: str) -> tuple[bool, float]:
    """Match numeric answers with tolerance.

    For math/calculation questions, we extract numbers from both the
    agent's answer and the expected answer, then compare with a small
    tolerance to handle floating-point imprecision.

    TOLERANCE:
      We use both absolute (0.01) and relative (1%) tolerance.
      This handles both "2 + 2 = 4" (small numbers) and
      "sqrt(2) = 1.41421356..." (decimal precision).

    Args:
        agent_answer: The agent's response.
        expected: The expected numeric answer (as string).

    Returns:
        Tuple of (is_correct, match_score).
    """
    # Extract numbers from both strings
    agent_numbers = _extract_numbers(agent_answer)
    expected_numbers = _extract_numbers(expected)

    if not expected_numbers:
        # No numbers to compare -- fall back to contains match
        return _contains_match(agent_answer, expected)

    if not agent_numbers:
        return False, 0.0

    # Check if ANY number in the agent's answer matches ANY expected number
    for exp_num in expected_numbers:
        for agent_num in agent_numbers:
            # Absolute tolerance
            if abs(agent_num - exp_num) < 0.01:
                return True, 1.0
            # Relative tolerance (1%)
            if exp_num != 0 and abs((agent_num - exp_num) / exp_num) < 0.01:
                return True, 1.0

    # Partial credit: how close is the closest match?
    min_distance = float("inf")
    for exp_num in expected_numbers:
        for agent_num in agent_numbers:
            if exp_num != 0:
                distance = abs((agent_num - exp_num) / exp_num)
            else:
                distance = abs(agent_num)
            min_distance = min(min_distance, distance)

    score = max(0.0, 1.0 - min_distance)
    return False, score


def _extract_numbers(text: str) -> list[float]:
    """Extract all numeric values from a text string.

    Handles integers, floats, negative numbers, and numbers with commas.

    Examples:
        "The answer is 42" -> [42.0]
        "Between -3.14 and 2,500" -> [-3.14, 2500.0]
        "sqrt(144) = 12" -> [144.0, 12.0]

    Args:
        text: The text to extract numbers from.

    Returns:
        List of extracted float values.
    """
    # Pattern matches: -3.14, 2500, 1,000,000, .5, etc.
    pattern = r"-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\.\d+"
    matches = re.findall(pattern, text)

    numbers: list[float] = []
    for match in matches:
        try:
            # Remove commas before converting
            cleaned = match.replace(",", "")
            numbers.append(float(cleaned))
        except ValueError:
            continue

    return numbers


# Match type dispatcher -- maps match_type strings to functions
MATCH_FUNCTIONS: dict[str, Callable[..., tuple[bool, float]]] = {
    "contains": _contains_match,
    "fuzzy": _fuzzy_match,
    "keywords": _keyword_match,
    "numeric": _numeric_match,
}


# ---------------------------------------------------------------------------
# Built-in question sets
# ---------------------------------------------------------------------------

def get_math_questions() -> list[BenchmarkQuestion]:
    """A set of math questions for testing calculator tool usage.

    These are designed to require the calculator tool. An agent that
    tries to do math "in its head" will likely get some wrong.

    Returns:
        List of math-focused benchmark questions.
    """
    return [
        BenchmarkQuestion(
            question="What is 247 * 389?",
            expected_answer="96083",
            category="math",
            match_type="numeric",
            difficulty="easy",
        ),
        BenchmarkQuestion(
            question="What is the square root of 7569?",
            expected_answer="87",
            category="math",
            match_type="numeric",
            difficulty="easy",
        ),
        BenchmarkQuestion(
            question="Calculate 15% of 2480.",
            expected_answer="372",
            category="math",
            match_type="numeric",
            difficulty="easy",
        ),
        BenchmarkQuestion(
            question="What is 2^16?",
            expected_answer="65536",
            category="math",
            match_type="numeric",
            difficulty="medium",
        ),
        BenchmarkQuestion(
            question="What is sin(pi/6) rounded to 2 decimal places?",
            expected_answer="0.5",
            category="math",
            match_type="numeric",
            difficulty="medium",
        ),
        BenchmarkQuestion(
            question="Calculate the factorial of 10.",
            expected_answer="3628800",
            category="math",
            match_type="numeric",
            difficulty="medium",
        ),
        BenchmarkQuestion(
            question="What is log base 10 of 10000?",
            expected_answer="4",
            category="math",
            match_type="numeric",
            difficulty="easy",
        ),
        BenchmarkQuestion(
            question="If a circle has radius 7, what is its area? Use pi * r^2.",
            expected_answer="153.94",
            category="math",
            match_type="numeric",
            difficulty="medium",
        ),
    ]


def get_search_questions() -> list[BenchmarkQuestion]:
    """A set of factual questions for testing web search tool usage.

    These require the agent to search the web. Answers may change over
    time (e.g., population figures), so we use keyword matching for
    flexibility.

    Returns:
        List of search-focused benchmark questions.
    """
    return [
        BenchmarkQuestion(
            question="What is the capital of Australia?",
            expected_answer="Canberra",
            category="search",
            match_type="contains",
            difficulty="easy",
        ),
        BenchmarkQuestion(
            question="Who wrote the novel '1984'?",
            expected_answer="George Orwell",
            category="search",
            match_type="contains",
            difficulty="easy",
        ),
        BenchmarkQuestion(
            question="What programming language was created by Guido van Rossum?",
            expected_answer="Python",
            category="search",
            match_type="contains",
            difficulty="easy",
        ),
        BenchmarkQuestion(
            question="What is the chemical symbol for gold?",
            expected_answer="Au",
            category="search",
            match_type="contains",
            difficulty="easy",
        ),
        BenchmarkQuestion(
            question="What year was the first iPhone released?",
            expected_answer="2007",
            category="search",
            match_type="contains",
            difficulty="medium",
        ),
    ]


def get_mixed_questions() -> list[BenchmarkQuestion]:
    """A mixed set combining math and search questions.

    Returns:
        Combined list of math and search benchmark questions.
    """
    return get_math_questions() + get_search_questions()


# ---------------------------------------------------------------------------
# AgentBenchmark class
# ---------------------------------------------------------------------------

class AgentBenchmark:
    """Run systematic benchmarks on AI agents.

    This class orchestrates the evaluation process:
      1. Takes an agent and a set of questions
      2. Runs the agent on each question
      3. Compares answers to expected answers
      4. Computes summary statistics
      5. Returns a structured report

    USAGE:
        >>> from agentexplorr.agents import ReActAgent
        >>> from agentexplorr.agents.evaluation import AgentBenchmark
        >>>
        >>> agent = ReActAgent()
        >>> benchmark = AgentBenchmark()
        >>>
        >>> # Run on built-in math questions
        >>> summary = benchmark.run(agent, benchmark.math_questions())
        >>> print(f"Accuracy: {summary.accuracy:.1%}")
        >>> print(f"Avg latency: {summary.avg_latency:.1f}s")
        >>>
        >>> # Run on custom questions
        >>> my_questions = [
        ...     BenchmarkQuestion("What is 2+2?", "4", match_type="numeric"),
        ...     BenchmarkQuestion("Capital of France?", "Paris"),
        ... ]
        >>> summary = benchmark.run(agent, my_questions)

    COMPARING AGENTS:
        >>> react_agent = ReActAgent()
        >>> tool_agent = ToolAgent()
        >>>
        >>> questions = benchmark.mixed_questions()
        >>> react_summary = benchmark.run(react_agent, questions)
        >>> tool_summary = benchmark.run(tool_agent, questions)
        >>>
        >>> print(f"ReAct accuracy: {react_summary.accuracy:.1%}")
        >>> print(f"Tool  accuracy: {tool_summary.accuracy:.1%}")

    Attributes:
        verbose: If True, print progress during benchmarking.
        fuzzy_threshold: Similarity threshold for fuzzy matching (0.0-1.0).
    """

    def __init__(
        self,
        verbose: bool = True,
        fuzzy_threshold: float = 0.8,
    ) -> None:
        """Initialize the benchmark runner.

        Args:
            verbose: Print progress and results during execution.
            fuzzy_threshold: Minimum similarity score for fuzzy matching.
        """
        self.verbose = verbose
        self.fuzzy_threshold = fuzzy_threshold

    # --- Built-in question set accessors ---

    @staticmethod
    def math_questions() -> list[BenchmarkQuestion]:
        """Get the built-in math benchmark questions."""
        return get_math_questions()

    @staticmethod
    def search_questions() -> list[BenchmarkQuestion]:
        """Get the built-in search benchmark questions."""
        return get_search_questions()

    @staticmethod
    def mixed_questions() -> list[BenchmarkQuestion]:
        """Get the built-in mixed benchmark questions."""
        return get_mixed_questions()

    # --- Main benchmark method ---

    def run(
        self,
        agent: Any,
        questions: list[BenchmarkQuestion],
    ) -> BenchmarkSummary:
        """Run the benchmark on an agent.

        Executes each question sequentially (agents are typically not
        thread-safe), measures latency, checks correctness, and
        compiles a summary.

        Args:
            agent: An agent object with a ``run(query: str)`` method.
                  Works with ReActAgent, ToolAgent, MultiAgentSupervisor,
                  or any object following the AgentProtocol.
            questions: List of BenchmarkQuestion objects to evaluate.

        Returns:
            A BenchmarkSummary with all metrics and individual results.

        Raises:
            ValueError: If the questions list is empty.
        """
        if not questions:
            raise ValueError("Questions list cannot be empty")

        total_start = time.perf_counter()
        results: list[BenchmarkResult] = []

        if self.verbose:
            print(f"\nRunning benchmark: {len(questions)} questions")
            print("=" * 60)

        for i, question in enumerate(questions, 1):
            result = self._run_single(agent, question, index=i, total=len(questions))
            results.append(result)

        total_time = time.perf_counter() - total_start

        # Compile summary
        summary = self._compile_summary(results, total_time)

        if self.verbose:
            self._print_summary(summary)

        return summary

    def _run_single(
        self,
        agent: Any,
        question: BenchmarkQuestion,
        index: int,
        total: int,
    ) -> BenchmarkResult:
        """Run the agent on a single question and evaluate the answer.

        Args:
            agent: The agent to test.
            question: The benchmark question.
            index: Question number (for progress display).
            total: Total number of questions (for progress display).

        Returns:
            BenchmarkResult for this question.
        """
        if self.verbose:
            print(f"\n[{index}/{total}] {question.question}")
            print(f"  Expected: {question.expected_answer}")

        # Time the agent's execution
        start = time.perf_counter()
        try:
            agent_result = agent.run(question.question)
            latency = time.perf_counter() - start

            # Extract the answer string from the result object.
            # Different agents return different result types, so we
            # check for common attribute names.
            answer = self._extract_answer(agent_result)

        except Exception as e:
            latency = time.perf_counter() - start
            logger.error(
                "benchmark_question_error",
                question=question.question[:50],
                error=str(e),
            )
            if self.verbose:
                print(f"  ERROR: {e}")

            return BenchmarkResult(
                question=question,
                agent_answer="",
                is_correct=False,
                match_score=0.0,
                latency_seconds=latency,
                error=str(e),
            )

        # Evaluate correctness using the specified matching strategy
        is_correct, match_score = self._evaluate_answer(
            answer, question.expected_answer, question.match_type, question.keywords
        )

        # Extract tool usage if available
        tools_used = self._extract_tools(agent_result)

        if self.verbose:
            status = "PASS" if is_correct else "FAIL"
            print(f"  Agent:    {answer[:100]}{'...' if len(answer) > 100 else ''}")
            print(f"  Result:   {status} (score: {match_score:.2f}, {latency:.1f}s)")
            if tools_used:
                print(f"  Tools:    {', '.join(tools_used)}")

        return BenchmarkResult(
            question=question,
            agent_answer=answer,
            is_correct=is_correct,
            match_score=match_score,
            latency_seconds=latency,
            tools_used=tools_used,
        )

    def _extract_answer(self, result: Any) -> str:
        """Extract the answer string from an agent result object.

        Handles different result types from our agents:
          - ReActResult has ``.final_answer``
          - ToolAgentResult has ``.answer``
          - MultiAgentResult has ``.final_answer``
          - Plain strings work too

        Args:
            result: The agent's result object.

        Returns:
            The extracted answer as a string.
        """
        if isinstance(result, str):
            return result

        # Try common attribute names
        for attr in ("final_answer", "answer", "content", "text", "output"):
            if hasattr(result, attr):
                value = getattr(result, attr)
                if isinstance(value, str):
                    return value

        # Last resort: convert to string
        return str(result)

    def _extract_tools(self, result: Any) -> list[str]:
        """Extract tool usage information from an agent result.

        Args:
            result: The agent's result object.

        Returns:
            List of tool names that were used.
        """
        # ReActResult has .tools_used (list of strings)
        if hasattr(result, "tools_used"):
            tools = getattr(result, "tools_used")
            if isinstance(tools, list):
                return [str(t) for t in tools]

        # ToolAgentResult has .tools_called (list of dicts)
        if hasattr(result, "tools_called"):
            tools_called = getattr(result, "tools_called")
            if isinstance(tools_called, list):
                return [
                    tc["name"] if isinstance(tc, dict) else str(tc)
                    for tc in tools_called
                ]

        return []

    def _evaluate_answer(
        self,
        agent_answer: str,
        expected: str,
        match_type: str,
        keywords: list[str],
    ) -> tuple[bool, float]:
        """Evaluate the agent's answer against the expected answer.

        Dispatches to the appropriate matching function based on
        the question's match_type.

        Args:
            agent_answer: The agent's response.
            expected: The expected answer.
            match_type: Matching strategy ("contains", "fuzzy", "keywords", "numeric").
            keywords: Keywords for keyword matching.

        Returns:
            Tuple of (is_correct, match_score).
        """
        if not agent_answer:
            return False, 0.0

        if match_type == "keywords":
            return _keyword_match(agent_answer, expected, keywords)

        match_func = MATCH_FUNCTIONS.get(match_type, _contains_match)
        return match_func(agent_answer, expected)

    def _compile_summary(
        self,
        results: list[BenchmarkResult],
        total_time: float,
    ) -> BenchmarkSummary:
        """Compile individual results into a summary report.

        Calculates aggregate statistics: accuracy, latency percentiles,
        tool usage counts, and per-category breakdowns.

        Args:
            results: List of individual question results.
            total_time: Total wall-clock time for the benchmark.

        Returns:
            BenchmarkSummary with all computed metrics.
        """
        total = len(results)
        correct = sum(1 for r in results if r.is_correct)
        errors = sum(1 for r in results if r.error is not None)
        incorrect = total - correct - errors

        latencies = [r.latency_seconds for r in results]
        sorted_latencies = sorted(latencies)

        # Calculate latency statistics
        avg_latency = statistics.mean(latencies) if latencies else 0.0
        median_latency = statistics.median(latencies) if latencies else 0.0

        # P95 latency: the value below which 95% of observations fall.
        # For small samples, we take the value at the 95th percentile index.
        p95_index = int(len(sorted_latencies) * 0.95)
        p95_latency = sorted_latencies[min(p95_index, len(sorted_latencies) - 1)] if sorted_latencies else 0.0

        # Tool usage counts across all questions
        tool_usage: dict[str, int] = {}
        for r in results:
            for tool_name in r.tools_used:
                tool_usage[tool_name] = tool_usage.get(tool_name, 0) + 1

        # Per-category breakdown
        per_category: dict[str, dict[str, Any]] = {}
        categories = set(r.question.category for r in results)
        for cat in categories:
            cat_results = [r for r in results if r.question.category == cat]
            cat_correct = sum(1 for r in cat_results if r.is_correct)
            cat_total = len(cat_results)
            cat_latencies = [r.latency_seconds for r in cat_results]
            per_category[cat] = {
                "total": cat_total,
                "correct": cat_correct,
                "accuracy": cat_correct / cat_total if cat_total > 0 else 0.0,
                "avg_latency": statistics.mean(cat_latencies) if cat_latencies else 0.0,
            }

        return BenchmarkSummary(
            total_questions=total,
            correct=correct,
            incorrect=incorrect,
            errors=errors,
            accuracy=correct / total if total > 0 else 0.0,
            avg_latency=avg_latency,
            median_latency=median_latency,
            p95_latency=p95_latency,
            total_time=total_time,
            tool_usage=tool_usage,
            per_category=per_category,
            results=results,
        )

    def _print_summary(self, summary: BenchmarkSummary) -> None:
        """Print a formatted benchmark summary to the console.

        Args:
            summary: The compiled benchmark summary.
        """
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)
        print(f"  Total questions:  {summary.total_questions}")
        print(f"  Correct:          {summary.correct}")
        print(f"  Incorrect:        {summary.incorrect}")
        print(f"  Errors:           {summary.errors}")
        print(f"  Accuracy:         {summary.accuracy:.1%}")
        print(f"  Avg latency:      {summary.avg_latency:.2f}s")
        print(f"  Median latency:   {summary.median_latency:.2f}s")
        print(f"  P95 latency:      {summary.p95_latency:.2f}s")
        print(f"  Total time:       {summary.total_time:.1f}s")

        if summary.tool_usage:
            print(f"\n  Tool usage:")
            for tool_name, count in sorted(summary.tool_usage.items()):
                print(f"    {tool_name}: {count} calls")

        if summary.per_category:
            print(f"\n  Per-category accuracy:")
            for cat, stats in sorted(summary.per_category.items()):
                print(
                    f"    {cat}: {stats['accuracy']:.1%} "
                    f"({stats['correct']}/{stats['total']}, "
                    f"avg {stats['avg_latency']:.1f}s)"
                )

        print("=" * 60)
