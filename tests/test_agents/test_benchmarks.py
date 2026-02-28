"""
Tests for Agent Benchmarking System
=====================================

Tests the benchmark question sets, matching strategies, and summary
compilation without needing a real agent or LLM.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from agentexplorr.agents.evaluation.benchmarks import (
    AgentBenchmark,
    BenchmarkQuestion,
    _contains_match,
    _extract_numbers,
    _fuzzy_match,
    _keyword_match,
    _numeric_match,
    get_math_questions,
    get_mixed_questions,
    get_search_questions,
)

# ---------------------------------------------------------------------------
# Matching strategy tests
# ---------------------------------------------------------------------------


class TestContainsMatch:
    def test_exact_contains(self) -> None:
        is_correct, score = _contains_match("The answer is Canberra.", "Canberra")
        assert is_correct is True
        assert score == 1.0

    def test_case_insensitive(self) -> None:
        is_correct, _ = _contains_match("canberra is the capital", "Canberra")
        assert is_correct is True

    def test_no_match(self) -> None:
        is_correct, score = _contains_match("Sydney is great", "Canberra")
        assert is_correct is False
        assert score < 0.8


class TestFuzzyMatch:
    def test_exact(self) -> None:
        is_correct, score = _fuzzy_match("George Orwell", "George Orwell")
        assert is_correct is True
        assert score >= 0.99

    def test_contained(self) -> None:
        is_correct, _score = _fuzzy_match(
            "The novel 1984 was written by George Orwell.", "George Orwell"
        )
        assert is_correct is True

    def test_similar(self) -> None:
        _is_correct, score = _fuzzy_match("George Orwel", "George Orwell")
        assert score > 0.8

    def test_completely_different(self) -> None:
        is_correct, _ = _fuzzy_match("banana", "George Orwell")
        assert is_correct is False


class TestKeywordMatch:
    def test_all_keywords_present(self) -> None:
        is_correct, score = _keyword_match(
            "Exercise improves health, fitness, and mental well-being.",
            "",
            ["health", "fitness", "mental"],
        )
        assert is_correct is True
        assert score == 1.0

    def test_partial_keywords(self) -> None:
        _is_correct, score = _keyword_match(
            "Exercise improves health.", "", ["health", "fitness", "mental"]
        )
        assert score == pytest.approx(1 / 3)

    def test_empty_keywords_falls_back(self) -> None:
        is_correct, _score = _keyword_match("Canberra", "Canberra", [])
        assert is_correct is True


class TestNumericMatch:
    def test_exact_integer(self) -> None:
        is_correct, score = _numeric_match("The answer is 42", "42")
        assert is_correct is True
        assert score == 1.0

    def test_float_tolerance(self) -> None:
        is_correct, _score = _numeric_match("Result: 3.14159", "3.14159")
        assert is_correct is True

    def test_number_in_longer_text(self) -> None:
        is_correct, _ = _numeric_match(
            "After computing, I get 96083 as the final answer.", "96083"
        )
        assert is_correct is True

    def test_no_numbers(self) -> None:
        is_correct, score = _numeric_match("no numbers here", "42")
        assert is_correct is False
        assert score == 0.0


class TestExtractNumbers:
    def test_integers(self) -> None:
        assert _extract_numbers("The answer is 42") == [42.0]

    def test_floats(self) -> None:
        nums = _extract_numbers("pi is approximately 3.14159")
        assert any(abs(n - 3.14159) < 0.001 for n in nums)

    def test_negative(self) -> None:
        assert -3.14 in _extract_numbers("Temperature: -3.14")

    def test_comma_separated(self) -> None:
        nums = _extract_numbers("Population: 1,000,000")
        assert 1000000.0 in nums

    def test_no_numbers(self) -> None:
        assert _extract_numbers("no numbers") == []


# ---------------------------------------------------------------------------
# Question set tests
# ---------------------------------------------------------------------------


class TestQuestionSets:
    def test_math_questions_not_empty(self) -> None:
        qs = get_math_questions()
        assert len(qs) > 0
        assert all(q.category == "math" for q in qs)
        assert all(q.match_type == "numeric" for q in qs)

    def test_search_questions_not_empty(self) -> None:
        qs = get_search_questions()
        assert len(qs) > 0
        assert all(q.category == "search" for q in qs)

    def test_mixed_is_union(self) -> None:
        mixed = get_mixed_questions()
        math = get_math_questions()
        search = get_search_questions()
        assert len(mixed) == len(math) + len(search)


# ---------------------------------------------------------------------------
# AgentBenchmark tests (with a fake agent)
# ---------------------------------------------------------------------------


@dataclass
class _FakeResult:
    """Mimics an agent result with a final_answer."""
    final_answer: str
    tools_used: list[str]


class _FakeAgent:
    """A fake agent that returns predetermined answers for testing."""

    def __init__(self, answers: dict[str, str]) -> None:
        self._answers = answers

    def run(self, query: str) -> _FakeResult:
        answer = self._answers.get(query, "I don't know")
        return _FakeResult(final_answer=answer, tools_used=["calculator"])


class TestAgentBenchmark:
    def test_run_with_fake_agent(self) -> None:
        questions = [
            BenchmarkQuestion(
                question="What is 2 + 2?",
                expected_answer="4",
                category="math",
                match_type="numeric",
            ),
            BenchmarkQuestion(
                question="Capital of France?",
                expected_answer="Paris",
                category="search",
                match_type="contains",
            ),
        ]
        agent = _FakeAgent(
            answers={
                "What is 2 + 2?": "The answer is 4.",
                "Capital of France?": "Paris is the capital of France.",
            }
        )

        benchmark = AgentBenchmark(verbose=False)
        summary = benchmark.run(agent, questions)

        assert summary.total_questions == 2
        assert summary.correct == 2
        assert summary.accuracy == 1.0
        assert summary.errors == 0
        assert "calculator" in summary.tool_usage

    def test_empty_questions_raises(self) -> None:
        benchmark = AgentBenchmark(verbose=False)
        with pytest.raises(ValueError, match="empty"):
            benchmark.run(_FakeAgent({}), [])

    def test_agent_error_handling(self) -> None:
        class _ErrorAgent:
            def run(self, query: str) -> None:
                raise RuntimeError("LLM down")

        questions = [
            BenchmarkQuestion(question="test?", expected_answer="yes"),
        ]
        benchmark = AgentBenchmark(verbose=False)
        summary = benchmark.run(_ErrorAgent(), questions)

        assert summary.errors == 1
        assert summary.correct == 0

    def test_per_category_breakdown(self) -> None:
        questions = [
            BenchmarkQuestion("q1", "4", category="math", match_type="numeric"),
            BenchmarkQuestion("q2", "Paris", category="search"),
        ]
        agent = _FakeAgent({"q1": "4", "q2": "Paris"})
        summary = AgentBenchmark(verbose=False).run(agent, questions)

        assert "math" in summary.per_category
        assert "search" in summary.per_category
        assert summary.per_category["math"]["accuracy"] == 1.0
