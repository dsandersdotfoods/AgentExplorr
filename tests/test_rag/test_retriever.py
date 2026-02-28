"""
Tests for the Hybrid Retriever
================================

Tests the HybridRetriever with mock vector stores to verify
deduplication, score-based ranking, and RRF re-ranking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from agentexplorr.rag.retriever import HybridRetriever, RetrievalResult

# ---------------------------------------------------------------------------
# Mock vector store
# ---------------------------------------------------------------------------


@dataclass
class _MockSearchResult:
    """Mimics a vector store search result."""
    chunk_id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class _MockVectorStore:
    """A fake vector store that returns predetermined results."""

    def __init__(self, results: list[_MockSearchResult]) -> None:
        self._results = results

    def search(self, query: str, top_k: int = 5) -> list[_MockSearchResult]:
        return self._results[:top_k]


class _BrokenStore:
    """A store that always raises an error."""

    def search(self, query: str, top_k: int = 5) -> list[Any]:
        raise RuntimeError("Store is down")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHybridRetriever:

    def test_single_store_search(self) -> None:
        store = _MockVectorStore([
            _MockSearchResult("c1", "chunk one", 0.9),
            _MockSearchResult("c2", "chunk two", 0.8),
        ])
        retriever = HybridRetriever(stores={"primary": store})
        results = retriever.search("test query", top_k=5)

        assert len(results) == 2
        assert results[0].score >= results[1].score

    def test_multi_store_deduplication(self) -> None:
        store_a = _MockVectorStore([
            _MockSearchResult("c1", "chunk one", 0.9),
            _MockSearchResult("c2", "chunk two", 0.7),
        ])
        store_b = _MockVectorStore([
            _MockSearchResult("c1", "chunk one", 0.85),  # duplicate
            _MockSearchResult("c3", "chunk three", 0.8),
        ])
        retriever = HybridRetriever(
            stores={"a": store_a, "b": store_b},
            strategy="score",
        )
        results = retriever.search("test", top_k=10)

        # c1 should appear only once (deduplicated)
        chunk_ids = [r.chunk_id for r in results]
        assert chunk_ids.count("c1") == 1
        assert len(results) == 3

    def test_dedup_keeps_highest_score(self) -> None:
        store_a = _MockVectorStore([
            _MockSearchResult("c1", "chunk one", 0.9),
        ])
        store_b = _MockVectorStore([
            _MockSearchResult("c1", "chunk one", 0.5),
        ])
        retriever = HybridRetriever(stores={"a": store_a, "b": store_b})
        results = retriever.search("test", top_k=5)

        assert len(results) == 1
        assert results[0].score == 0.9

    def test_rrf_strategy(self) -> None:
        store_a = _MockVectorStore([
            _MockSearchResult("c1", "chunk one", 0.9),
            _MockSearchResult("c2", "chunk two", 0.5),
        ])
        store_b = _MockVectorStore([
            _MockSearchResult("c2", "chunk two", 0.95),
            _MockSearchResult("c3", "chunk three", 0.4),
        ])
        retriever = HybridRetriever(
            stores={"a": store_a, "b": store_b},
            strategy="rrf",
        )
        results = retriever.search("test", top_k=10)

        # c2 appears in both stores, so its RRF score should be highest
        assert results[0].chunk_id == "c2"

    def test_score_ordering(self) -> None:
        store = _MockVectorStore([
            _MockSearchResult("c1", "low", 0.3),
            _MockSearchResult("c2", "high", 0.9),
            _MockSearchResult("c3", "mid", 0.6),
        ])
        retriever = HybridRetriever(stores={"main": store})
        results = retriever.search("test", top_k=3)

        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_limit(self) -> None:
        store = _MockVectorStore([
            _MockSearchResult(f"c{i}", f"chunk {i}", 1.0 - i * 0.1)
            for i in range(10)
        ])
        retriever = HybridRetriever(stores={"main": store})
        results = retriever.search("test", top_k=3)
        assert len(results) == 3

    def test_empty_query(self) -> None:
        store = _MockVectorStore([
            _MockSearchResult("c1", "chunk", 0.9),
        ])
        retriever = HybridRetriever(stores={"main": store})
        results = retriever.search("", top_k=5)
        assert len(results) == 0

    def test_no_stores(self) -> None:
        retriever = HybridRetriever(stores={})
        results = retriever.search("test", top_k=5)
        assert len(results) == 0

    def test_store_error_resilience(self) -> None:
        good_store = _MockVectorStore([
            _MockSearchResult("c1", "chunk one", 0.9),
        ])
        bad_store = _BrokenStore()

        retriever = HybridRetriever(
            stores={"good": good_store, "bad": bad_store},
        )
        # Should not raise, should return results from the good store
        results = retriever.search("test", top_k=5)
        assert len(results) == 1

    def test_add_store(self) -> None:
        retriever = HybridRetriever()
        store = _MockVectorStore([_MockSearchResult("c1", "text", 0.9)])
        retriever.add_store("new_store", store)
        results = retriever.search("test", top_k=5)
        assert len(results) == 1

    def test_add_store_validates_interface(self) -> None:
        retriever = HybridRetriever()
        with pytest.raises(TypeError, match="search"):
            retriever.add_store("bad", object())

    def test_remove_store(self) -> None:
        store = _MockVectorStore([])
        retriever = HybridRetriever(stores={"main": store})
        retriever.remove_store("main")
        assert retriever.search("test") == []

    def test_remove_nonexistent_store(self) -> None:
        retriever = HybridRetriever()
        with pytest.raises(KeyError):
            retriever.remove_store("nonexistent")

    def test_invalid_strategy(self) -> None:
        with pytest.raises(ValueError, match="Unknown strategy"):
            HybridRetriever(strategy="invalid")

    def test_store_weights(self) -> None:
        store_a = _MockVectorStore([
            _MockSearchResult("c1", "from A", 1.0),
        ])
        store_b = _MockVectorStore([
            _MockSearchResult("c2", "from B", 1.0),
        ])
        retriever = HybridRetriever(
            stores={"a": store_a, "b": store_b},
            weights={"a": 0.3, "b": 0.7},
        )
        results = retriever.search("test", top_k=5)

        by_id = {r.chunk_id: r for r in results}
        assert by_id["c1"].score == pytest.approx(0.3)
        assert by_id["c2"].score == pytest.approx(0.7)


class TestRetrievalResult:
    def test_repr(self) -> None:
        r = RetrievalResult(chunk_id="abc", text="Hello world", score=0.95)
        assert "0.95" in repr(r)
        assert "Hello world" in repr(r)
