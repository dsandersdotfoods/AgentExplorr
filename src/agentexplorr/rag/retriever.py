"""
Hybrid Retriever — Searching Across Multiple Vector Stores
============================================================

WHAT IS A HYBRID RETRIEVER?
  A Hybrid Retriever queries multiple vector stores (e.g. ChromaDB AND FAISS)
  and combines the results into a single ranked list.  This is a form of
  **ensemble retrieval**: different stores may have different strengths, and
  combining them often yields better recall than any single store alone.

WHY HYBRID RETRIEVAL?
  In practice, no single retrieval method is perfect:
    - Dense retrieval (embeddings) excels at semantic similarity but can miss
      exact keyword matches.
    - Different vector stores may use different index structures (HNSW vs.
      flat vs. IVF) that find different sets of approximate neighbors.
    - You might have data partitioned across stores by domain or time period.

  By querying multiple stores and re-ranking the union, you get:
    1. **Higher recall**: items found by ANY store make the candidate list.
    2. **Better ranking**: re-ranking by score normalizes differences between
       stores' scoring functions.
    3. **Resilience**: if one store is down or empty, the other still works.

HOW THIS RETRIEVER WORKS:
  1. Send the query to each registered vector store in parallel.
  2. Collect all SearchResult objects from all stores.
  3. Deduplicate by chunk_id (same chunk might be in both stores).
  4. Re-rank by score (optionally with Reciprocal Rank Fusion).
  5. Return the top K results.

RE-RANKING STRATEGIES:

  **Score-based** (default):
    Simply sort all results by their similarity score.  Fast and simple, but
    scores from different stores may not be directly comparable (one store's
    0.8 might be another's 0.5 for the same pair).

  **Reciprocal Rank Fusion (RRF)**:
    A scoring method that cares about *rank position* rather than raw scores.
    RRF score = sum(1 / (k + rank_in_store_i)) for each store.
    This is robust to score scale differences and is used in production search
    engines.

    Paper: "Reciprocal Rank Fusion outperforms Condorcet and individual Rank
    Learning Methods" (Cormack et al., 2009)

LEARNING RESOURCES:
  - "Hybrid Search" (Pinecone):
    https://www.pinecone.io/learn/hybrid-search-intro/
  - "Reciprocal Rank Fusion" explained:
    https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf
  - VIDEO: "RAG From Scratch — Part 9: Retrieval" (LangChain):
    https://www.youtube.com/watch?v=kl6NwWYxvbM
  - VIDEO: "Advanced RAG — Hybrid Search" (James Briggs):
    https://www.youtube.com/watch?v=lYxGYXjfrNI
  - VIDEO: "Ensemble Retriever Tutorial" (LangChain):
    https://www.youtube.com/watch?v=zDMBsYtKPHY
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from agentexplorr.core import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Protocol for vector stores (structural typing)
# ---------------------------------------------------------------------------


@runtime_checkable
class VectorStoreProtocol(Protocol):
    """Protocol defining the interface that any vector store must satisfy.

    WHY A PROTOCOL?
      Python's ``Protocol`` (PEP 544) enables structural typing: any class
      with a ``search`` method matching this signature is accepted, regardless
      of its inheritance hierarchy.  This is more flexible than requiring a
      specific base class.

      Learn more: https://docs.python.org/3/library/typing.html#typing.Protocol
      VIDEO: "Protocols in Python" — https://www.youtube.com/watch?v=xvb5hGLoK0A
    """

    def search(self, query: str, top_k: int = 5) -> list[Any]:
        """Search the store and return results with chunk_id, text, score, metadata."""
        ...  # pragma: no cover


# ---------------------------------------------------------------------------
# Unified Search Result (store-agnostic)
# ---------------------------------------------------------------------------


@dataclass
class RetrievalResult:
    """A search result from the HybridRetriever.

    This is a store-agnostic result that normalizes the output from different
    vector stores into a common format.

    Attributes:
        chunk_id:  Unique chunk identifier.
        text:      The chunk text content.
        score:     Final re-ranked similarity score (0-1, higher = better).
        metadata:  Chunk metadata (source, page, etc.).
        source_store: Name/type of the store that returned this result.
                      Useful for debugging which store found what.
    """

    chunk_id: str = ""
    text: str = ""
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    source_store: str = ""

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", "\\n")
        return (
            f"RetrievalResult(score={self.score:.4f}, "
            f'store={self.source_store!r}, text="{preview}...")'
        )


# ---------------------------------------------------------------------------
# Hybrid Retriever
# ---------------------------------------------------------------------------


class HybridRetriever:
    """Query multiple vector stores and combine results with re-ranking.

    The HybridRetriever acts as a unified search interface across any number
    of vector stores (ChromaDB, FAISS, or any future implementation).

    Args:
        stores:     Dict mapping store names to store instances.
                    Example: ``{"chroma": chroma_store, "faiss": faiss_store}``
        strategy:   Re-ranking strategy.  Options:
                    - ``"score"`` — Sort by raw similarity score (default).
                    - ``"rrf"``   — Reciprocal Rank Fusion.
        rrf_k:      The K parameter for RRF scoring (default 60).
                    Higher values give more weight to lower-ranked results.
        weights:    Optional per-store weights for score-based ranking.
                    Example: ``{"chroma": 0.6, "faiss": 0.4}``.
                    Weights are normalized to sum to 1.0.

    Usage::

        from agentexplorr.rag.retriever import HybridRetriever

        retriever = HybridRetriever(
            stores={"chroma": chroma_store, "faiss": faiss_store},
            strategy="rrf",
        )

        results = retriever.search("What is attention in transformers?", top_k=10)
        for r in results:
            print(f"[{r.score:.3f}] [{r.source_store}] {r.text[:80]}")
    """

    VALID_STRATEGIES = ("score", "rrf")

    def __init__(
        self,
        stores: dict[str, Any] | None = None,
        strategy: str = "score",
        rrf_k: int = 60,
        weights: dict[str, float] | None = None,
    ) -> None:
        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"Unknown strategy {strategy!r}. "
                f"Valid: {', '.join(self.VALID_STRATEGIES)}"
            )

        self._stores: dict[str, Any] = stores or {}
        self._strategy = strategy
        self._rrf_k = rrf_k
        self._weights = weights or {}

        # Validate that all stores have a .search() method
        for name, store in self._stores.items():
            if not hasattr(store, "search"):
                raise TypeError(
                    f"Store {name!r} does not have a 'search' method. "
                    f"It must implement the VectorStoreProtocol."
                )

        logger.info(
            "hybrid_retriever_initialized",
            stores=list(self._stores.keys()),
            strategy=strategy,
        )

    def add_store(self, name: str, store: Any) -> None:
        """Register a new vector store.

        Args:
            name:  A human-readable name for this store (e.g. "chroma", "faiss").
            store: A vector store instance with a ``search(query, top_k)`` method.
        """
        if not hasattr(store, "search"):
            raise TypeError(f"Store {name!r} must have a 'search' method.")

        self._stores[name] = store
        logger.info("store_added_to_retriever", store_name=name)

    def remove_store(self, name: str) -> None:
        """Unregister a vector store.

        Args:
            name: The name of the store to remove.

        Raises:
            KeyError: If the store name is not found.
        """
        if name not in self._stores:
            raise KeyError(f"Store {name!r} not found. Available: {list(self._stores.keys())}")
        del self._stores[name]
        logger.info("store_removed_from_retriever", store_name=name)

    def search(
        self,
        query: str,
        top_k: int = 5,
        per_store_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Search across all registered vector stores and return re-ranked results.

        THE SEARCH FLOW:
          1. Query each store for ``per_store_k`` results (default: ``top_k * 2``
             to give the re-ranker more candidates to work with).
          2. Convert all results to a common ``RetrievalResult`` format.
          3. Deduplicate by chunk_id (keep the highest-scoring occurrence).
          4. Re-rank using the selected strategy (score or RRF).
          5. Return the top ``top_k`` results.

        Args:
            query:       The search query string.
            top_k:       Number of final results to return.
            per_store_k: Number of results to request from each store.
                         Defaults to ``top_k * 2`` to give the re-ranker
                         a larger candidate pool.

        Returns:
            List of RetrievalResult objects, sorted by score (descending).
        """
        if not self._stores:
            logger.warning("search_called_with_no_stores")
            return []

        if not query.strip():
            logger.warning("search_called_with_empty_query")
            return []

        # Request more from each store than we need, to give re-ranking
        # a richer candidate pool.
        store_k = per_store_k or (top_k * 2)

        logger.info(
            "hybrid_search_starting",
            query_preview=query[:100],
            top_k=top_k,
            per_store_k=store_k,
            stores=list(self._stores.keys()),
        )

        # Step 1: Query each store
        all_results: list[tuple[str, Any]] = []  # (store_name, result)
        for store_name, store in self._stores.items():
            try:
                store_results = store.search(query, top_k=store_k)
                for result in store_results:
                    all_results.append((store_name, result))
                logger.debug(
                    "store_search_complete",
                    store=store_name,
                    results=len(store_results),
                )
            except Exception as exc:
                # Don't let one failing store break the entire search
                logger.error(
                    "store_search_failed",
                    store=store_name,
                    error=str(exc),
                )

        if not all_results:
            logger.warning("no_results_from_any_store")
            return []

        # Step 2: Convert to RetrievalResult
        candidates = self._normalize_results(all_results)

        # Step 3: Deduplicate by chunk_id
        deduplicated = self._deduplicate(candidates)

        # Step 4: Re-rank
        if self._strategy == "rrf":
            ranked = self._rerank_rrf(all_results, deduplicated, top_k)
        else:
            ranked = self._rerank_score(deduplicated, top_k)

        logger.info(
            "hybrid_search_complete",
            total_candidates=len(all_results),
            deduplicated=len(deduplicated),
            returned=len(ranked),
        )

        return ranked

    def _normalize_results(
        self, raw_results: list[tuple[str, Any]]
    ) -> list[RetrievalResult]:
        """Convert store-specific results to RetrievalResult objects.

        Each vector store returns its own result type (ChromaDB's SearchResult,
        FAISS's SearchResult).  This method normalizes them to our common
        ``RetrievalResult`` format by accessing common attributes.
        """
        normalized: list[RetrievalResult] = []
        for store_name, result in raw_results:
            # Access attributes that both ChromaDB and FAISS SearchResult have
            retrieval_result = RetrievalResult(
                chunk_id=getattr(result, "chunk_id", ""),
                text=getattr(result, "text", ""),
                score=float(getattr(result, "score", 0.0)),
                metadata=dict(getattr(result, "metadata", {})),
                source_store=store_name,
            )

            # Apply per-store weight if configured
            if store_name in self._weights:
                retrieval_result.score *= self._weights[store_name]

            normalized.append(retrieval_result)
        return normalized

    @staticmethod
    def _deduplicate(results: list[RetrievalResult]) -> list[RetrievalResult]:
        """Remove duplicate chunks, keeping the highest-scoring occurrence.

        WHY DEDUPLICATE?
          The same chunk may exist in both ChromaDB and FAISS.  We don't
          want it to appear twice in the results.  We keep the occurrence
          with the highest score (best match from any store).
        """
        best_by_id: dict[str, RetrievalResult] = {}
        for result in results:
            existing = best_by_id.get(result.chunk_id)
            if existing is None or result.score > existing.score:
                best_by_id[result.chunk_id] = result
        return list(best_by_id.values())

    @staticmethod
    def _rerank_score(
        results: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Re-rank by raw similarity score (simple sort).

        This is the simplest re-ranking: just sort by score descending.
        It works well when all stores use the same embedding model and
        similar distance metrics.
        """
        sorted_results = sorted(results, key=lambda r: r.score, reverse=True)
        return sorted_results[:top_k]

    def _rerank_rrf(
        self,
        raw_results: list[tuple[str, Any]],
        deduplicated: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Re-rank using Reciprocal Rank Fusion (RRF).

        HOW RRF WORKS:
          For each result from each store, compute:
              rrf_score = 1 / (k + rank)

          where ``rank`` is the result's position in that store's ranked list
          (1-indexed) and ``k`` is a smoothing constant (default 60).

          The final score for each chunk is the SUM of its RRF scores across
          all stores.  This means:
            - A chunk ranked #1 in two stores gets a higher score than one
              ranked #1 in one store.
            - Rank matters more than raw score — this is robust to scale
              differences between stores.

        EXAMPLE:
          Chunk "abc" is ranked #1 in ChromaDB, #3 in FAISS:
            rrf_score = 1/(60+1) + 1/(60+3) = 0.0164 + 0.0159 = 0.0323

          Chunk "xyz" is ranked #2 in ChromaDB only:
            rrf_score = 1/(60+2) = 0.0161

          "abc" wins because it appears in both stores.

        Args:
            raw_results:  The original (store_name, result) tuples.
            deduplicated: Deduplicated RetrievalResult objects.
            top_k:        Number of results to return.

        Returns:
            Re-ranked list of RetrievalResult objects.
        """
        # Build per-store ranked lists
        store_rankings: dict[str, list[str]] = {}
        for store_name, result in raw_results:
            if store_name not in store_rankings:
                store_rankings[store_name] = []
            chunk_id = getattr(result, "chunk_id", "")
            if chunk_id not in store_rankings[store_name]:
                store_rankings[store_name].append(chunk_id)

        # Compute RRF score for each chunk
        rrf_scores: dict[str, float] = {}
        for _store_name, ranked_ids in store_rankings.items():
            for rank, chunk_id in enumerate(ranked_ids, start=1):
                score = 1.0 / (self._rrf_k + rank)
                rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + score

        # Update the RetrievalResult scores
        id_to_result: dict[str, RetrievalResult] = {
            r.chunk_id: r for r in deduplicated
        }

        for chunk_id, rrf_score in rrf_scores.items():
            if chunk_id in id_to_result:
                id_to_result[chunk_id].score = rrf_score

        # Sort by RRF score descending
        sorted_results = sorted(
            id_to_result.values(),
            key=lambda r: r.score,
            reverse=True,
        )

        return sorted_results[:top_k]

    def __repr__(self) -> str:
        return (
            f"HybridRetriever(stores={list(self._stores.keys())}, "
            f"strategy={self._strategy!r})"
        )
