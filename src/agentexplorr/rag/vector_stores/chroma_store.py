"""
ChromaDB Vector Store — Persistent Semantic Search
====================================================

WHAT IS CHROMADB?
  ChromaDB is an open-source embedding database designed for AI applications.
  Think of it as "SQLite for embeddings" — it handles persistence, indexing,
  metadata filtering, and similarity search out of the box, with minimal setup.

HOW IT WORKS:
  1. You create a **collection** (like a table in SQL).
  2. You add documents with their embeddings and metadata.
  3. To search, you provide a query embedding and ChromaDB returns the K
     most similar documents using approximate nearest neighbor (ANN) search.
  4. You can filter results by metadata (e.g. "only docs from 2024").

WHY CHROMADB FOR RAG?
  - Zero-config persistence: data survives restarts without extra setup.
  - Metadata filtering: filter by source file, date, category, etc.
  - Built-in deduplication: adding the same ID twice updates the existing doc.
  - Python-native: no external server needed (though it supports client-server).
  - Active community and good documentation.

ARCHITECTURE:
  ChromaDB uses HNSW (Hierarchical Navigable Small World) graphs internally
  for ANN search.  HNSW is one of the fastest ANN algorithms:
    - Build time: O(N log N)
    - Query time: O(log N)
    - Memory: O(N × dimension)

LEARNING RESOURCES:
  - ChromaDB docs:
    https://docs.trychroma.com/
  - ChromaDB getting started:
    https://docs.trychroma.com/docs/overview/getting-started
  - ChromaDB GitHub:
    https://github.com/chroma-core/chroma
  - VIDEO: "ChromaDB Tutorial — Vector Database for AI" (pixegami):
    https://www.youtube.com/watch?v=QSW2L8dkaZk
  - VIDEO: "Build a RAG App with ChromaDB" (Tech With Tim):
    https://www.youtube.com/watch?v=tcqEUSNCn8I
  - HNSW paper (the algorithm ChromaDB uses):
    https://arxiv.org/abs/1603.09320
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentexplorr.core import get_logger
from agentexplorr.rag.chunking import Chunk

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Search Result
# ---------------------------------------------------------------------------


@dataclass
class SearchResult:
    """A single result from a vector store search.

    Attributes:
        chunk_id:  The unique identifier of the matching chunk.
        text:      The text content of the matching chunk.
        score:     Similarity score (higher = more similar).  ChromaDB returns
                   *distances* (lower = more similar), so we convert to
                   similarity:  ``score = 1.0 / (1.0 + distance)`` for L2, or
                   ``score = 1.0 - distance`` for cosine.
        metadata:  The metadata dict stored with the chunk.
    """

    chunk_id: str = ""
    text: str = ""
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# ChromaDB Vector Store
# ---------------------------------------------------------------------------


class ChromaVectorStore:
    """Vector store backed by ChromaDB for persistent semantic search.

    This class wraps ChromaDB's collection API and provides a clean interface
    for the rest of the RAG pipeline.  It uses OUR ``EmbeddingModel`` class
    for generating embeddings (rather than ChromaDB's built-in embedding
    function) so that all components share the same embedding space.

    Args:
        collection_name:  Name of the ChromaDB collection (like a table name).
        persist_directory: Directory where ChromaDB stores its data on disk.
                           If ``None``, uses an in-memory (ephemeral) store.
        embedding_model:   An ``EmbeddingModel`` instance for generating vectors.
                           If ``None``, a default one will be created.

    Usage::

        from agentexplorr.rag.embeddings import EmbeddingModel
        from agentexplorr.rag.vector_stores.chroma_store import ChromaVectorStore

        model = EmbeddingModel()
        store = ChromaVectorStore(
            collection_name="my_docs",
            persist_directory="./chroma_data",
            embedding_model=model,
        )

        # Add chunks
        store.add_documents(chunks)

        # Search
        results = store.search("What is attention?", top_k=5)
        for r in results:
            print(f"[{r.score:.3f}] {r.text[:100]}")

        # Delete
        store.delete(["chunk_id_1", "chunk_id_2"])
    """

    def __init__(
        self,
        collection_name: str = "agentexplorr_rag",
        persist_directory: str | Path | None = None,
        embedding_model: Any | None = None,
    ) -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise ImportError(
                "ChromaVectorStore requires the 'chromadb' package. "
                "Install with:  pip install chromadb\n"
                "Docs: https://docs.trychroma.com/"
            ) from exc

        self.collection_name = collection_name

        # Create the ChromaDB client.
        # - If persist_directory is set, data is stored on disk (survives restarts).
        # - If None, data lives only in memory (fast for testing).
        if persist_directory is not None:
            persist_path = Path(persist_directory).resolve()
            persist_path.mkdir(parents=True, exist_ok=True)

            logger.info(
                "creating_persistent_chroma_client",
                path=str(persist_path),
            )
            self._client = chromadb.PersistentClient(path=str(persist_path))
        else:
            logger.info("creating_ephemeral_chroma_client")
            self._client = chromadb.EphemeralClient()

        # Get or create the collection.
        # ``get_or_create_collection`` is idempotent — safe to call repeatedly.
        # We use cosine distance (the default and most common choice for text).
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # Use cosine similarity
        )

        # Set up the embedding model.
        # We lazily import to avoid circular dependencies.
        if embedding_model is not None:
            self._embedding_model = embedding_model
        else:
            from agentexplorr.rag.embeddings import EmbeddingModel

            self._embedding_model = EmbeddingModel()

        logger.info(
            "chroma_store_initialized",
            collection=collection_name,
            existing_count=self._collection.count(),
        )

    @property
    def count(self) -> int:
        """Return the number of documents in the collection."""
        return self._collection.count()

    def add_documents(
        self,
        chunks: list[Chunk],
        batch_size: int = 100,
    ) -> list[str]:
        """Add document chunks to the vector store.

        HOW THIS WORKS:
          1. Extract text from each Chunk.
          2. Generate embeddings using our EmbeddingModel.
          3. Upsert into ChromaDB (add if new, update if ID exists).

        ChromaDB's ``upsert`` is idempotent: adding the same chunk_id twice
        overwrites the previous entry.  This is useful when re-processing
        documents — you don't get duplicates.

        Args:
            chunks:     List of Chunk objects to store.
            batch_size: Number of chunks to upsert in one call.  ChromaDB
                        has a per-call limit, so we batch.

        Returns:
            List of chunk IDs that were added.

        Raises:
            ValueError: If chunks list is empty.
        """
        if not chunks:
            logger.warning("add_documents_called_with_empty_list")
            return []

        logger.info(
            "adding_documents_to_chroma",
            count=len(chunks),
            collection=self.collection_name,
        )

        # Extract the components ChromaDB needs:
        #   - ids: unique identifiers (our chunk_id)
        #   - documents: the raw text (ChromaDB stores this for retrieval)
        #   - embeddings: the vector representations
        #   - metadatas: metadata dicts
        texts = [chunk.text for chunk in chunks]
        ids = [chunk.chunk_id for chunk in chunks]

        # Sanitize metadata: ChromaDB only supports str, int, float, bool values.
        # We convert anything else to a string representation.
        metadatas = [self._sanitize_metadata(chunk.metadata) for chunk in chunks]

        # Generate embeddings in batch for efficiency
        embeddings = self._embedding_model.embed_batch(texts)

        # Upsert in batches to respect ChromaDB's per-call limits
        added_ids: list[str] = []
        for i in range(0, len(chunks), batch_size):
            batch_end = min(i + batch_size, len(chunks))
            self._collection.upsert(
                ids=ids[i:batch_end],
                documents=texts[i:batch_end],
                embeddings=embeddings[i:batch_end],
                metadatas=metadatas[i:batch_end],
            )
            added_ids.extend(ids[i:batch_end])

        logger.info(
            "documents_added_to_chroma",
            count=len(added_ids),
            total_in_collection=self._collection.count(),
        )

        return added_ids

    def search(
        self,
        query: str,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for the most similar documents to a query.

        HOW CHROMADB SEARCH WORKS:
          1. The query text is embedded into a vector.
          2. ChromaDB's HNSW index finds the ``top_k`` nearest vectors.
          3. Results are returned with distances and metadata.

        METADATA FILTERING (the ``where`` parameter):
          ChromaDB supports SQL-like filters on metadata.  Examples:
            - ``{"source": "paper.pdf"}``         → exact match
            - ``{"page": {"$gt": 5}}``            → page > 5
            - ``{"$and": [{"source": "a.pdf"}, {"page": {"$lte": 10}}]}``

          Full filter syntax:
          https://docs.trychroma.com/docs/collections/filter

        Args:
            query:  The search query string.
            top_k:  Number of results to return (default 5).
            where:  Optional metadata filter dict.

        Returns:
            List of SearchResult objects, sorted by score (descending).
        """
        if not query.strip():
            logger.warning("search_called_with_empty_query")
            return []

        logger.debug(
            "searching_chroma",
            query_preview=query[:100],
            top_k=top_k,
            collection=self.collection_name,
        )

        # Embed the query using the same model used for documents.
        # This is CRITICAL: query and document embeddings must come from the
        # same model, or similarity scores will be meaningless.
        query_embedding = self._embedding_model.embed_text(query)

        # Build the query kwargs
        query_kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, self._collection.count()) if self._collection.count() > 0 else top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where is not None:
            query_kwargs["where"] = where

        # Execute the search
        try:
            results = self._collection.query(**query_kwargs)
        except Exception as exc:
            logger.error("chroma_search_failed", error=str(exc))
            return []

        # Parse ChromaDB's response format.
        # ChromaDB returns lists-of-lists because it supports multi-query.
        # Since we only send one query, we take index [0] from each.
        search_results: list[SearchResult] = []

        if results and results["ids"] and results["ids"][0]:
            ids = results["ids"][0]
            documents = results["documents"][0] if results["documents"] else [""] * len(ids)
            metadatas = results["metadatas"][0] if results["metadatas"] else [{}] * len(ids)
            distances = results["distances"][0] if results["distances"] else [0.0] * len(ids)

            for chunk_id, text, meta, distance in zip(ids, documents, metadatas, distances):
                # Convert ChromaDB distance to similarity score.
                # ChromaDB returns cosine *distance* (0 = identical, 2 = opposite).
                # We convert to similarity: score = 1.0 - (distance / 2.0)
                # This gives us a 0-1 range where 1 = identical.
                score = 1.0 - (distance / 2.0)

                search_results.append(
                    SearchResult(
                        chunk_id=chunk_id,
                        text=text or "",
                        score=score,
                        metadata=meta or {},
                    )
                )

        # Sort by score descending (most similar first)
        search_results.sort(key=lambda r: r.score, reverse=True)

        logger.debug(
            "chroma_search_complete",
            results=len(search_results),
            top_score=search_results[0].score if search_results else 0.0,
        )

        return search_results

    def delete(self, chunk_ids: list[str]) -> None:
        """Delete documents from the collection by their chunk IDs.

        Args:
            chunk_ids: List of chunk IDs to delete.

        Raises:
            ValueError: If chunk_ids is empty.
        """
        if not chunk_ids:
            logger.warning("delete_called_with_empty_list")
            return

        logger.info(
            "deleting_from_chroma",
            count=len(chunk_ids),
            collection=self.collection_name,
        )

        try:
            self._collection.delete(ids=chunk_ids)
        except Exception as exc:
            logger.error("chroma_delete_failed", error=str(exc), ids=chunk_ids[:5])
            raise

        logger.info(
            "deleted_from_chroma",
            count=len(chunk_ids),
            remaining=self._collection.count(),
        )

    def clear(self) -> None:
        """Delete ALL documents from the collection.

        WARNING: This is destructive and irreversible!  Use with caution.
        Useful for testing or resetting a knowledge base.
        """
        logger.warning(
            "clearing_entire_collection",
            collection=self.collection_name,
            count=self._collection.count(),
        )
        # ChromaDB doesn't have a "clear" method, so we delete and recreate
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        """Sanitize metadata for ChromaDB compatibility.

        ChromaDB metadata values must be str, int, float, or bool.
        Lists, dicts, None, and other types are converted to strings.

        This prevents cryptic ChromaDB errors like:
          "Expected metadata value to be a str, int, float or bool"
        """
        sanitized: dict[str, Any] = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            elif value is None:
                sanitized[key] = ""
            else:
                # Convert complex types (lists, dicts) to their string repr
                sanitized[key] = str(value)
        return sanitized

    def __repr__(self) -> str:
        return (
            f"ChromaVectorStore(collection={self.collection_name!r}, "
            f"count={self.count})"
        )
