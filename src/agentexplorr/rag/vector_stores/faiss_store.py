"""
FAISS Vector Store — High-Performance Similarity Search
=========================================================

WHAT IS FAISS?
  FAISS (Facebook AI Similarity Search) is a library developed by Meta AI
  Research for efficient similarity search and clustering of dense vectors.
  It is the gold standard for vector search at scale — used internally at
  Meta to search billions of vectors in milliseconds.

HOW FAISS WORKS (the key concepts):

  **Index Types** — FAISS provides multiple index structures, each with
  different speed/accuracy/memory trade-offs:

    - ``IndexFlatL2``:  Brute-force L2 distance search.  100% accurate but
      O(N) per query.  Best for < 100K vectors.  This is what we use here.

    - ``IndexFlatIP``:  Brute-force inner product (dot product) search.
      Equivalent to cosine similarity when vectors are normalized.

    - ``IndexIVFFlat``:  Inverted file index.  Partitions vectors into clusters,
      only searches relevant clusters.  10-100x faster for large datasets.

    - ``IndexHNSWFlat``:  HNSW graph-based index.  Best speed/accuracy ratio
      for most real-world workloads.

    - ``IndexIVFPQ``:  Combines inverted file with product quantization for
      compressed storage.  Handles billions of vectors in limited RAM.

  **Distance Metrics**:
    - L2 (Euclidean):  distance = sum((a_i - b_i)^2).  Lower = more similar.
    - Inner Product:   score = sum(a_i * b_i).  Higher = more similar.
      When vectors are L2-normalized, inner product = cosine similarity.

  **The Search Flow**:
    1. Add N vectors to the index:  ``index.add(numpy_array)``
    2. Query with K vectors:         ``distances, indices = index.search(query, k)``
    3. ``indices[i][j]`` = the j-th nearest neighbor of query i
    4. ``distances[i][j]`` = the distance to that neighbor

WHY FAISS FOR RAG?
  - Blazing fast: optimized C++ with optional GPU acceleration.
  - Battle-tested: used in production at Meta for search, recommendations.
  - Flexible: choose the right index for your scale and accuracy needs.
  - No external server: it's a library, not a service.

LEARNING RESOURCES:
  - FAISS wiki (excellent!):
    https://github.com/facebookresearch/faiss/wiki
  - FAISS tutorial notebooks:
    https://github.com/facebookresearch/faiss/tree/main/tutorial/python
  - VIDEO: "FAISS — Introduction to Similarity Search" (James Briggs):
    https://www.youtube.com/watch?v=sKyvsdEv6rk
  - VIDEO: "Vector Search with FAISS" (Aladdin Persson):
    https://www.youtube.com/watch?v=kJa2GePvGBs
  - VIDEO: "Approximate Nearest Neighbors — HNSW" (James Briggs):
    https://www.youtube.com/watch?v=QvKMwLjdK-s

PAPERS:
  - "Billion-scale similarity search with GPUs" (Johnson, Douze, Jegou, 2019):
    https://arxiv.org/abs/1702.08734
  - "Product Quantization for Nearest Neighbor Search" (Jegou et al., 2011):
    https://ieeexplore.ieee.org/document/5432202
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentexplorr.core import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Search Result (same interface as chroma_store for interchangeability)
# ---------------------------------------------------------------------------


@dataclass
class SearchResult:
    """A single result from a FAISS search.

    Attributes:
        chunk_id: Unique identifier of the matching chunk.
        text:     The text content of the matching chunk.
        score:    Similarity score (0-1, higher = more similar).
                  Computed as ``1 / (1 + L2_distance)`` for L2 index,
                  or raw inner product for IP index.
        metadata: The metadata dict associated with the chunk.
    """

    chunk_id: str = ""
    text: str = ""
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal document record (stored alongside the FAISS index)
# ---------------------------------------------------------------------------


@dataclass
class _DocumentRecord:
    """Internal record linking a FAISS integer index to our chunk data.

    WHY DO WE NEED THIS?
      FAISS only stores vectors — it has no concept of text, metadata, or IDs.
      It returns integer indices (the position in which the vector was added).
      We maintain a parallel list mapping those indices to our richer Chunk data.
    """

    chunk_id: str
    text: str
    metadata: dict[str, Any]


# ---------------------------------------------------------------------------
# FAISS Vector Store
# ---------------------------------------------------------------------------


class FAISSVectorStore:
    """Vector store backed by FAISS for high-performance similarity search.

    This class manages a FAISS index and a parallel metadata store.  It
    provides the same add/search interface as ``ChromaVectorStore`` so they
    can be used interchangeably in the ``HybridRetriever``.

    IMPORTANT ARCHITECTURAL NOTE:
      FAISS itself only stores and searches raw float vectors.  All text,
      metadata, and chunk IDs are stored in a separate Python data structure
      (``_documents``) that we serialize alongside the FAISS index.

    Args:
        dimension:       The dimension of the embedding vectors.  Must match
                         the embedding model's output dimension (384 for
                         all-MiniLM-L6-v2).  If ``None``, auto-detected from
                         the first batch of added documents.
        index_type:      Which FAISS index to use.  Currently supported:
                         - ``"flat_l2"``:  Exact L2 search (default, 100% recall).
                         - ``"flat_ip"``:  Exact inner product search.
        embedding_model: An ``EmbeddingModel`` instance.  If ``None``, a
                         default one is created.

    Usage::

        from agentexplorr.rag.embeddings import EmbeddingModel
        from agentexplorr.rag.vector_stores.faiss_store import FAISSVectorStore

        model = EmbeddingModel()
        store = FAISSVectorStore(dimension=384, embedding_model=model)

        # Add chunks
        store.add_documents(chunks)

        # Search
        results = store.search("What is attention?", top_k=5)
        for r in results:
            print(f"[{r.score:.3f}] {r.text[:80]}")

        # Save to disk
        store.save("./faiss_index/")

        # Load from disk
        store = FAISSVectorStore.load("./faiss_index/", embedding_model=model)
    """

    def __init__(
        self,
        dimension: int | None = None,
        index_type: str = "flat_l2",
        embedding_model: Any | None = None,
    ) -> None:
        try:
            import faiss  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "FAISSVectorStore requires the 'faiss-cpu' package. "
                "Install with:  pip install faiss-cpu\n"
                "For GPU support:  pip install faiss-gpu\n"
                "Docs: https://github.com/facebookresearch/faiss/wiki"
            ) from exc

        self._dimension = dimension
        self._index_type = index_type
        self._index: Any = None  # The FAISS index (created when dimension is known)

        # Parallel metadata store: position i in _documents corresponds to
        # vector i in the FAISS index.
        self._documents: list[_DocumentRecord] = []

        # Mapping from chunk_id → index position for O(1) lookup
        self._id_to_position: dict[str, int] = {}

        # Set up the embedding model
        if embedding_model is not None:
            self._embedding_model = embedding_model
        else:
            from agentexplorr.rag.embeddings import EmbeddingModel

            self._embedding_model = EmbeddingModel()

        # If dimension is known, create the index now
        if self._dimension is not None:
            self._create_index()

        logger.info(
            "faiss_store_initialized",
            dimension=self._dimension,
            index_type=self._index_type,
        )

    def _create_index(self) -> None:
        """Create the FAISS index with the specified dimension and type.

        INDEX TYPES EXPLAINED:

        ``IndexFlatL2``:
          - Brute-force search computing L2 distance to every vector.
          - 100% recall (never misses the true nearest neighbor).
          - O(N * dimension) per query — fine for < 100K vectors.
          - No training needed.

        ``IndexFlatIP``:
          - Same as FlatL2 but uses inner product instead of L2 distance.
          - When vectors are L2-normalized (our default), IP = cosine similarity.
          - Useful when you want cosine similarity scores directly.
        """
        import faiss

        if self._dimension is None or self._dimension <= 0:
            raise ValueError(
                f"Cannot create FAISS index: dimension must be positive, "
                f"got {self._dimension}"
            )

        if self._index_type == "flat_l2":
            self._index = faiss.IndexFlatL2(self._dimension)
        elif self._index_type == "flat_ip":
            self._index = faiss.IndexFlatIP(self._dimension)
        else:
            raise ValueError(
                f"Unsupported index_type: {self._index_type!r}. "
                f"Supported: 'flat_l2', 'flat_ip'"
            )

        logger.debug(
            "faiss_index_created",
            dimension=self._dimension,
            index_type=self._index_type,
        )

    @property
    def count(self) -> int:
        """Return the number of vectors in the index."""
        if self._index is None:
            return 0
        return self._index.ntotal

    def add_documents(
        self,
        chunks: list[Any],
        batch_size: int = 256,
    ) -> list[str]:
        """Add document chunks to the FAISS index.

        HOW THIS WORKS:
          1. Extract text from each Chunk.
          2. Generate embeddings using our EmbeddingModel (in batch for speed).
          3. Convert to a numpy float32 array (FAISS requires this specific dtype).
          4. Add to the FAISS index.
          5. Store the chunk text and metadata in our parallel ``_documents`` list.

        WHY float32?
          FAISS is a C++ library that expects float32 arrays.  Using float64
          (Python's default) would silently produce wrong results or crash.
          Always ``astype(np.float32)``!

        Args:
            chunks:     List of Chunk objects to add.
            batch_size: Batch size for embedding generation.

        Returns:
            List of chunk IDs that were added.
        """
        import numpy as np

        if not chunks:
            logger.warning("add_documents_called_with_empty_list")
            return []

        texts = [chunk.text for chunk in chunks]
        ids = [chunk.chunk_id for chunk in chunks]

        logger.info(
            "adding_documents_to_faiss",
            count=len(chunks),
        )

        # Generate embeddings
        embeddings = self._embedding_model.embed_batch(texts, batch_size=batch_size)

        # Convert to numpy float32 array — CRITICAL for FAISS compatibility
        embedding_array = np.array(embeddings, dtype=np.float32)

        # Auto-detect dimension from first batch if not set
        if self._dimension is None:
            self._dimension = embedding_array.shape[1]
            self._create_index()

        # Validate dimension matches
        if embedding_array.shape[1] != self._dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self._dimension}, "
                f"got {embedding_array.shape[1]}. "
                f"Are you using a different embedding model?"
            )

        # Add vectors to the FAISS index.
        # FAISS assigns sequential integer IDs starting from 0.
        # Vector at position ``self._index.ntotal + i`` corresponds to
        # ``_documents[self._index.ntotal + i]``.
        start_position = self._index.ntotal
        self._index.add(embedding_array)

        # Store metadata in our parallel list
        added_ids: list[str] = []
        for i, chunk in enumerate(chunks):
            position = start_position + i
            record = _DocumentRecord(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                metadata=chunk.metadata,
            )
            self._documents.append(record)
            self._id_to_position[chunk.chunk_id] = position
            added_ids.append(chunk.chunk_id)

        logger.info(
            "documents_added_to_faiss",
            count=len(added_ids),
            total_in_index=self._index.ntotal,
        )

        return added_ids

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Search for the most similar documents to a query.

        HOW FAISS SEARCH WORKS:
          1. The query is embedded into a vector.
          2. The vector is passed to ``index.search(query_vector, k)``.
          3. FAISS returns two arrays:
             - ``distances[0]``: the distance to each of the K nearest neighbors.
             - ``indices[0]``:   the integer positions of those neighbors.
          4. We map positions back to our chunk data via ``_documents[position]``.

        DISTANCE → SIMILARITY CONVERSION:
          FAISS returns raw distances.  We convert to a 0-1 similarity score
          so that results from FAISS and ChromaDB are comparable:
            - L2:  ``score = 1.0 / (1.0 + distance)``   (0 distance → score 1.0)
            - IP:  ``score = max(0, distance)``           (already a similarity)

        Args:
            query: The search query string.
            top_k: Number of results to return.

        Returns:
            List of SearchResult objects sorted by score (descending).
        """
        import numpy as np

        if self._index is None or self._index.ntotal == 0:
            logger.warning("searching_empty_faiss_index")
            return []

        if not query.strip():
            logger.warning("search_called_with_empty_query")
            return []

        logger.debug(
            "searching_faiss",
            query_preview=query[:100],
            top_k=top_k,
            index_size=self._index.ntotal,
        )

        # Embed the query
        query_embedding = self._embedding_model.embed_text(query)

        # Convert to numpy float32 array with shape (1, dimension)
        # FAISS expects a 2D array even for a single query
        query_array = np.array([query_embedding], dtype=np.float32)

        # Clamp top_k to the number of vectors in the index
        actual_k = min(top_k, self._index.ntotal)

        # Execute the search
        distances, indices = self._index.search(query_array, actual_k)

        # Parse results.
        # distances[0] and indices[0] are 1D arrays (one row per query; we have 1).
        results: list[SearchResult] = []
        for distance, idx in zip(distances[0], indices[0]):
            # FAISS returns -1 for padding when there are fewer results than k
            if idx == -1:
                continue

            idx_int = int(idx)
            if idx_int < 0 or idx_int >= len(self._documents):
                logger.warning("faiss_returned_invalid_index", index=idx_int)
                continue

            record = self._documents[idx_int]

            # Convert distance to similarity score
            if self._index_type == "flat_l2":
                # L2 distance: 0 = identical, larger = more different
                # Convert to 0-1 score where 1 = identical
                score = 1.0 / (1.0 + float(distance))
            else:
                # Inner product: higher = more similar
                # Clamp to [0, 1] for normalized vectors
                score = max(0.0, min(1.0, float(distance)))

            results.append(
                SearchResult(
                    chunk_id=record.chunk_id,
                    text=record.text,
                    score=score,
                    metadata=record.metadata,
                )
            )

        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)

        logger.debug(
            "faiss_search_complete",
            results=len(results),
            top_score=results[0].score if results else 0.0,
        )

        return results

    def save(self, directory: str | Path) -> None:
        """Save the FAISS index and metadata to disk.

        This creates two files:
          - ``index.faiss``: The FAISS index binary (vectors + index structure).
          - ``metadata.pkl``: Python pickle of the document records and config.

        WHY TWO FILES?
          FAISS has its own efficient binary serialization (``write_index``).
          Our metadata (text, chunk IDs) is pure Python, so we use pickle.
          Keeping them separate means you could rebuild metadata without
          re-computing embeddings.

        Args:
            directory: Directory to save the files in.  Created if it doesn't exist.
        """
        import faiss

        if self._index is None:
            raise RuntimeError("Cannot save: no index has been created yet.")

        save_dir = Path(directory).resolve()
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save the FAISS index
        index_path = save_dir / "index.faiss"
        faiss.write_index(self._index, str(index_path))

        # Save the metadata
        metadata_path = save_dir / "metadata.pkl"
        metadata = {
            "documents": self._documents,
            "id_to_position": self._id_to_position,
            "dimension": self._dimension,
            "index_type": self._index_type,
        }
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)

        logger.info(
            "faiss_index_saved",
            directory=str(save_dir),
            index_size=self._index.ntotal,
        )

    @classmethod
    def load(
        cls,
        directory: str | Path,
        embedding_model: Any | None = None,
    ) -> FAISSVectorStore:
        """Load a saved FAISS index and metadata from disk.

        Args:
            directory:       Directory containing ``index.faiss`` and ``metadata.pkl``.
            embedding_model: An ``EmbeddingModel`` instance for search queries.

        Returns:
            A new FAISSVectorStore instance with the loaded data.

        Raises:
            FileNotFoundError: If the required files are missing.
        """
        import faiss

        load_dir = Path(directory).resolve()
        index_path = load_dir / "index.faiss"
        metadata_path = load_dir / "metadata.pkl"

        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        # Load metadata first to get dimension and index_type
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)

        # Create a new instance with the saved config
        store = cls(
            dimension=metadata["dimension"],
            index_type=metadata["index_type"],
            embedding_model=embedding_model,
        )

        # Load the FAISS index (overwrites the empty one created in __init__)
        store._index = faiss.read_index(str(index_path))
        store._documents = metadata["documents"]
        store._id_to_position = metadata["id_to_position"]

        logger.info(
            "faiss_index_loaded",
            directory=str(load_dir),
            index_size=store._index.ntotal,
            documents=len(store._documents),
        )

        return store

    def __repr__(self) -> str:
        return (
            f"FAISSVectorStore(dimension={self._dimension}, "
            f"index_type={self._index_type!r}, count={self.count})"
        )
