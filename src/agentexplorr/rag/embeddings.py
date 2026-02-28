"""
Embedding Model — Converting Text to Vector Representations
=============================================================

WHAT ARE EMBEDDINGS?
  An embedding is a dense vector (a list of floating-point numbers) that
  represents the *meaning* of a piece of text in a high-dimensional space.
  Texts with similar meanings end up close together in this space, enabling
  semantic search — finding documents by *meaning*, not just keyword matching.

  Example (384-dimensional all-MiniLM-L6-v2):
    "The cat sat on the mat"  →  [0.032, -0.118, 0.045, ..., 0.091]
    "A kitten rested on a rug" → [0.029, -0.112, 0.051, ..., 0.088]
    These two vectors will have HIGH cosine similarity (~0.9) because the
    sentences mean roughly the same thing.

    "Stock prices rose sharply" → [-0.064, 0.231, -0.012, ..., 0.155]
    This vector will be FAR from the cat sentences (cosine similarity ~0.1)
    because the topic is completely different.

HOW SENTENCE-TRANSFORMERS WORK:
  1. Tokenize the input text into subword tokens (WordPiece/BPE).
  2. Pass tokens through a pre-trained Transformer encoder (BERT variant).
  3. Pool the token-level outputs into a single sentence-level vector
     (typically mean pooling over all token embeddings).
  4. The model was fine-tuned with a contrastive loss (similar sentences
     should have similar embeddings, dissimilar ones should be far apart).

WHY all-MiniLM-L6-v2?
  - Tiny: 80MB, 6 layers (vs. 440MB for all-mpnet-base-v2).
  - Fast: ~14,000 sentences/sec on GPU, ~400/sec on CPU.
  - Quality: Strong performance on semantic textual similarity benchmarks.
  - 384 dimensions: good balance between expressiveness and storage cost.
  - It's the most popular model on Hugging Face for a reason.

LEARNING RESOURCES:
  - Sentence-Transformers docs:
    https://www.sbert.net/
  - Model card (all-MiniLM-L6-v2):
    https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
  - Pre-trained models list:
    https://www.sbert.net/docs/sentence_transformer/pretrained_models.html
  - VIDEO: "Sentence Transformers Explained" (James Briggs):
    https://www.youtube.com/watch?v=OATCgQtNX2o
  - VIDEO: "What are Vector Embeddings?" (IBM Technology):
    https://www.youtube.com/watch?v=dN0lsF2cvm4
  - VIDEO: "Word Embeddings — EXPLAINED!" (CodeEmporium):
    https://www.youtube.com/watch?v=5PL0TmQhA4Y

PAPERS:
  - "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"
    (Reimers & Gurevych, 2019) — https://arxiv.org/abs/1908.10084
  - "Efficient Natural Language Response Suggestion for Smart Reply"
    (Henderson et al., 2017) — https://arxiv.org/abs/1705.00652
  - "Attention Is All You Need" (Vaswani et al., 2017)
    — the Transformer architecture: https://arxiv.org/abs/1706.03762
"""

from __future__ import annotations

from typing import Any

from agentexplorr.core import Settings, get_logger

logger = get_logger(__name__)


class EmbeddingModel:
    """Wrapper around sentence-transformers for generating text embeddings.

    This class provides a clean, type-safe interface for embedding text into
    dense vectors.  It handles model loading, batching, normalization, and
    device selection (CPU/GPU) automatically.

    WHY WRAP sentence-transformers?
      1. **Consistent interface**: The rest of our RAG pipeline depends on
         ``embed_text()`` and ``embed_batch()``.  If we later swap the backend
         (e.g. to a ONNX-quantized model), only this class changes.
      2. **Lazy loading**: The model is loaded on first use, not at import time,
         so importing this module is instant.
      3. **Logging & error handling**: Centralized diagnostics for embedding
         failures, performance metrics, etc.

    Attributes:
        model_name:   The Hugging Face model identifier.
        dimension:    The dimensionality of the output vectors (set after load).

    Args:
        model_name: Name of the sentence-transformers model on Hugging Face.
                    Default: ``"all-MiniLM-L6-v2"`` (384-dim, fast, good quality).
        device:     Device to run inference on.  ``None`` means auto-detect
                    (GPU if available, else CPU).
        normalize:  Whether to L2-normalize embeddings (default ``True``).
                    Normalized vectors make cosine similarity equivalent to
                    dot product, which is cheaper to compute.

    Example::

        from agentexplorr.rag.embeddings import EmbeddingModel

        model = EmbeddingModel()

        # Single text
        vec = model.embed_text("What is retrieval augmented generation?")
        print(len(vec))  # 384

        # Batch of texts (much faster than calling embed_text in a loop!)
        vecs = model.embed_batch(["Hello world", "Goodbye world"])
        print(len(vecs))     # 2
        print(len(vecs[0]))  # 384
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        normalize: bool = True,
    ) -> None:
        # Use the model name from settings if not explicitly provided.
        # This lets you configure the model once in .env / Settings and
        # have it apply everywhere.
        if model_name is None:
            try:
                settings = Settings()
                model_name = settings.embedding_model
            except Exception:
                model_name = "all-MiniLM-L6-v2"

        self.model_name: str = model_name
        self.device: str | None = device
        self.normalize: bool = normalize
        self.dimension: int = 0  # Set after model loads

        # The actual SentenceTransformer instance — loaded lazily.
        self._model: Any = None

    def _load_model(self) -> Any:
        """Load the sentence-transformer model into memory.

        WHY LAZY LOADING?
          Model loading takes 1-5 seconds and consumes significant RAM.
          By deferring it to first use, we keep import-time fast and avoid
          loading models that may never be used (e.g. in tests).

        DEVICE SELECTION:
          sentence-transformers auto-detects CUDA GPUs.  If you want to force
          CPU (e.g. for reproducibility or memory constraints), pass
          ``device="cpu"`` to the constructor.
        """
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "EmbeddingModel requires the 'sentence-transformers' package. "
                "Install it with:  pip install sentence-transformers\n"
                "Docs: https://www.sbert.net/docs/installation.html"
            ) from exc

        logger.info(
            "loading_embedding_model",
            model_name=self.model_name,
            device=self.device or "auto",
        )

        self._model = SentenceTransformer(
            self.model_name,
            device=self.device,
        )

        # Extract the embedding dimension from the model config.
        # This is useful for downstream components (FAISS needs to know the
        # vector dimension upfront when creating an index).
        self.dimension = self._model.get_sentence_embedding_dimension()

        logger.info(
            "embedding_model_loaded",
            model_name=self.model_name,
            dimension=self.dimension,
            device=str(self._model.device),
        )

        return self._model

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string into a dense vector.

        This is a convenience wrapper around ``embed_batch`` for the common
        case of embedding one string.  If you have multiple texts, use
        ``embed_batch`` instead — it is significantly faster because it
        processes texts in parallel on the GPU.

        Args:
            text: The input text to embed.

        Returns:
            A list of floats representing the embedding vector.
            Length equals ``self.dimension`` (384 for all-MiniLM-L6-v2).

        Raises:
            ImportError: If sentence-transformers is not installed.
            RuntimeError: If the model fails to encode the text.

        Example::

            vec = model.embed_text("What is machine learning?")
            assert len(vec) == 384
        """
        if not text.strip():
            logger.warning("embedding_empty_text", text_preview=text[:50])
            # Return a zero vector rather than crashing — the caller can
            # decide whether to filter it out.
            model = self._load_model()
            return [0.0] * self.dimension

        result = self.embed_batch([text])
        return result[0]

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> list[list[float]]:
        """Embed a batch of texts into dense vectors.

        WHY BATCH?
          Transformer models are heavily parallelized.  Encoding 64 sentences
          at once is barely slower than encoding 1 — the GPU does them all
          simultaneously.  So always prefer batching over loops.

          Performance comparison (all-MiniLM-L6-v2 on CPU):
            - 1000 texts, one at a time:   ~25 seconds
            - 1000 texts, batch_size=64:   ~3 seconds   (8x faster!)

        Args:
            texts:          List of text strings to embed.
            batch_size:     Number of texts to encode in one forward pass.
                            Larger = faster but more memory.  64 is a safe default.
            show_progress:  Whether to show a tqdm progress bar (useful for
                            large batches, but noisy in production).

        Returns:
            A list of embedding vectors (each is a list[float]).
            Length of the outer list equals ``len(texts)``.

        Raises:
            ImportError: If sentence-transformers is not installed.
            ValueError:  If texts is empty.

        Example::

            texts = ["First document", "Second document", "Third document"]
            vectors = model.embed_batch(texts)
            assert len(vectors) == 3
            assert all(len(v) == 384 for v in vectors)
        """
        if not texts:
            return []

        model = self._load_model()

        logger.debug(
            "embedding_batch",
            count=len(texts),
            batch_size=batch_size,
            model=self.model_name,
        )

        # sentence-transformers returns a numpy array of shape (n, dimension).
        # We convert to nested Python lists for portability (JSON-serializable,
        # no numpy dependency for downstream consumers).
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize,
        )

        # Convert numpy array → list of lists
        # Each row is one embedding vector
        result: list[list[float]] = [embedding.tolist() for embedding in embeddings]

        logger.debug(
            "embedding_batch_complete",
            count=len(result),
            dimension=len(result[0]) if result else 0,
        )

        return result

    @property
    def is_loaded(self) -> bool:
        """Whether the underlying model has been loaded into memory."""
        return self._model is not None

    def __repr__(self) -> str:
        status = "loaded" if self.is_loaded else "not loaded"
        return (
            f"EmbeddingModel(model={self.model_name!r}, "
            f"dim={self.dimension}, {status})"
        )
