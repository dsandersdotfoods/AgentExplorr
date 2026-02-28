"""
Text Chunking Strategies — Splitting Documents for Retrieval
=============================================================

WHY DO WE CHUNK DOCUMENTS?
  Embedding models have a finite context window (typically 256-512 tokens for
  sentence-transformers).  Even if the model *could* embed an entire book, the
  resulting single vector would be a blurry average of every topic in the book
  — useless for precise retrieval.

  Chunking breaks documents into small, focused pieces so that:
    1. Each chunk fits within the embedding model's context window.
    2. Each chunk covers a coherent topic (ideally one idea per chunk).
    3. The retriever can return the *specific* passage that answers the query,
       not an entire chapter.

HOW CHUNKING AFFECTS RAG QUALITY:
  Chunking is the #1 overlooked factor in RAG quality.  Bad chunks → bad
  retrieval → bad answers.  The ideal chunk:
    - Is self-contained (makes sense without surrounding context).
    - Is small enough to be focused, but large enough to be meaningful.
    - Preserves natural boundaries (paragraphs, sections, sentences).

THE THREE STRATEGIES IN THIS MODULE:

  1. **FixedSizeChunker**  — Split every N characters with M characters of
     overlap.  Dead simple, fast, predictable.  Works well for uniform text
     like transcripts or logs.

  2. **RecursiveChunker**  — Try splitting on ``\\n\\n`` (paragraphs) first.
     If chunks are still too big, split on ``\\n`` (lines), then ``. ``
     (sentences), then `` `` (words).  This preserves natural structure and
     is the most widely used strategy in production RAG systems.

  3. **SemanticChunker**   — Group consecutive sentences that are
     semantically similar (by embedding cosine similarity).  The most
     sophisticated approach: it finds natural topic boundaries even within
     a single paragraph.  Requires an embedding model.

LEARNING RESOURCES:
  - "Chunking Strategies for LLM Applications" (Pinecone):
    https://www.pinecone.io/learn/chunking-strategies/
  - "Text Splitters" (LangChain docs):
    https://python.langchain.com/docs/concepts/text_splitters/
  - "5 Levels of Text Splitting" (Greg Kamradt):
    https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb
  - VIDEO: "Chunking for RAG: Best Practices" (James Briggs):
    https://www.youtube.com/watch?v=eGE5AraIkNM
  - VIDEO: "5 Levels of Text Splitting" (Greg Kamradt):
    https://www.youtube.com/watch?v=8OJC21T2SL4
  - VIDEO: "RAG From Scratch — Part 2: Indexing" (LangChain):
    https://www.youtube.com/watch?v=bjb_EMsTDKI

PAPERS:
  - "Dense Passage Retrieval for Open-Domain QA" (Karpukhin et al., 2020)
    — discusses optimal passage length: https://arxiv.org/abs/2004.04906
  - "Sentence-BERT" (Reimers & Gurevych, 2019)
    — the embedding model used for semantic chunking: https://arxiv.org/abs/1908.10084
"""

from __future__ import annotations

import hashlib
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from agentexplorr.core import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data Model
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    """A single chunk of text produced by a chunking strategy.

    Attributes:
        text:      The chunk content.
        metadata:  Provenance information inherited from the parent Document,
                   plus chunk-specific fields (``chunk_index``, ``chunker``,
                   ``char_start``, ``char_end``).
        chunk_id:  Unique identifier.  Defaults to a deterministic SHA-256
                   hash of the text so that identical chunks get the same ID
                   (useful for deduplication).

    WHY DETERMINISTIC IDs?
      If you re-process the same document, you want to detect "I already have
      this chunk" without storing the full text.  A content-based hash makes
      deduplication trivial:  ``if chunk_id in existing_ids: skip``.
    """

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_id: str = ""

    def __post_init__(self) -> None:
        """Generate a deterministic chunk_id from the text content if not set."""
        if not self.chunk_id:
            # SHA-256 truncated to 16 hex chars — low collision, short string
            self.chunk_id = hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:16]

    def __len__(self) -> int:
        return len(self.text)

    def __repr__(self) -> str:
        preview = self.text[:80].replace("\n", "\\n")
        return f'Chunk(id={self.chunk_id!r}, len={len(self.text)}, text="{preview}...")'


# ---------------------------------------------------------------------------
# Abstract Base Class
# ---------------------------------------------------------------------------


class BaseChunker(ABC):
    """Abstract base class for all chunking strategies.

    WHY AN ABC?
      An Abstract Base Class enforces a contract: every chunker MUST implement
      ``chunk(text, metadata)``.  This lets downstream code treat all chunkers
      uniformly (polymorphism) and prevents accidentally using an incomplete
      implementation.

      Learn more: https://docs.python.org/3/library/abc.html
      VIDEO: "Abstract Classes in Python" — https://www.youtube.com/watch?v=UDmJGvM-OUw
    """

    @abstractmethod
    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        """Split text into a list of Chunks.

        Args:
            text:     The source text to split.
            metadata: Optional metadata to attach to every chunk (e.g. source file).

        Returns:
            A list of Chunk objects.
        """
        ...  # pragma: no cover


# ---------------------------------------------------------------------------
# Strategy 1: Fixed-Size Chunking
# ---------------------------------------------------------------------------


class FixedSizeChunker(BaseChunker):
    """Split text into fixed-size chunks with configurable overlap.

    HOW IT WORKS:
      Imagine a sliding window of ``chunk_size`` characters that advances by
      ``chunk_size - overlap`` characters each step:

        Text:    |---- chunk 1 ----|
                            |---- chunk 2 ----|
                                       |---- chunk 3 ----|

      The **overlap** ensures that sentences straddling a chunk boundary
      appear in BOTH adjacent chunks, so the retriever can find them
      regardless of which chunk is returned.

    WHEN TO USE:
      - When you need dead-simple, deterministic chunking.
      - For uniform text where natural boundaries are rare (e.g. raw logs).
      - As a baseline to compare against smarter strategies.

    WHEN NOT TO USE:
      - When the text has clear structure (sections, paragraphs) — prefer
        RecursiveChunker.
      - When you care about semantic coherence — prefer SemanticChunker.

    Args:
        chunk_size: Target size of each chunk in characters (default 1000).
        overlap:    Number of overlapping characters between consecutive chunks
                    (default 200).  Set to 0 for no overlap.

    Example::

        chunker = FixedSizeChunker(chunk_size=500, overlap=50)
        chunks = chunker.chunk("Your long document text here...")
    """

    def __init__(self, chunk_size: int = 1000, overlap: int = 200) -> None:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if overlap < 0:
            raise ValueError(f"overlap must be non-negative, got {overlap}")
        if overlap >= chunk_size:
            raise ValueError(
                f"overlap ({overlap}) must be less than chunk_size ({chunk_size}). "
                "Otherwise the window never advances!"
            )

        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        """Split text into fixed-size chunks.

        Args:
            text:     Source text.
            metadata: Optional metadata dict copied into each chunk.

        Returns:
            List of Chunk objects with ``chunk_index`` in metadata.
        """
        if not text.strip():
            return []

        base_metadata = metadata or {}
        chunks: list[Chunk] = []

        # Step size = how far the window moves each iteration.
        # With overlap, this is less than chunk_size.
        step = self.chunk_size - self.overlap
        start = 0
        index = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]

            # Skip chunks that are only whitespace
            if chunk_text.strip():
                chunk_metadata = {
                    **base_metadata,
                    "chunk_index": index,
                    "char_start": start,
                    "char_end": end,
                    "chunker": "FixedSizeChunker",
                    "chunk_size": self.chunk_size,
                    "overlap": self.overlap,
                }
                chunks.append(Chunk(text=chunk_text, metadata=chunk_metadata))
                index += 1

            start += step

        logger.info(
            "fixed_size_chunking_complete",
            total_chunks=len(chunks),
            chunk_size=self.chunk_size,
            overlap=self.overlap,
            source_length=len(text),
        )

        return chunks


# ---------------------------------------------------------------------------
# Strategy 2: Recursive Character Text Splitting
# ---------------------------------------------------------------------------


class RecursiveChunker(BaseChunker):
    """Split text recursively using a hierarchy of separators.

    HOW IT WORKS:
      This is the strategy used by LangChain's ``RecursiveCharacterTextSplitter``
      and is considered the default "smart" chunker in production RAG systems.

      The idea is simple but effective:
        1. Try to split the text on the highest-priority separator (``\\n\\n``
           — paragraph breaks).
        2. If any resulting piece is still larger than ``chunk_size``, split
           THAT piece on the next separator (``\\n`` — line breaks).
        3. Continue down the hierarchy:  ``. `` → `` `` → character-level.
        4. After splitting, merge consecutive small pieces back together
           (with overlap) to avoid tiny chunks.

      This preserves document structure: paragraphs stay intact when possible,
      sentences stay intact when paragraphs are too long, etc.

    SEPARATOR HIERARCHY (default):
      ``["\\n\\n", "\\n", ". ", " ", ""]``

      - ``\\n\\n`` → Paragraph boundaries (strongest semantic boundary)
      - ``\\n``   → Line breaks (weaker boundary, but still structural)
      - ``. ``   → Sentence endings (preserve sentence integrity)
      - `` ``    → Word boundaries (last resort before character-level)
      - ``""``   → Character-level split (emergency fallback)

    WHEN TO USE:
      - For structured text with paragraphs, headings, lists (Markdown, docs).
      - As your default chunking strategy — it works well in most cases.
      - When you want a good balance between chunk quality and simplicity.

    Args:
        chunk_size:  Maximum chunk size in characters (default 1000).
        overlap:     Number of overlapping characters (default 200).
        separators:  Ordered list of separators to try.  Defaults to the
                     hierarchy described above.

    Example::

        chunker = RecursiveChunker(chunk_size=800, overlap=100)
        chunks = chunker.chunk(markdown_text, metadata={"source": "README.md"})
    """

    # The default separator hierarchy, from strongest to weakest boundary
    DEFAULT_SEPARATORS: list[str] = ["\n\n", "\n", ". ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 1000,
        overlap: int = 200,
        separators: list[str] | None = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if overlap < 0:
            raise ValueError(f"overlap must be non-negative, got {overlap}")
        if overlap >= chunk_size:
            raise ValueError(
                f"overlap ({overlap}) must be less than chunk_size ({chunk_size})"
            )

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = separators or self.DEFAULT_SEPARATORS

    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        """Split text using recursive separator hierarchy.

        Args:
            text:     Source text.
            metadata: Optional metadata dict copied into each chunk.

        Returns:
            List of Chunk objects.
        """
        if not text.strip():
            return []

        base_metadata = metadata or {}

        # Step 1: Recursively split the text
        raw_chunks = self._recursive_split(text, self.separators)

        # Step 2: Merge small consecutive pieces back together (with overlap)
        merged = self._merge_chunks(raw_chunks)

        # Step 3: Wrap in Chunk dataclass with metadata
        chunks: list[Chunk] = []
        for index, chunk_text in enumerate(merged):
            if chunk_text.strip():
                chunk_metadata = {
                    **base_metadata,
                    "chunk_index": index,
                    "chunker": "RecursiveChunker",
                    "chunk_size": self.chunk_size,
                    "overlap": self.overlap,
                }
                chunks.append(Chunk(text=chunk_text, metadata=chunk_metadata))

        logger.info(
            "recursive_chunking_complete",
            total_chunks=len(chunks),
            chunk_size=self.chunk_size,
            source_length=len(text),
        )

        return chunks

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text, trying separators from strongest to weakest.

        This is the core algorithm.  For each separator:
          1. Split the text on that separator.
          2. Collect pieces that are already small enough.
          3. For pieces that are still too large, recurse with the NEXT separator.

        The recursion bottoms out at the empty string separator ``""``, which
        splits into individual characters (guaranteed to be < chunk_size).

        Args:
            text:       Text to split.
            separators: Remaining separators to try (shrinks on each recursion).

        Returns:
            List of text pieces, each <= chunk_size.
        """
        # Base case: text is already small enough
        if len(text) <= self.chunk_size:
            return [text]

        # Base case: no more separators to try (should not happen with "" in list)
        if not separators:
            return [text]

        separator = separators[0]
        remaining_separators = separators[1:]

        # Split on the current separator
        if separator == "":
            # Character-level split (emergency fallback)
            # Just chop the text into chunk_size pieces
            pieces = [
                text[i : i + self.chunk_size]
                for i in range(0, len(text), self.chunk_size)
            ]
            return pieces

        parts = text.split(separator)

        result: list[str] = []
        for part in parts:
            if len(part) <= self.chunk_size:
                result.append(part)
            else:
                # This part is still too large — recurse with weaker separators
                sub_parts = self._recursive_split(part, remaining_separators)
                result.extend(sub_parts)

        return result

    def _merge_chunks(self, pieces: list[str]) -> list[str]:
        """Merge small consecutive pieces into chunks respecting size limits.

        After the recursive split, we may have many tiny pieces (individual
        sentences or lines).  This method greedily merges consecutive pieces
        until adding the next piece would exceed ``chunk_size``.

        Overlap is implemented by including the tail of the previous merged
        chunk at the start of the next one.

        Args:
            pieces: List of text pieces from the recursive split.

        Returns:
            List of merged text chunks.
        """
        if not pieces:
            return []

        merged: list[str] = []
        current = pieces[0]

        for piece in pieces[1:]:
            # Would adding this piece (with a space) exceed the chunk_size?
            combined = current + " " + piece
            if len(combined) <= self.chunk_size:
                current = combined
            else:
                # Emit the current chunk
                merged.append(current)

                # Start the new chunk with overlap from the end of the previous
                if self.overlap > 0 and len(current) > self.overlap:
                    # Take the last `overlap` characters of the emitted chunk
                    overlap_text = current[-self.overlap :]
                    current = overlap_text + " " + piece
                else:
                    current = piece

        # Don't forget the last chunk
        if current.strip():
            merged.append(current)

        return merged


# ---------------------------------------------------------------------------
# Strategy 3: Semantic Chunking
# ---------------------------------------------------------------------------


class SemanticChunker(BaseChunker):
    """Group consecutive sentences by semantic similarity.

    HOW IT WORKS:
      This is the most sophisticated chunking strategy.  Instead of splitting
      on character counts or syntactic markers, it uses embedding similarity
      to find natural topic boundaries:

        1. Split the text into sentences.
        2. Embed each sentence using a sentence-transformer model.
        3. Compute the cosine similarity between each pair of adjacent sentences.
        4. Where the similarity drops below a threshold, insert a chunk boundary.
        5. Group the sentences between boundaries into chunks.

      The intuition: within a coherent paragraph about "vector databases",
      consecutive sentences will have high similarity.  When the text transitions
      to "evaluation metrics", the similarity drops — that's our split point.

    WHEN TO USE:
      - When chunk quality is critical (e.g. customer-facing RAG).
      - When the text lacks clear structural markers (no headings, few paragraphs).
      - When you want each chunk to cover exactly one topic/idea.

    TRADE-OFFS:
      - Slower than FixedSize/Recursive (requires embedding every sentence).
      - Requires a sentence-transformer model loaded in memory.
      - Threshold tuning: too low → huge chunks; too high → tiny chunks.

    Args:
        similarity_threshold: Cosine similarity below which we insert a boundary
                              (default 0.5).  Range is 0.0 to 1.0.
        embedding_model_name: The sentence-transformer model to use for
                              computing sentence embeddings.  Default is
                              ``"all-MiniLM-L6-v2"`` (fast, 384-dim).
        min_chunk_size:       Minimum chunk size in characters.  Chunks smaller
                              than this are merged with their neighbors.
        max_chunk_size:       Maximum chunk size in characters.  Chunks larger
                              than this are force-split.

    LEARNING RESOURCES:
      - "Semantic Chunking" (LlamaIndex docs):
        https://docs.llamaindex.ai/en/stable/examples/node_parsers/semantic_chunking/
      - VIDEO: "Semantic Chunking for RAG" (James Briggs):
        https://www.youtube.com/watch?v=8OJC21T2SL4
      - Sentence-BERT paper: https://arxiv.org/abs/1908.10084
      - Cosine Similarity explained:
        https://en.wikipedia.org/wiki/Cosine_similarity

    Example::

        chunker = SemanticChunker(similarity_threshold=0.4)
        chunks = chunker.chunk(long_document_text)
    """

    def __init__(
        self,
        similarity_threshold: float = 0.5,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        min_chunk_size: int = 100,
        max_chunk_size: int = 2000,
    ) -> None:
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError(
                f"similarity_threshold must be between 0.0 and 1.0, "
                f"got {similarity_threshold}"
            )

        self.similarity_threshold = similarity_threshold
        self.embedding_model_name = embedding_model_name
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

        # The embedding model is loaded lazily on first use to avoid
        # paying the import/load cost if the chunker is never called.
        self._model: Any = None

    def _get_model(self) -> Any:
        """Lazily load the sentence-transformer model.

        WHY LAZY LOADING?
          Loading a transformer model takes 1-3 seconds and ~100MB of RAM.
          If you create a SemanticChunker but never call ``.chunk()``, you
          shouldn't pay that cost.  Lazy loading defers it to first use.
        """
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "SemanticChunker requires 'sentence-transformers'. "
                    "Install with:  pip install sentence-transformers"
                ) from exc

            logger.info(
                "loading_embedding_model",
                model=self.embedding_model_name,
                purpose="semantic_chunking",
            )
            self._model = SentenceTransformer(self.embedding_model_name)
        return self._model

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences using simple heuristics.

        This is a pragmatic sentence splitter — not a full NLP sentence
        tokenizer.  It handles the most common cases:
          - Period + space + uppercase letter (standard English sentence)
          - Question marks and exclamation marks
          - Newlines as sentence boundaries

        For production use, consider ``nltk.sent_tokenize()`` or ``spaCy``.
        """
        import re

        # Split on sentence-ending punctuation followed by whitespace
        # The lookbehind ensures we keep the punctuation with the sentence.
        sentences = re.split(r"(?<=[.!?])\s+", text)

        # Also split on double newlines (paragraph breaks)
        expanded: list[str] = []
        for sent in sentences:
            parts = sent.split("\n\n")
            expanded.extend(p.strip() for p in parts if p.strip())

        return expanded

    @staticmethod
    def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
        """Compute cosine similarity between two vectors.

        WHAT IS COSINE SIMILARITY?
          It measures the angle between two vectors in high-dimensional space.
          - 1.0 means identical direction (same meaning).
          - 0.0 means orthogonal (unrelated).
          - -1.0 means opposite (rare in practice for text embeddings).

          Formula:  cos(θ) = (A · B) / (||A|| × ||B||)

          We implement it manually with basic math to avoid a numpy dependency
          in this function, but numpy would be faster for large batches.
        """
        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        magnitude_a = sum(a * a for a in vec_a) ** 0.5
        magnitude_b = sum(b * b for b in vec_b) ** 0.5

        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0

        return dot_product / (magnitude_a * magnitude_b)

    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        """Split text into semantically coherent chunks.

        Algorithm:
          1. Split into sentences.
          2. Embed all sentences.
          3. Compute pairwise cosine similarity between adjacent sentences.
          4. Insert boundaries where similarity < threshold.
          5. Merge small groups, split large groups.

        Args:
            text:     Source text.
            metadata: Optional metadata dict copied into each chunk.

        Returns:
            List of Chunk objects grouped by semantic similarity.
        """
        if not text.strip():
            return []

        base_metadata = metadata or {}
        sentences = self._split_sentences(text)

        if not sentences:
            return []

        # If only one sentence, return it as a single chunk
        if len(sentences) == 1:
            return [
                Chunk(
                    text=sentences[0],
                    metadata={
                        **base_metadata,
                        "chunk_index": 0,
                        "chunker": "SemanticChunker",
                    },
                )
            ]

        # Step 2: Embed all sentences in a batch for efficiency
        model = self._get_model()
        embeddings = model.encode(sentences, show_progress_bar=False)
        # embeddings is a numpy array of shape (n_sentences, embedding_dim)
        # Convert rows to lists for our cosine_similarity function
        embedding_lists: list[list[float]] = [emb.tolist() for emb in embeddings]

        # Step 3: Compute similarity between each adjacent pair
        similarities: list[float] = []
        for i in range(len(embedding_lists) - 1):
            sim = self._cosine_similarity(embedding_lists[i], embedding_lists[i + 1])
            similarities.append(sim)

        # Step 4: Find boundary indices where similarity drops below threshold
        # boundary_indices[k] means: split BETWEEN sentence[k] and sentence[k+1]
        boundary_indices: list[int] = [
            i
            for i, sim in enumerate(similarities)
            if sim < self.similarity_threshold
        ]

        # Step 5: Group sentences into chunks based on boundaries
        groups: list[list[str]] = []
        prev_boundary = 0
        for boundary in boundary_indices:
            # Sentences from prev_boundary to boundary (inclusive)
            group = sentences[prev_boundary : boundary + 1]
            groups.append(group)
            prev_boundary = boundary + 1

        # Don't forget the last group
        if prev_boundary < len(sentences):
            groups.append(sentences[prev_boundary:])

        # Step 6: Build Chunk objects, respecting min/max size constraints
        chunks: list[Chunk] = []
        index = 0
        carry_over: list[str] = []  # sentences too small to form their own chunk

        for group in groups:
            combined_sentences = carry_over + group
            chunk_text = " ".join(combined_sentences)

            if len(chunk_text) < self.min_chunk_size:
                # Too small — carry these sentences forward to merge with the next group
                carry_over = combined_sentences
                continue

            carry_over = []

            if len(chunk_text) > self.max_chunk_size:
                # Too large — force-split into max_chunk_size pieces
                for i in range(0, len(chunk_text), self.max_chunk_size):
                    sub_text = chunk_text[i : i + self.max_chunk_size]
                    if sub_text.strip():
                        chunk_metadata = {
                            **base_metadata,
                            "chunk_index": index,
                            "chunker": "SemanticChunker",
                            "threshold": self.similarity_threshold,
                        }
                        chunks.append(Chunk(text=sub_text, metadata=chunk_metadata))
                        index += 1
            else:
                chunk_metadata = {
                    **base_metadata,
                    "chunk_index": index,
                    "chunker": "SemanticChunker",
                    "threshold": self.similarity_threshold,
                }
                chunks.append(Chunk(text=chunk_text, metadata=chunk_metadata))
                index += 1

        # Handle any remaining carry_over sentences
        if carry_over:
            leftover_text = " ".join(carry_over)
            if leftover_text.strip():
                # Merge with the last chunk if possible, otherwise create a new one
                if chunks and len(chunks[-1].text) + len(leftover_text) <= self.max_chunk_size:
                    chunks[-1] = Chunk(
                        text=chunks[-1].text + " " + leftover_text,
                        metadata=chunks[-1].metadata,
                    )
                else:
                    chunk_metadata = {
                        **base_metadata,
                        "chunk_index": index,
                        "chunker": "SemanticChunker",
                        "threshold": self.similarity_threshold,
                    }
                    chunks.append(Chunk(text=leftover_text, metadata=chunk_metadata))

        logger.info(
            "semantic_chunking_complete",
            total_chunks=len(chunks),
            num_sentences=len(sentences),
            num_boundaries=len(boundary_indices),
            threshold=self.similarity_threshold,
        )

        return chunks
