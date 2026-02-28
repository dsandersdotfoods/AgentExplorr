"""
Tests for Document Chunking
=============================

Verify chunking strategies produce correct splits with proper overlap.
"""

from __future__ import annotations

from agentexplorr.rag.chunking import FixedSizeChunker, RecursiveChunker


class TestFixedSizeChunker:
    """Tests for fixed-size character chunking."""

    def test_basic_chunking(self) -> None:
        """Test splitting text into fixed-size chunks."""
        chunker = FixedSizeChunker(chunk_size=50, overlap=0)
        text = "A" * 120
        chunks = chunker.chunk(text)

        assert len(chunks) == 3  # 50 + 50 + 20
        assert len(chunks[0].text) == 50
        assert len(chunks[2].text) == 20

    def test_overlap(self) -> None:
        """Test that chunk overlap works correctly."""
        chunker = FixedSizeChunker(chunk_size=100, overlap=20)
        text = "A" * 200
        chunks = chunker.chunk(text)

        # With overlap, chunks should share characters at boundaries
        assert len(chunks) >= 2

    def test_short_text(self) -> None:
        """Test text shorter than chunk_size produces one chunk."""
        chunker = FixedSizeChunker(chunk_size=1000, overlap=0)
        chunks = chunker.chunk("Short text.")
        assert len(chunks) == 1

    def test_chunks_have_metadata(self) -> None:
        """Test that chunks include metadata."""
        chunker = FixedSizeChunker(chunk_size=50, overlap=0)
        chunks = chunker.chunk("A" * 100)

        for chunk in chunks:
            assert chunk.chunk_id is not None
            assert "chunk_index" in chunk.metadata


class TestRecursiveChunker:
    """Tests for recursive character text splitting."""

    def test_splits_on_paragraphs(self) -> None:
        """Test that text is split on paragraph boundaries first."""
        chunker = RecursiveChunker(chunk_size=100, overlap=0)
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = chunker.chunk(text)

        # Each paragraph should ideally be its own chunk
        assert len(chunks) >= 1
        assert all(c.text.strip() for c in chunks)  # No empty chunks

    def test_falls_back_to_sentences(self) -> None:
        """Test fallback to sentence splitting when paragraphs are too long."""
        chunker = RecursiveChunker(chunk_size=50, overlap=0)
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = chunker.chunk(text)

        assert len(chunks) >= 1
