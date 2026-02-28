"""
Document Processor — Loading Documents from Multiple File Formats
=================================================================

WHY DO WE NEED A DOCUMENT PROCESSOR?
  RAG systems need to ingest knowledge from existing documents. Those documents
  come in many formats: PDF reports, Markdown documentation, plain text logs,
  HTML pages, etc.  The DocumentProcessor abstracts away format-specific parsing
  so the rest of the pipeline receives a uniform ``Document`` object regardless
  of the source file type.

HOW IT WORKS:
  1. You point the processor at a file or directory.
  2. It detects the file type from the suffix (.pdf, .md, .txt, …).
  3. A format-specific loader extracts the raw text content.
  4. It wraps the text + metadata (source path, page numbers, timestamps) into
     a ``Document`` dataclass that the chunker can consume downstream.

WHEN TO USE:
  - At the *very first stage* of a RAG pipeline, before chunking.
  - When building a knowledge base from heterogeneous file formats.
  - When you need to track provenance (which file/page an answer came from).

SUPPORTED FORMATS:
  - .txt   — Plain text (read as-is)
  - .md    — Markdown  (read as-is; formatting preserved for the LLM)
  - .pdf   — PDF       (extracted via PyPDF2 / pypdf)
  - .html  — HTML      (stripped to text via BeautifulSoup)
  - .csv   — CSV       (each row becomes a document)

LEARNING RESOURCES:
  - pypdf docs:
    https://pypdf.readthedocs.io/en/stable/
  - BeautifulSoup docs:
    https://www.crummy.com/software/BeautifulSoup/bs4/doc/
  - pathlib guide:
    https://docs.python.org/3/library/pathlib.html
  - VIDEO: "Python pathlib — The Modern Way to Handle File Paths":
    https://www.youtube.com/watch?v=UcKkmwaRbsQ
  - VIDEO: "Extract Text from PDF in Python" (NeuralNine):
    https://www.youtube.com/watch?v=w2r2Bg42UPY
  - VIDEO: "RAG From Scratch — Part 1: Indexing" (LangChain):
    https://www.youtube.com/watch?v=wd7TZ4w1mSw

DESIGN DECISIONS:
  - We use ``pathlib.Path`` everywhere instead of raw strings because it is
    cross-platform, typo-resistant, and composes naturally with ``/``.
  - Each loader is a private method so you can subclass and override just the
    format you need without touching the rest.
  - Metadata is a plain ``dict[str, Any]`` so downstream components can add to
    it freely (chunk IDs, embedding vectors, scores, …).
"""

from __future__ import annotations

import csv
import datetime
import io
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentexplorr.core import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Data Model
# ---------------------------------------------------------------------------


@dataclass
class Document:
    """A single logical document extracted from a source file.

    Attributes:
        doc_id:   Unique identifier for this document (auto-generated UUID4).
        text:     The raw textual content of the document.
        metadata: Arbitrary key-value pairs tracking provenance and context.
                  Commonly includes: ``source`` (file path), ``page`` (for PDFs),
                  ``format`` (file extension), ``created_at`` (ISO timestamp).

    WHY A DATACLASS?
      Dataclasses give us a lightweight, immutable-ish container with
      ``__repr__``, ``__eq__``, and ``__hash__`` for free.  They are the
      Pythonic way to model "plain data" objects without the boilerplate of
      writing ``__init__`` yourself.

      Learn more: https://docs.python.org/3/library/dataclasses.html
      VIDEO: https://www.youtube.com/watch?v=vBH6GRJ1REM
    """

    doc_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        """Return the character length of the document text."""
        return len(self.text)

    def __bool__(self) -> bool:
        """A Document is truthy if it contains non-whitespace text."""
        return bool(self.text.strip())


# ---------------------------------------------------------------------------
# Supported file extensions → human-readable names (for logging/errors)
# ---------------------------------------------------------------------------

_SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".txt": "Plain Text",
    ".md": "Markdown",
    ".pdf": "PDF",
    ".html": "HTML",
    ".htm": "HTML",
    ".csv": "CSV",
}


# ---------------------------------------------------------------------------
# Document Processor
# ---------------------------------------------------------------------------


class DocumentProcessor:
    """Load documents from files or directories into uniform ``Document`` objects.

    The processor auto-detects file format from the suffix and delegates to
    the appropriate loader.  It is intentionally *stateless*: every call to
    ``load_file`` or ``load_directory`` returns a fresh list of Documents.

    Usage::

        from agentexplorr.rag.document_processor import DocumentProcessor

        processor = DocumentProcessor()

        # Single file
        docs = processor.load_file("papers/attention.pdf")

        # Entire directory (recursively)
        docs = processor.load_directory("knowledge_base/")

        # Filter by extension
        docs = processor.load_directory("docs/", extensions=[".md", ".txt"])

    Args:
        default_metadata: Optional dict of metadata added to *every* document.
                          Useful for tagging a batch (e.g. ``{"project": "v2"}``).
    """

    def __init__(self, default_metadata: dict[str, Any] | None = None) -> None:
        self._default_metadata: dict[str, Any] = default_metadata or {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_file(self, path: str | Path) -> list[Document]:
        """Load a single file and return a list of Documents.

        For most formats this returns a single-element list.  PDFs may return
        one Document per page (each with ``metadata["page"]`` set).

        Args:
            path: Path to the source file.

        Returns:
            List of Document objects extracted from the file.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError:        If the file extension is not supported.
        """
        file_path = Path(path).resolve()

        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        suffix = file_path.suffix.lower()

        if suffix not in _SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file format '{suffix}'. "
                f"Supported: {', '.join(_SUPPORTED_EXTENSIONS.keys())}"
            )

        logger.info(
            "loading_document",
            path=str(file_path),
            format=_SUPPORTED_EXTENSIONS[suffix],
        )

        # Dispatch to the right loader based on file extension
        loader_map: dict[str, Any] = {
            ".txt": self._load_text,
            ".md": self._load_text,       # Markdown is plain text
            ".pdf": self._load_pdf,
            ".html": self._load_html,
            ".htm": self._load_html,
            ".csv": self._load_csv,
        }

        loader = loader_map[suffix]
        documents: list[Document] = loader(file_path)

        logger.info(
            "documents_loaded",
            path=str(file_path),
            count=len(documents),
            total_chars=sum(len(d) for d in documents),
        )

        return documents

    def load_directory(
        self,
        directory: str | Path,
        extensions: list[str] | None = None,
        recursive: bool = True,
    ) -> list[Document]:
        """Load all supported documents from a directory.

        Args:
            directory:  Path to the directory to scan.
            extensions: Optional whitelist of extensions (e.g. ``[".md", ".pdf"]``).
                        If ``None``, all supported extensions are loaded.
            recursive:  Whether to recurse into subdirectories (default ``True``).

        Returns:
            Flat list of all Document objects found.

        Raises:
            FileNotFoundError: If the directory does not exist.
        """
        dir_path = Path(directory).resolve()

        if not dir_path.is_dir():
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        # Determine which extensions to look for
        target_extensions = set(extensions or _SUPPORTED_EXTENSIONS.keys())

        # Collect file paths — ``rglob`` for recursive, ``glob`` for flat
        # rglob("*") yields every file in the tree; we then filter by suffix.
        glob_method = dir_path.rglob if recursive else dir_path.glob
        file_paths = sorted(
            p for p in glob_method("*")
            if p.is_file() and p.suffix.lower() in target_extensions
        )

        logger.info(
            "scanning_directory",
            directory=str(dir_path),
            files_found=len(file_paths),
            recursive=recursive,
        )

        all_documents: list[Document] = []
        for fp in file_paths:
            try:
                docs = self.load_file(fp)
                all_documents.extend(docs)
            except Exception as exc:
                # Log and skip problematic files rather than crashing the whole batch
                logger.warning(
                    "skipping_file",
                    path=str(fp),
                    error=str(exc),
                )

        logger.info(
            "directory_loaded",
            directory=str(dir_path),
            total_documents=len(all_documents),
        )

        return all_documents

    # ------------------------------------------------------------------
    # Private loaders
    # ------------------------------------------------------------------

    def _build_metadata(self, path: Path, **extra: Any) -> dict[str, Any]:
        """Build a metadata dict combining defaults, file info, and extras.

        Every Document gets at minimum:
          - ``source``:      absolute path of the source file
          - ``format``:      human-readable format name
          - ``file_size``:   size in bytes
          - ``created_at``:  ISO-8601 timestamp of when the Document was created
        """
        metadata: dict[str, Any] = {
            **self._default_metadata,
            "source": str(path),
            "format": _SUPPORTED_EXTENSIONS.get(path.suffix.lower(), "Unknown"),
            "file_size": path.stat().st_size,
            "created_at": datetime.datetime.now(tz=datetime.UTC).isoformat(),
            **extra,
        }
        return metadata

    def _load_text(self, path: Path) -> list[Document]:
        """Load a plain-text or Markdown file.

        WHY KEEP MARKDOWN AS-IS?
          Markdown formatting (headers, lists, bold) actually helps the LLM
          understand document structure.  Stripping it would lose valuable
          context about what is a heading vs. body text.
        """
        text = path.read_text(encoding="utf-8")
        return [
            Document(
                text=text,
                metadata=self._build_metadata(path),
            )
        ]

    def _load_pdf(self, path: Path) -> list[Document]:
        """Load a PDF file, returning one Document per page.

        WHY ONE DOCUMENT PER PAGE?
          Page-level granularity lets the chunker make smarter decisions.
          It also preserves page numbers in metadata so retrieved chunks can
          cite "page 42 of report.pdf" — essential for source attribution.

        DEPENDENCY: pypdf  (pip install pypdf)
          We use ``pypdf`` (the maintained fork of PyPDF2).
          Docs: https://pypdf.readthedocs.io/en/stable/
        """
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ImportError(
                "PDF support requires the 'pypdf' package. "
                "Install it with:  pip install pypdf"
            ) from exc

        reader = PdfReader(str(path))
        documents: list[Document] = []

        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            # Skip effectively empty pages (headers/footers only)
            if not text.strip():
                logger.debug("skipping_empty_page", path=str(path), page=page_num)
                continue

            documents.append(
                Document(
                    text=text,
                    metadata=self._build_metadata(
                        path,
                        page=page_num,
                        total_pages=len(reader.pages),
                    ),
                )
            )

        return documents

    def _load_html(self, path: Path) -> list[Document]:
        """Load an HTML file, stripping tags to extract readable text.

        DEPENDENCY: beautifulsoup4  (pip install beautifulsoup4)
          We use ``BeautifulSoup`` with the built-in ``html.parser`` so there
          is no C dependency.
          Docs: https://www.crummy.com/software/BeautifulSoup/bs4/doc/
        """
        try:
            from bs4 import BeautifulSoup
        except ImportError as exc:
            raise ImportError(
                "HTML support requires the 'beautifulsoup4' package. "
                "Install it with:  pip install beautifulsoup4"
            ) from exc

        raw_html = path.read_text(encoding="utf-8")
        soup = BeautifulSoup(raw_html, "html.parser")

        # Remove script and style elements that would pollute the text
        for element in soup(["script", "style", "nav", "footer"]):
            element.decompose()

        # get_text with separator="\n" preserves paragraph breaks
        text = soup.get_text(separator="\n", strip=True)

        # Extract the <title> if present for metadata
        title = soup.title.string.strip() if soup.title and soup.title.string else None

        return [
            Document(
                text=text,
                metadata=self._build_metadata(
                    path,
                    title=title,
                ),
            )
        ]

    def _load_csv(self, path: Path) -> list[Document]:
        """Load a CSV file, converting each row into a separate Document.

        WHY PER-ROW DOCUMENTS?
          CSV files are inherently tabular.  Each row is a self-contained
          record (a customer, a log entry, a product).  Making each row its
          own Document lets the retriever find the specific row that answers
          the user's question rather than returning the entire table.

        The row content is formatted as ``key: value`` pairs for readability.
        """
        raw_text = path.read_text(encoding="utf-8")
        reader = csv.DictReader(io.StringIO(raw_text))

        documents: list[Document] = []
        for row_num, row in enumerate(reader, start=1):
            # Convert the row dict to a human-readable string
            # e.g. "name: Alice\nage: 30\ncity: Paris"
            text = "\n".join(f"{key}: {value}" for key, value in row.items())
            documents.append(
                Document(
                    text=text,
                    metadata=self._build_metadata(
                        path,
                        row_number=row_num,
                        columns=list(row.keys()),
                    ),
                )
            )

        return documents
