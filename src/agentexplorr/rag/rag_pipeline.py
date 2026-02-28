"""
RAG Pipeline — End-to-End Retrieval Augmented Generation
==========================================================

WHAT IS THE RAG PIPELINE?
  The RAGPipeline orchestrates the entire RAG workflow from start to finish:

    1. **LOAD**:     Read documents from files (PDF, Markdown, Text, …).
    2. **CHUNK**:    Split documents into focused passages.
    3. **EMBED**:    Convert chunks into vector embeddings.
    4. **STORE**:    Save embeddings in a vector store (ChromaDB or FAISS).
    5. **RETRIEVE**: Given a question, find the most relevant chunks.
    6. **GENERATE**: Feed the retrieved context to an LLM to produce an answer.

  This class wires together all the individual components we've built:
    DocumentProcessor → Chunker → EmbeddingModel → VectorStore → Retriever → LLM

WHY A PIPELINE CLASS?
  Without a pipeline, every RAG user has to write 50 lines of boilerplate:
  load docs, create a chunker, set up embeddings, configure a store, build
  a retriever, format a prompt, call the LLM.  The RAGPipeline encapsulates
  all of this into a clean, configurable interface.

  It also enforces consistency: the same embedding model is used for indexing
  AND retrieval (a common bug is using different models, which produces
  garbage results because the vector spaces don't align).

HOW THE LLM GENERATION WORKS (Ollama via langchain-ollama):
  We use ``langchain-ollama``'s ``ChatOllama`` to call a local Ollama server.
  The prompt template follows the standard RAG pattern:

    "Given the following context, answer the question.
     If the answer is not in the context, say 'I don't know.'

     Context: {retrieved chunks}
     Question: {user query}
     Answer:"

  The "say I don't know" instruction is critical: it reduces hallucination
  by giving the model permission to abstain rather than fabricate answers.

LEARNING RESOURCES:
  - LangChain RAG Tutorial:
    https://python.langchain.com/docs/tutorials/rag/
  - Ollama docs:
    https://ollama.com/
  - langchain-ollama docs:
    https://python.langchain.com/docs/integrations/chat/ollama/
  - VIDEO: "RAG From Scratch" (LangChain, full series):
    https://www.youtube.com/watch?v=wd7TZ4w1mSw
  - VIDEO: "Building RAG from Scratch" (freeCodeCamp):
    https://www.youtube.com/watch?v=BrsocJb-fAo
  - VIDEO: "Ollama + Python — Build Local RAG" (pixegami):
    https://www.youtube.com/watch?v=d0o89z134CQ
  - VIDEO: "Advanced RAG Techniques" (Sam Witteveen):
    https://www.youtube.com/watch?v=TRjq7t2Ms5I

PAPERS:
  - "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
    (Lewis et al., 2020) — https://arxiv.org/abs/2005.11401
  - "Self-RAG: Learning to Retrieve, Generate, and Critique through
    Self-Reflection" (Asai et al., 2023) — https://arxiv.org/abs/2310.11511
  - "Active Retrieval Augmented Generation" (Jiang et al., 2023)
    — https://arxiv.org/abs/2305.06983
"""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentexplorr.core import Settings, get_logger
from agentexplorr.rag.chunking import BaseChunker, Chunk, RecursiveChunker
from agentexplorr.rag.document_processor import DocumentProcessor
from agentexplorr.rag.embeddings import EmbeddingModel
from agentexplorr.rag.retriever import HybridRetriever, RetrievalResult
from agentexplorr.rag.vector_stores.chroma_store import ChromaVectorStore

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------------------------

# The RAG prompt template — this is the heart of the "augmentation" step.
# We explicitly instruct the model to:
#   1. ONLY use the provided context (reduces hallucination).
#   2. Say "I don't know" when the answer isn't in the context.
#   3. Cite the source when possible.
RAG_PROMPT_TEMPLATE = """You are a helpful assistant that answers questions based on the provided context.

INSTRUCTIONS:
- Answer the question using ONLY the information in the context below.
- If the context does not contain enough information to answer, say "I don't have enough information to answer this question based on the provided context."
- Be concise and specific in your answer.
- If possible, mention which part of the context supports your answer.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

# A simpler template for when you want the model to be more creative
# (less strict about sticking to context)
CONVERSATIONAL_RAG_TEMPLATE = """You are a knowledgeable assistant. Use the following context to help answer the question. You may also use your general knowledge, but prioritize information from the context.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


# ---------------------------------------------------------------------------
# RAG Response
# ---------------------------------------------------------------------------


@dataclass
class RAGResponse:
    """The full response from a RAG query, including provenance.

    WHY RETURN MORE THAN JUST THE ANSWER?
      In production RAG, you need to know:
        - Which chunks were used (for debugging and transparency).
        - The retrieval scores (to detect low-confidence answers).
        - The full prompt sent to the LLM (for prompt engineering iteration).
        - Timing information (to optimize bottlenecks).

    Attributes:
        answer:         The LLM-generated answer text.
        retrieved_chunks: The chunks retrieved from the vector store.
        query:          The original user question.
        prompt:         The full prompt sent to the LLM.
        model:          The LLM model name used for generation.
        metadata:       Additional information (timing, token counts, etc.).
    """

    answer: str = ""
    retrieved_chunks: list[RetrievalResult] = field(default_factory=list)
    query: str = ""
    prompt: str = ""
    model: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.answer[:100].replace("\n", "\\n")
        return (
            f"RAGResponse(model={self.model!r}, "
            f'chunks={len(self.retrieved_chunks)}, answer="{preview}...")'
        )


# ---------------------------------------------------------------------------
# RAG Pipeline
# ---------------------------------------------------------------------------


class RAGPipeline:
    """End-to-end Retrieval Augmented Generation pipeline.

    This class orchestrates the complete RAG workflow:
      ingest documents → chunk → embed → store → retrieve → generate

    It is the main entry point for using RAG in AgentExplorr.

    Args:
        embedding_model:  An ``EmbeddingModel`` instance.  If ``None``,
                          creates one with the default model.
        chunker:          A chunking strategy.  If ``None``, uses
                          ``RecursiveChunker`` (the best default).
        vector_store:     A vector store instance (ChromaDB or FAISS).
                          If ``None``, creates an ephemeral ChromaVectorStore.
        ollama_model:     The Ollama model name for generation (e.g. "llama3.2").
                          If ``None``, reads from Settings.
        ollama_base_url:  The Ollama server URL.  If ``None``, reads from Settings.
        prompt_template:  The RAG prompt template string.  Must contain
                          ``{context}`` and ``{question}`` placeholders.
        top_k:            Default number of chunks to retrieve per query.

    Usage::

        from agentexplorr.rag.rag_pipeline import RAGPipeline

        # Create pipeline with defaults
        pipeline = RAGPipeline()

        # Ingest documents
        pipeline.ingest("./documents/")

        # Ask questions
        response = pipeline.query("What is attention in transformers?")
        print(response.answer)
        print(f"Based on {len(response.retrieved_chunks)} chunks")

        # Inspect sources
        for chunk in response.retrieved_chunks:
            print(f"  [{chunk.score:.3f}] {chunk.metadata.get('source', 'unknown')}")
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel | None = None,
        chunker: BaseChunker | None = None,
        vector_store: Any | None = None,
        ollama_model: str | None = None,
        ollama_base_url: str | None = None,
        prompt_template: str | None = None,
        top_k: int = 5,
    ) -> None:
        # Load settings for defaults
        try:
            settings = Settings()
        except Exception:
            settings = None

        # Embedding model (shared across indexing and retrieval)
        self._embedding_model = embedding_model or EmbeddingModel()

        # Document processor (stateless, no config needed)
        self._doc_processor = DocumentProcessor()

        # Chunking strategy
        self._chunker = chunker or RecursiveChunker(
            chunk_size=1000,
            overlap=200,
        )

        # Vector store
        if vector_store is not None:
            self._vector_store = vector_store
        else:
            self._vector_store = ChromaVectorStore(
                collection_name="agentexplorr_rag",
                embedding_model=self._embedding_model,
            )

        # Retriever (wraps the vector store for unified search)
        self._retriever = HybridRetriever(
            stores={"primary": self._vector_store},
            strategy="score",
        )

        # LLM settings
        self._ollama_model = ollama_model or (
            settings.ollama_model if settings else "llama3.2"
        )
        self._ollama_base_url = ollama_base_url or (
            settings.ollama_base_url if settings else "http://localhost:11434"
        )

        # Prompt template
        self._prompt_template = prompt_template or RAG_PROMPT_TEMPLATE

        # Default retrieval depth
        self._top_k = top_k

        # Track ingested document counts for diagnostics
        self._total_docs_ingested: int = 0
        self._total_chunks_stored: int = 0

        logger.info(
            "rag_pipeline_initialized",
            embedding_model=self._embedding_model.model_name,
            chunker=type(self._chunker).__name__,
            vector_store=type(self._vector_store).__name__,
            ollama_model=self._ollama_model,
            top_k=self._top_k,
        )

    # ------------------------------------------------------------------
    # Ingestion (Load → Chunk → Embed → Store)
    # ------------------------------------------------------------------

    def ingest(
        self,
        source: str | Path | list[str | Path],
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Ingest documents from files or directories into the vector store.

        This runs the full indexing pipeline:
          1. Load documents from the source path(s).
          2. Chunk each document using the configured chunker.
          3. Embed and store the chunks in the vector store.

        Args:
            source:   A file path, directory path, or list of paths.
            metadata: Optional metadata to attach to all ingested documents.

        Returns:
            The number of chunks that were stored.
        """
        # Normalize to a list of paths
        if isinstance(source, (str, Path)):
            sources = [Path(source)]
        else:
            sources = [Path(s) for s in source]

        logger.info(
            "ingestion_starting",
            sources=[str(s) for s in sources],
        )

        all_chunks: list[Chunk] = []

        for src in sources:
            # Step 1: Load documents
            if src.is_dir():
                documents = self._doc_processor.load_directory(src)
            elif src.is_file():
                documents = self._doc_processor.load_file(src)
            else:
                logger.warning("skipping_invalid_source", source=str(src))
                continue

            self._total_docs_ingested += len(documents)

            # Step 2: Chunk each document
            for doc in documents:
                doc_metadata = {**(metadata or {}), **doc.metadata}
                chunks = self._chunker.chunk(doc.text, metadata=doc_metadata)
                all_chunks.extend(chunks)

        if not all_chunks:
            logger.warning("no_chunks_produced_from_ingestion")
            return 0

        # Step 3: Embed and store
        # The vector store handles embedding internally via our shared EmbeddingModel
        stored_ids = self._vector_store.add_documents(all_chunks)
        self._total_chunks_stored += len(stored_ids)

        logger.info(
            "ingestion_complete",
            documents_loaded=self._total_docs_ingested,
            chunks_stored=len(stored_ids),
            total_chunks=self._total_chunks_stored,
        )

        return len(stored_ids)

    def ingest_texts(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> int:
        """Ingest raw text strings directly (without loading from files).

        Useful for adding content from APIs, databases, or user input.

        Args:
            texts:     List of text strings to ingest.
            metadatas: Optional list of metadata dicts (one per text).

        Returns:
            The number of chunks stored.
        """
        if not texts:
            return 0

        metadatas = metadatas or [{} for _ in texts]

        all_chunks: list[Chunk] = []
        for text, meta in zip(texts, metadatas):
            chunks = self._chunker.chunk(text, metadata=meta)
            all_chunks.extend(chunks)

        if not all_chunks:
            return 0

        stored_ids = self._vector_store.add_documents(all_chunks)
        self._total_chunks_stored += len(stored_ids)
        return len(stored_ids)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve the most relevant chunks for a query WITHOUT generating an answer.

        Useful for:
          - Debugging retrieval quality independently from generation.
          - Building custom prompts from retrieved chunks.
          - Feeding retrieved context to a different LLM.

        Args:
            query: The search query.
            top_k: Number of results (defaults to pipeline's configured top_k).

        Returns:
            List of RetrievalResult objects sorted by relevance.
        """
        k = top_k or self._top_k
        return self._retriever.search(query, top_k=k)

    # ------------------------------------------------------------------
    # Generation (Retrieve → Augment Prompt → LLM)
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        top_k: int | None = None,
        prompt_template: str | None = None,
    ) -> RAGResponse:
        """Ask a question and get an answer grounded in the knowledge base.

        This is the main RAG method.  It:
          1. Retrieves relevant chunks from the vector store.
          2. Formats them into a prompt with the question.
          3. Sends the prompt to Ollama for generation.
          4. Returns the answer with full provenance.

        Args:
            question:         The user's question.
            top_k:            Number of chunks to retrieve.
            prompt_template:  Override the default prompt template for this query.

        Returns:
            A RAGResponse with the answer, retrieved chunks, and metadata.
        """
        import time

        start_time = time.perf_counter()
        k = top_k or self._top_k
        template = prompt_template or self._prompt_template

        logger.info(
            "rag_query_starting",
            question_preview=question[:100],
            top_k=k,
        )

        # Step 1: Retrieve relevant chunks
        retrieval_start = time.perf_counter()
        retrieved = self._retriever.search(question, top_k=k)
        retrieval_time = time.perf_counter() - retrieval_start

        if not retrieved:
            logger.warning("no_chunks_retrieved", question=question[:100])
            return RAGResponse(
                answer="I couldn't find any relevant information in the knowledge base.",
                retrieved_chunks=[],
                query=question,
                model=self._ollama_model,
                metadata={"retrieval_time": retrieval_time},
            )

        # Step 2: Format the context from retrieved chunks
        # We include the source metadata so the LLM can cite its sources.
        context_parts: list[str] = []
        for i, chunk in enumerate(retrieved, start=1):
            source = chunk.metadata.get("source", "unknown")
            page = chunk.metadata.get("page", "")
            source_info = f" (page {page})" if page else ""
            context_parts.append(
                f"[Source {i}: {source}{source_info}]\n{chunk.text}"
            )

        context = "\n\n---\n\n".join(context_parts)

        # Step 3: Format the prompt
        prompt = template.format(context=context, question=question)

        # Step 4: Generate the answer using Ollama
        generation_start = time.perf_counter()
        answer = self._generate(prompt)
        generation_time = time.perf_counter() - generation_start

        total_time = time.perf_counter() - start_time

        logger.info(
            "rag_query_complete",
            question_preview=question[:80],
            chunks_retrieved=len(retrieved),
            answer_length=len(answer),
            retrieval_time_s=round(retrieval_time, 3),
            generation_time_s=round(generation_time, 3),
            total_time_s=round(total_time, 3),
        )

        return RAGResponse(
            answer=answer,
            retrieved_chunks=retrieved,
            query=question,
            prompt=prompt,
            model=self._ollama_model,
            metadata={
                "retrieval_time": retrieval_time,
                "generation_time": generation_time,
                "total_time": total_time,
                "top_k": k,
                "chunks_retrieved": len(retrieved),
            },
        )

    def _generate(self, prompt: str) -> str:
        """Generate text using Ollama via langchain-ollama's ChatOllama.

        WHY ChatOllama (langchain-ollama)?
          - It handles the HTTP connection to the Ollama server.
          - It provides a clean message-based API (system, human, ai messages).
          - It supports streaming, but we use synchronous invoke for simplicity.
          - It's the officially recommended way to use Ollama with LangChain.

        INSTALLATION:
          pip install langchain-ollama
          Also requires Ollama running locally: https://ollama.com/

        FALLBACK:
          If langchain-ollama is not installed or Ollama is not running,
          we fall back to a helpful error message explaining how to set up.

        Args:
            prompt: The complete prompt to send to the LLM.

        Returns:
            The generated text response.
        """
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            logger.error(
                "langchain_ollama_not_installed",
                hint="pip install langchain-ollama",
            )
            return (
                "[ERROR] langchain-ollama is not installed. "
                "Install with: pip install langchain-ollama\n"
                "Also ensure Ollama is running: https://ollama.com/"
            )

        try:
            llm = ChatOllama(
                model=self._ollama_model,
                base_url=self._ollama_base_url,
                temperature=0.1,  # Low temperature for factual RAG answers
            )

            # Invoke the model with the prompt.
            # ChatOllama expects a list of messages or a single string.
            response = llm.invoke(prompt)

            # Extract the text content from the response
            if hasattr(response, "content"):
                return str(response.content)
            return str(response)

        except Exception as exc:
            error_msg = str(exc)
            logger.error(
                "ollama_generation_failed",
                model=self._ollama_model,
                base_url=self._ollama_base_url,
                error=error_msg,
            )

            # Provide a helpful error message
            if "Connection" in error_msg or "refused" in error_msg:
                return (
                    f"[ERROR] Could not connect to Ollama at {self._ollama_base_url}. "
                    f"Please ensure Ollama is running:\n"
                    f"  1. Install Ollama: https://ollama.com/\n"
                    f"  2. Start the server: `ollama serve`\n"
                    f"  3. Pull the model: `ollama pull {self._ollama_model}`"
                )
            elif "not found" in error_msg.lower() or "404" in error_msg:
                return (
                    f"[ERROR] Model '{self._ollama_model}' not found. "
                    f"Pull it with: `ollama pull {self._ollama_model}`"
                )
            else:
                return f"[ERROR] Ollama generation failed: {error_msg}"

    # ------------------------------------------------------------------
    # Streaming generation
    # ------------------------------------------------------------------

    def query_stream(
        self,
        question: str,
        top_k: int | None = None,
        prompt_template: str | None = None,
    ) -> Generator[str, None, RAGResponse]:
        """Ask a question and stream the answer token-by-token.

        Works identically to ``query()`` but yields answer tokens as they
        arrive from the LLM, enabling real-time display in chat interfaces.

        The final ``return`` value is a full ``RAGResponse`` (accessible via
        ``StopIteration.value`` or by wrapping in a helper).

        Usage::

            gen = pipeline.query_stream("What is attention?")
            try:
                while True:
                    token = next(gen)
                    print(token, end="", flush=True)
            except StopIteration as e:
                response = e.value  # Full RAGResponse

        Args:
            question:         The user's question.
            top_k:            Number of chunks to retrieve.
            prompt_template:  Override the default prompt template.

        Yields:
            Individual text tokens as strings.

        Returns:
            A complete RAGResponse (via generator return).
        """
        import time

        start_time = time.perf_counter()
        k = top_k or self._top_k
        template = prompt_template or self._prompt_template

        # Step 1: Retrieve
        retrieval_start = time.perf_counter()
        retrieved = self._retriever.search(question, top_k=k)
        retrieval_time = time.perf_counter() - retrieval_start

        if not retrieved:
            no_info = "I couldn't find any relevant information in the knowledge base."
            yield no_info
            return RAGResponse(
                answer=no_info,
                retrieved_chunks=[],
                query=question,
                model=self._ollama_model,
                metadata={"retrieval_time": retrieval_time},
            )

        # Step 2: Build context and prompt
        context_parts: list[str] = []
        for i, chunk in enumerate(retrieved, start=1):
            source = chunk.metadata.get("source", "unknown")
            page = chunk.metadata.get("page", "")
            source_info = f" (page {page})" if page else ""
            context_parts.append(
                f"[Source {i}: {source}{source_info}]\n{chunk.text}"
            )
        context = "\n\n---\n\n".join(context_parts)
        prompt = template.format(context=context, question=question)

        # Step 3: Stream from Ollama
        generation_start = time.perf_counter()
        collected_tokens: list[str] = []

        try:
            from langchain_ollama import ChatOllama

            llm = ChatOllama(
                model=self._ollama_model,
                base_url=self._ollama_base_url,
                temperature=0.1,
            )

            for chunk_msg in llm.stream(prompt):
                token = chunk_msg.content if hasattr(chunk_msg, "content") else str(chunk_msg)
                collected_tokens.append(token)
                yield token

        except ImportError:
            error = "[ERROR] langchain-ollama is not installed."
            collected_tokens.append(error)
            yield error
        except Exception as exc:
            error = f"[ERROR] Streaming failed: {exc}"
            collected_tokens.append(error)
            yield error

        generation_time = time.perf_counter() - generation_start
        total_time = time.perf_counter() - start_time
        answer = "".join(collected_tokens)

        return RAGResponse(
            answer=answer,
            retrieved_chunks=retrieved,
            query=question,
            prompt=prompt,
            model=self._ollama_model,
            metadata={
                "retrieval_time": retrieval_time,
                "generation_time": generation_time,
                "total_time": total_time,
                "top_k": k,
                "chunks_retrieved": len(retrieved),
                "streamed": True,
            },
        )

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def add_vector_store(self, name: str, store: Any) -> None:
        """Add an additional vector store to the retriever.

        This enables hybrid retrieval across multiple stores.

        Args:
            name:  A name for the store (e.g. "faiss_technical_docs").
            store: A vector store instance with a ``search`` method.
        """
        self._retriever.add_store(name, store)

    @property
    def stats(self) -> dict[str, Any]:
        """Return pipeline statistics for monitoring and debugging."""
        return {
            "total_docs_ingested": self._total_docs_ingested,
            "total_chunks_stored": self._total_chunks_stored,
            "embedding_model": self._embedding_model.model_name,
            "chunker": type(self._chunker).__name__,
            "vector_store": type(self._vector_store).__name__,
            "ollama_model": self._ollama_model,
            "top_k": self._top_k,
        }

    def __repr__(self) -> str:
        return (
            f"RAGPipeline(model={self._ollama_model!r}, "
            f"chunks={self._total_chunks_stored}, "
            f"store={type(self._vector_store).__name__})"
        )
