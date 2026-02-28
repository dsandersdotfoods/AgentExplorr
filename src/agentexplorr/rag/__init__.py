"""
Retrieval Augmented Generation (RAG) Module
=============================================

WHAT IS RAG?
  Retrieval Augmented Generation is a technique that enhances Large Language
  Models by giving them access to external knowledge at inference time. Instead
  of relying solely on the LLM's training data (which is frozen at training
  time and may be outdated or incomplete), RAG *retrieves* relevant documents
  from a knowledge base and *augments* the LLM's prompt with that context.

  In plain English:
    1. User asks a question
    2. We search a knowledge base for relevant documents (retrieval)
    3. We paste those documents into the LLM's prompt (augmentation)
    4. The LLM generates an answer grounded in the retrieved context (generation)

WHY RAG INSTEAD OF FINE-TUNING?
  - No retraining needed: Update your knowledge base, not your model weights
  - Source attribution: You can cite exactly which documents informed the answer
  - Reduced hallucination: The model answers from provided context, not memory
  - Cost-effective: Works with smaller, cheaper models (even local ones via Ollama)
  - Fresh data: Your knowledge base can be updated in real-time

THE RAG PIPELINE (implemented in this module):
  +-----------+    +-----------+    +------------+    +--------------+
  | Load Docs | -> | Chunk     | -> | Embed      | -> | Store in     |
  | (PDF,MD,  |    | (Fixed,   |    | (sentence- |    | Vector DB    |
  |  TXT)     |    |  Recursive|    |  transform)|    | (Chroma/FAISS|
  +-----------+    |  Semantic)|    +------------+    +--------------+
                   +-----------+            |                |
                                            v                v
                   +-----------+    +--------------+   +----------+
                   | Generate  | <- | Augment      | <- | Retrieve |
                   | (Ollama)  |    | Prompt       |    | Top-K    |
                   +-----------+    +--------------+    +----------+

MODULE OVERVIEW:
  - document_processor.py  : Load documents from various file formats
  - chunking.py            : Split documents into smaller chunks
  - embeddings.py          : Convert text to vector embeddings
  - vector_stores/         : Store and search embeddings (ChromaDB, FAISS)
  - retriever.py           : Hybrid retrieval across multiple vector stores
  - rag_pipeline.py        : End-to-end RAG pipeline orchestration
  - evaluation.py          : Evaluate RAG quality (relevancy, faithfulness)

ALL OPEN SOURCE — NO COMMERCIAL APIs REQUIRED:
  - Embeddings : sentence-transformers (Hugging Face)
  - Vector DB  : ChromaDB (persistent) + FAISS (in-memory, Facebook Research)
  - LLM        : Ollama (local inference server)

LEARNING RESOURCES:
  - Original RAG Paper (Lewis et al., 2020):
    https://arxiv.org/abs/2005.11401
  - Sentence-Transformers docs:
    https://www.sbert.net/
  - ChromaDB docs:
    https://docs.trychroma.com/
  - FAISS wiki:
    https://github.com/facebookresearch/faiss/wiki
  - Ollama docs:
    https://ollama.com/
  - LangChain RAG Tutorial:
    https://python.langchain.com/docs/tutorials/rag/

  VIDEO RESOURCES:
  - "RAG From Scratch" (LangChain, full series):
    https://www.youtube.com/watch?v=wd7TZ4w1mSw
  - "Building RAG from Scratch" (freeCodeCamp):
    https://www.youtube.com/watch?v=BrsocJb-fAo
  - "Vector Databases Explained" (Fireship):
    https://www.youtube.com/watch?v=klTvEwg3oJ4
  - "Sentence Transformers Explained" (James Briggs):
    https://www.youtube.com/watch?v=OATCgQtNX2o
  - "What are Vector Embeddings?" (IBM Technology):
    https://www.youtube.com/watch?v=dN0lsF2cvm4

PAPERS WORTH READING:
  - "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
    (Lewis et al., 2020) — https://arxiv.org/abs/2005.11401
  - "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"
    (Reimers & Gurevych, 2019) — https://arxiv.org/abs/1908.10084
  - "Billion-scale similarity search with GPUs" (FAISS)
    (Johnson, Douze, Jegou, 2019) — https://arxiv.org/abs/1702.08734
  - "RAGAS: Automated Evaluation of Retrieval Augmented Generation"
    (Es et al., 2023) — https://arxiv.org/abs/2309.15217
"""

from agentexplorr.rag.chunking import (
    Chunk,
    FixedSizeChunker,
    RecursiveChunker,
    SemanticChunker,
)
from agentexplorr.rag.document_processor import Document, DocumentProcessor
from agentexplorr.rag.embeddings import EmbeddingModel
from agentexplorr.rag.evaluation import RAGEvaluator
from agentexplorr.rag.rag_pipeline import RAGPipeline
from agentexplorr.rag.retriever import HybridRetriever
from agentexplorr.rag.vector_stores import ChromaVectorStore, FAISSVectorStore

__all__ = [
    # Document processing
    "Document",
    "DocumentProcessor",
    # Chunking strategies
    "Chunk",
    "FixedSizeChunker",
    "RecursiveChunker",
    "SemanticChunker",
    # Embeddings
    "EmbeddingModel",
    # Vector stores
    "ChromaVectorStore",
    "FAISSVectorStore",
    # Retrieval
    "HybridRetriever",
    # End-to-end pipeline
    "RAGPipeline",
    # Evaluation
    "RAGEvaluator",
]
