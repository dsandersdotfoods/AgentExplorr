"""
Vector Stores — Persistent Storage for Embeddings
===================================================

WHAT IS A VECTOR STORE?
  A vector store (or vector database) is a specialized database optimized for
  storing and searching high-dimensional vectors.  In a RAG system, we embed
  our document chunks into vectors and store them in a vector store.  At query
  time, we embed the user's question and find the most similar document vectors.

  This is called **approximate nearest neighbor (ANN) search** — we find the
  K vectors closest to the query vector using distance metrics like cosine
  similarity or L2 (Euclidean) distance.

WHY TWO VECTOR STORES?
  We provide two implementations to teach different approaches:

  **ChromaDB** (chroma_store.py):
    - Full-featured vector database with persistence, metadata filtering, and
      a built-in collection abstraction.
    - Great for prototyping and small-to-medium datasets.
    - Stores data on disk — survives restarts.
    - Docs: https://docs.trychroma.com/

  **FAISS** (faiss_store.py):
    - Facebook AI Similarity Search — a battle-tested C++ library with Python
      bindings, optimized for speed on large datasets.
    - In-memory by default (we add save/load support).
    - Best for when you need raw speed and control over the index type.
    - Docs: https://github.com/facebookresearch/faiss/wiki

CHOOSING BETWEEN THEM:
  ┌─────────────────┬──────────────┬────────────────┐
  │ Feature         │ ChromaDB     │ FAISS          │
  ├─────────────────┼──────────────┼────────────────┤
  │ Persistence     │ Built-in     │ Manual save/   │
  │                 │              │ load           │
  │ Metadata filter │ Yes          │ No (manual)    │
  │ Speed (large)   │ Good         │ Excellent      │
  │ GPU support     │ No           │ Yes (faiss-gpu)│
  │ Setup           │ pip install  │ pip install    │
  │                 │ chromadb     │ faiss-cpu      │
  │ Best for        │ Prototyping, │ Production,    │
  │                 │ small data   │ large datasets │
  └─────────────────┴──────────────┴────────────────┘

LEARNING RESOURCES:
  - VIDEO: "Vector Databases Explained" (Fireship):
    https://www.youtube.com/watch?v=klTvEwg3oJ4
  - VIDEO: "FAISS — Introduction to Similarity Search" (James Briggs):
    https://www.youtube.com/watch?v=sKyvsdEv6rk
  - ChromaDB getting started:
    https://docs.trychroma.com/docs/overview/getting-started
  - FAISS wiki:
    https://github.com/facebookresearch/faiss/wiki/Getting-started
  - "Billion-scale similarity search with GPUs" (FAISS paper):
    https://arxiv.org/abs/1702.08734
"""

from agentexplorr.rag.vector_stores.chroma_store import ChromaVectorStore
from agentexplorr.rag.vector_stores.faiss_store import FAISSVectorStore

__all__ = [
    "ChromaVectorStore",
    "FAISSVectorStore",
]
