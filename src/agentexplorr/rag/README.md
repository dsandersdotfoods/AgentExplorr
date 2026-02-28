# RAG Module — Retrieval Augmented Generation

## What You'll Learn

This module teaches you how to build a **production-grade RAG system** from scratch, using only open-source tools. By the end, you'll understand:

- How to convert documents into searchable vector embeddings
- Why chunking strategy is the #1 factor in RAG quality
- How vector databases store and search high-dimensional vectors
- How to combine retrieval with LLM generation for grounded answers
- How to evaluate whether your RAG system is working well

---

## Table of Contents

1. [The Big Picture](#the-big-picture)
2. [Architecture Overview](#architecture-overview)
3. [Quick Start](#quick-start)
4. [Component Deep Dives](#component-deep-dives)
   - [Document Processing](#1-document-processing)
   - [Chunking Strategies](#2-chunking-strategies)
   - [Embeddings](#3-embeddings)
   - [Vector Stores](#4-vector-stores)
   - [Retrieval](#5-retrieval)
   - [Generation](#6-generation)
   - [Evaluation](#7-evaluation)
5. [Advanced Topics](#advanced-topics)
6. [Recommended Learning Path](#recommended-learning-path)
7. [Paper References](#paper-references)
8. [Video Resources](#video-resources)
9. [Troubleshooting](#troubleshooting)

---

## The Big Picture

### Why RAG Exists

Large Language Models (LLMs) have a fundamental limitation: their knowledge is frozen at training time. If you ask GPT about events after its training cutoff, or about your company's internal documents, it will either hallucinate or say "I don't know."

**RAG solves this** by giving the LLM access to external knowledge at inference time:

```
User Question: "What were our Q3 revenue numbers?"
         |
         v
    [RETRIEVE] Search your document database for relevant passages
         |
         v
    [AUGMENT] Insert those passages into the LLM's prompt
         |
         v
    [GENERATE] LLM answers based on the provided context
         |
         v
    Answer: "According to the Q3 report, revenue was $12.4M..."
```

### RAG vs. Fine-Tuning

| Aspect | RAG | Fine-Tuning |
|--------|-----|-------------|
| Knowledge updates | Instant (update docs) | Hours (retrain model) |
| Source attribution | Yes (cite passages) | No |
| Cost | Low (no training) | High (GPU hours) |
| Hallucination | Reduced (grounded) | Still possible |
| Best for | Factual Q&A, search | Style, behavior |

**Rule of thumb**: Use RAG for *knowledge*, fine-tuning for *behavior*.

---

## Architecture Overview

```
                    AgentExplorr RAG Pipeline
 ┌──────────────────────────────────────────────────────────┐
 │                                                          │
 │  ┌─────────────┐    ┌──────────┐    ┌────────────────┐  │
 │  │  Document    │───>│ Chunking │───>│  Embedding     │  │
 │  │  Processor   │    │ Strategy │    │  Model         │  │
 │  │              │    │          │    │  (sentence-    │  │
 │  │  PDF/MD/TXT  │    │ Fixed    │    │   transformers)│  │
 │  │  HTML/CSV    │    │ Recursive│    │                │  │
 │  │              │    │ Semantic │    │  all-MiniLM    │  │
 │  └─────────────┘    └──────────┘    └───────┬────────┘  │
 │                                              │           │
 │                                              v           │
 │  ┌─────────────┐    ┌──────────┐    ┌───────────────┐   │
 │  │   Ollama     │<───│  Prompt  │<───│ Vector Store  │   │
 │  │   (LLM)      │    │ Template │    │               │   │
 │  │              │    │          │    │ ChromaDB      │   │
 │  │  llama3.2    │    │ Context  │    │ FAISS         │   │
 │  │  mistral     │    │ +Question│    │               │   │
 │  └──────┬───────┘    └──────────┘    └───────────────┘   │
 │         │                                                │
 │         v                                                │
 │  ┌─────────────┐    ┌──────────────────┐                 │
 │  │   Answer     │    │   Evaluation     │                 │
 │  │   + Sources  │───>│   Metrics        │                 │
 │  │              │    │                  │                 │
 │  └─────────────┘    │ Context Relevancy │                 │
 │                      │ Faithfulness     │                 │
 │                      │ Answer Relevancy │                 │
 │                      └──────────────────┘                 │
 └──────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites

```bash
# Install the required packages
pip install sentence-transformers chromadb faiss-cpu langchain-ollama pypdf beautifulsoup4

# Install and start Ollama (for LLM generation)
# See: https://ollama.com/
ollama serve              # Start the server
ollama pull llama3.2      # Pull a model
```

### Minimal Example

```python
from agentexplorr.rag import RAGPipeline

# Create a pipeline with sensible defaults
pipeline = RAGPipeline()

# Ingest your documents
pipeline.ingest("./my_documents/")

# Ask questions
response = pipeline.query("What is the main topic of these documents?")
print(response.answer)

# See which sources were used
for chunk in response.retrieved_chunks:
    print(f"  [{chunk.score:.3f}] {chunk.metadata.get('source', 'unknown')}")
```

### Step-by-Step Example

```python
from agentexplorr.rag import (
    DocumentProcessor,
    RecursiveChunker,
    EmbeddingModel,
    ChromaVectorStore,
    HybridRetriever,
    RAGEvaluator,
)

# Step 1: Load documents
processor = DocumentProcessor()
docs = processor.load_directory("./knowledge_base/", extensions=[".md", ".txt"])
print(f"Loaded {len(docs)} documents")

# Step 2: Chunk documents
chunker = RecursiveChunker(chunk_size=800, overlap=100)
chunks = []
for doc in docs:
    chunks.extend(chunker.chunk(doc.text, metadata=doc.metadata))
print(f"Created {len(chunks)} chunks")

# Step 3: Set up embedding model
embedding_model = EmbeddingModel(model_name="all-MiniLM-L6-v2")

# Step 4: Store in vector database
store = ChromaVectorStore(
    collection_name="my_knowledge_base",
    persist_directory="./chroma_data",
    embedding_model=embedding_model,
)
store.add_documents(chunks)
print(f"Stored {store.count} chunks in ChromaDB")

# Step 5: Search
results = store.search("What is attention in transformers?", top_k=5)
for r in results:
    print(f"  [{r.score:.3f}] {r.text[:100]}...")

# Step 6: Evaluate
evaluator = RAGEvaluator(embedding_model=embedding_model)
eval_result = evaluator.evaluate(
    query="What is attention?",
    contexts=[r.text for r in results],
    answer="Attention allows the model to focus on relevant parts of the input.",
)
print(f"Context Relevancy: {eval_result.context_relevancy:.3f}")
print(f"Faithfulness:      {eval_result.answer_faithfulness:.3f}")
print(f"Answer Relevancy:  {eval_result.answer_relevancy:.3f}")
print(f"Overall Score:     {eval_result.overall_score:.3f}")
```

---

## Component Deep Dives

### 1. Document Processing

**File**: `document_processor.py`

The `DocumentProcessor` loads documents from various file formats into a uniform `Document` dataclass.

```python
from agentexplorr.rag.document_processor import DocumentProcessor, Document

processor = DocumentProcessor(default_metadata={"project": "agentexplorr"})

# Load a single PDF
docs = processor.load_file("paper.pdf")  # Returns one Document per page

# Load an entire directory
docs = processor.load_directory("./docs/", extensions=[".md", ".pdf"])
```

**Supported formats**: `.txt`, `.md`, `.pdf`, `.html`, `.csv`

**Key concept**: Each `Document` carries metadata (source file, page number, timestamp) that flows through the entire pipeline. When the final answer cites "page 42 of report.pdf", that provenance comes from the Document metadata.

**Learn more**:
- [pypdf docs](https://pypdf.readthedocs.io/en/stable/)
- [Python pathlib guide](https://docs.python.org/3/library/pathlib.html)

---

### 2. Chunking Strategies

**File**: `chunking.py`

Chunking is the **most important** step in a RAG pipeline. Bad chunks = bad retrieval = bad answers.

#### FixedSizeChunker

Split text every N characters with overlap:

```python
from agentexplorr.rag.chunking import FixedSizeChunker

chunker = FixedSizeChunker(chunk_size=500, overlap=50)
chunks = chunker.chunk("Your long document text...", metadata={"source": "doc.txt"})
```

**Best for**: Uniform text (logs, transcripts).

#### RecursiveChunker (Recommended Default)

Split on natural boundaries (paragraphs, then sentences, then words):

```python
from agentexplorr.rag.chunking import RecursiveChunker

chunker = RecursiveChunker(chunk_size=1000, overlap=200)
chunks = chunker.chunk(markdown_text)
```

**Best for**: Structured text with paragraphs and sections.

#### SemanticChunker

Group sentences by embedding similarity:

```python
from agentexplorr.rag.chunking import SemanticChunker

chunker = SemanticChunker(similarity_threshold=0.5)
chunks = chunker.chunk(long_document_text)
```

**Best for**: Text without clear structure where topic boundaries matter.

#### How to Choose Chunk Size

| Chunk Size | Pros | Cons |
|-----------|------|------|
| Small (200-500 chars) | Precise retrieval | May lose context |
| Medium (500-1000 chars) | Good balance | Default choice |
| Large (1000-2000 chars) | More context | Less precise |

**Golden rule**: Start with RecursiveChunker at 1000 chars / 200 overlap, then tune based on evaluation metrics.

**Learn more**:
- VIDEO: [5 Levels of Text Splitting (Greg Kamradt)](https://www.youtube.com/watch?v=8OJC21T2SL4)
- VIDEO: [Chunking for RAG: Best Practices (James Briggs)](https://www.youtube.com/watch?v=eGE5AraIkNM)
- [Pinecone Chunking Guide](https://www.pinecone.io/learn/chunking-strategies/)

---

### 3. Embeddings

**File**: `embeddings.py`

Embeddings convert text into dense vectors (lists of numbers) that capture semantic meaning.

```python
from agentexplorr.rag.embeddings import EmbeddingModel

model = EmbeddingModel(model_name="all-MiniLM-L6-v2")

# Single text
vec = model.embed_text("What is machine learning?")
print(f"Dimension: {len(vec)}")  # 384

# Batch (much faster!)
vecs = model.embed_batch(["Hello", "World"], batch_size=64)
```

#### Why all-MiniLM-L6-v2?

- 80MB model size (tiny!)
- 384-dimensional output
- Fast: ~14K sentences/sec on GPU
- Strong quality for its size
- Most popular model on Hugging Face

#### Other Models to Try

| Model | Dimensions | Quality | Speed |
|-------|-----------|---------|-------|
| all-MiniLM-L6-v2 | 384 | Good | Fast |
| all-mpnet-base-v2 | 768 | Better | Medium |
| e5-large-v2 | 1024 | Best | Slow |
| bge-small-en-v1.5 | 384 | Good | Fast |

**Learn more**:
- VIDEO: [What are Vector Embeddings? (IBM Technology)](https://www.youtube.com/watch?v=dN0lsF2cvm4)
- VIDEO: [Sentence Transformers Explained (James Briggs)](https://www.youtube.com/watch?v=OATCgQtNX2o)
- [Sentence-BERT Paper](https://arxiv.org/abs/1908.10084)

---

### 4. Vector Stores

**Files**: `vector_stores/chroma_store.py`, `vector_stores/faiss_store.py`

#### ChromaDB — The Easy Choice

```python
from agentexplorr.rag.vector_stores import ChromaVectorStore

store = ChromaVectorStore(
    collection_name="my_docs",
    persist_directory="./chroma_data",  # Data survives restarts
)
store.add_documents(chunks)
results = store.search("What is RAG?", top_k=5)

# Metadata filtering (unique to ChromaDB)
results = store.search(
    "What is RAG?",
    top_k=5,
    where={"source": "rag_paper.pdf"},
)
```

#### FAISS — The Fast Choice

```python
from agentexplorr.rag.vector_stores import FAISSVectorStore

store = FAISSVectorStore(dimension=384)
store.add_documents(chunks)
results = store.search("What is RAG?", top_k=5)

# Save and load
store.save("./faiss_index/")
store = FAISSVectorStore.load("./faiss_index/")
```

**Learn more**:
- VIDEO: [Vector Databases Explained (Fireship)](https://www.youtube.com/watch?v=klTvEwg3oJ4)
- VIDEO: [ChromaDB Tutorial (pixegami)](https://www.youtube.com/watch?v=QSW2L8dkaZk)
- VIDEO: [FAISS Introduction (James Briggs)](https://www.youtube.com/watch?v=sKyvsdEv6rk)

---

### 5. Retrieval

**File**: `retriever.py`

The `HybridRetriever` searches across multiple vector stores and re-ranks results:

```python
from agentexplorr.rag.retriever import HybridRetriever

retriever = HybridRetriever(
    stores={"chroma": chroma_store, "faiss": faiss_store},
    strategy="rrf",  # Reciprocal Rank Fusion
)

results = retriever.search("What is attention?", top_k=10)
for r in results:
    print(f"[{r.score:.4f}] [{r.source_store}] {r.text[:80]}")
```

#### Re-Ranking Strategies

**Score-based** (default): Sort by raw similarity score. Simple and fast.

**Reciprocal Rank Fusion (RRF)**: Uses rank position instead of raw scores. More robust when combining results from different stores with incomparable score scales.

```
RRF_score(chunk) = sum(1 / (k + rank_in_store_i)) for each store
```

**Learn more**:
- VIDEO: [Advanced RAG — Hybrid Search (James Briggs)](https://www.youtube.com/watch?v=lYxGYXjfrNI)
- [RRF Paper (Cormack et al., 2009)](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)

---

### 6. Generation

**File**: `rag_pipeline.py`

The `RAGPipeline` orchestrates the entire flow and uses Ollama for generation:

```python
from agentexplorr.rag.rag_pipeline import RAGPipeline

pipeline = RAGPipeline(
    ollama_model="llama3.2",      # Any Ollama model
    top_k=5,                       # Chunks to retrieve
)

# Ingest and query
pipeline.ingest("./documents/")
response = pipeline.query("Summarize the key findings.")

print(response.answer)
print(f"Model: {response.model}")
print(f"Time: {response.metadata['total_time']:.2f}s")
```

#### The Prompt Template

The default prompt explicitly instructs the LLM to:
1. Only use the provided context (reduces hallucination)
2. Say "I don't know" when appropriate (reduces confabulation)
3. Cite sources when possible (improves transparency)

You can customize it:

```python
custom_template = """You are a {role}. Answer based on the context.
Context: {context}
Question: {question}
Answer:"""

response = pipeline.query("What is X?", prompt_template=custom_template)
```

**Learn more**:
- VIDEO: [RAG From Scratch (LangChain)](https://www.youtube.com/watch?v=wd7TZ4w1mSw)
- VIDEO: [Ollama + Python RAG (pixegami)](https://www.youtube.com/watch?v=d0o89z134CQ)
- [Ollama docs](https://ollama.com/)

---

### 7. Evaluation

**File**: `evaluation.py`

The `RAGEvaluator` measures three dimensions of RAG quality:

```python
from agentexplorr.rag.evaluation import RAGEvaluator

evaluator = RAGEvaluator()

result = evaluator.evaluate(
    query="What is the attention mechanism?",
    contexts=["Attention allows models to focus on relevant input parts..."],
    answer="The attention mechanism enables models to weigh input importance...",
)

print(f"Context Relevancy:  {result.context_relevancy:.3f}")
print(f"Faithfulness:       {result.answer_faithfulness:.3f}")
print(f"Answer Relevancy:   {result.answer_relevancy:.3f}")
print(f"Overall Score:      {result.overall_score:.3f}")
```

#### Understanding the Metrics

| Metric | What It Measures | Low Score Means |
|--------|-----------------|-----------------|
| Context Relevancy | Are retrieved chunks relevant to the query? | Bad retrieval / wrong chunks |
| Answer Faithfulness | Is the answer grounded in the context? | Hallucination detected |
| Answer Relevancy | Does the answer address the question? | Off-topic response |

#### Batch Evaluation

```python
results = evaluator.evaluate_batch(
    queries=["Q1", "Q2", "Q3"],
    contexts_list=[["ctx1"], ["ctx2"], ["ctx3"]],
    answers=["A1", "A2", "A3"],
)

# Compute averages
avg_score = sum(r.overall_score for r in results) / len(results)
```

**Learn more**:
- VIDEO: [Evaluating RAG with RAGAS (James Briggs)](https://www.youtube.com/watch?v=25mMPVt-gRk)
- [RAGAS Paper](https://arxiv.org/abs/2309.15217)
- [RAGAS docs](https://docs.ragas.io/)

---

## Advanced Topics

### Improving Retrieval Quality

1. **Tune chunk size**: Start at 1000, try 500 and 1500. Measure with `context_relevancy`.
2. **Add overlap**: 10-20% overlap prevents splitting important context across chunks.
3. **Try semantic chunking**: Use `SemanticChunker` for unstructured text.
4. **Increase top_k**: Retrieve more chunks (10-20) to improve recall.
5. **Use hybrid retrieval**: Combine ChromaDB + FAISS via `HybridRetriever`.

### Reducing Hallucination

1. **Use strict prompts**: The default template says "only use the context."
2. **Lower temperature**: `temperature=0.1` in Ollama reduces creative generation.
3. **Monitor faithfulness**: Track `answer_faithfulness` metric over time.
4. **Add more context**: Sometimes hallucination means the right chunk wasn't retrieved.

### Scaling to Large Document Collections

1. **Use FAISS with IVF**: For >100K documents, switch to `IndexIVFFlat`.
2. **Use persistent ChromaDB**: Set `persist_directory` to keep data across restarts.
3. **Batch ingestion**: Process documents in batches to manage memory.
4. **Consider quantization**: FAISS `IndexIVFPQ` compresses vectors for billions of docs.

---

## Recommended Learning Path

### Beginner (Week 1-2)

1. Watch: [RAG From Scratch](https://www.youtube.com/watch?v=wd7TZ4w1mSw) (LangChain series)
2. Watch: [Vector Databases Explained](https://www.youtube.com/watch?v=klTvEwg3oJ4) (Fireship)
3. Read: [Pinecone Chunking Guide](https://www.pinecone.io/learn/chunking-strategies/)
4. Do: Run the Quick Start example above

### Intermediate (Week 3-4)

5. Watch: [What are Vector Embeddings?](https://www.youtube.com/watch?v=dN0lsF2cvm4) (IBM)
6. Watch: [Sentence Transformers Explained](https://www.youtube.com/watch?v=OATCgQtNX2o) (James Briggs)
7. Read: [Sentence-BERT Paper](https://arxiv.org/abs/1908.10084)
8. Do: Try all three chunking strategies and compare evaluation scores

### Advanced (Week 5-6)

9. Watch: [5 Levels of Text Splitting](https://www.youtube.com/watch?v=8OJC21T2SL4) (Greg Kamradt)
10. Watch: [FAISS Introduction](https://www.youtube.com/watch?v=sKyvsdEv6rk) (James Briggs)
11. Read: [Original RAG Paper](https://arxiv.org/abs/2005.11401) (Lewis et al., 2020)
12. Read: [RAGAS Paper](https://arxiv.org/abs/2309.15217) (Es et al., 2023)
13. Do: Build a hybrid retriever with custom re-ranking

---

## Paper References

| Paper | Year | Key Contribution |
|-------|------|-----------------|
| [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) | 2020 | The original RAG paper |
| [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084) | 2019 | The embedding model we use |
| [Billion-scale similarity search with GPUs (FAISS)](https://arxiv.org/abs/1702.08734) | 2019 | FAISS library paper |
| [Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) | 2020 | DPR — discusses optimal passage length |
| [RAGAS: Automated Evaluation of RAG](https://arxiv.org/abs/2309.15217) | 2023 | RAG evaluation framework |
| [Self-RAG: Learning to Retrieve, Generate, and Critique](https://arxiv.org/abs/2310.11511) | 2023 | Advanced RAG with self-reflection |
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | 2017 | The Transformer architecture |

---

## Video Resources

### Fundamentals
- [RAG From Scratch — Full Series (LangChain)](https://www.youtube.com/watch?v=wd7TZ4w1mSw)
- [Building RAG from Scratch (freeCodeCamp)](https://www.youtube.com/watch?v=BrsocJb-fAo)
- [Vector Databases Explained (Fireship)](https://www.youtube.com/watch?v=klTvEwg3oJ4)
- [What are Vector Embeddings? (IBM Technology)](https://www.youtube.com/watch?v=dN0lsF2cvm4)

### Deep Dives
- [Sentence Transformers Explained (James Briggs)](https://www.youtube.com/watch?v=OATCgQtNX2o)
- [5 Levels of Text Splitting (Greg Kamradt)](https://www.youtube.com/watch?v=8OJC21T2SL4)
- [Chunking for RAG: Best Practices (James Briggs)](https://www.youtube.com/watch?v=eGE5AraIkNM)
- [FAISS Introduction (James Briggs)](https://www.youtube.com/watch?v=sKyvsdEv6rk)
- [ChromaDB Tutorial (pixegami)](https://www.youtube.com/watch?v=QSW2L8dkaZk)

### Advanced
- [Advanced RAG Techniques (Sam Witteveen)](https://www.youtube.com/watch?v=TRjq7t2Ms5I)
- [Hybrid Search for RAG (James Briggs)](https://www.youtube.com/watch?v=lYxGYXjfrNI)
- [Evaluating RAG with RAGAS (James Briggs)](https://www.youtube.com/watch?v=25mMPVt-gRk)
- [Ollama + Python RAG (pixegami)](https://www.youtube.com/watch?v=d0o89z134CQ)

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'sentence_transformers'"
```bash
pip install sentence-transformers
```

### "ModuleNotFoundError: No module named 'chromadb'"
```bash
pip install chromadb
```

### "ModuleNotFoundError: No module named 'faiss'"
```bash
pip install faiss-cpu    # CPU version
pip install faiss-gpu    # GPU version (NVIDIA only)
```

### "Connection refused" when querying (Ollama not running)
```bash
# Install Ollama: https://ollama.com/
ollama serve              # Start the server
ollama pull llama3.2      # Pull the model
```

### ChromaDB "sqlite3" version error
```bash
pip install pysqlite3-binary
```
Then add to your script before importing chromadb:
```python
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
```

### FAISS dimension mismatch error
Ensure you use the **same embedding model** for indexing AND searching. If you change models, you must re-index all documents.

### Poor retrieval quality
1. Check `context_relevancy` score with `RAGEvaluator`
2. Try smaller chunk sizes (500-800 chars)
3. Increase `top_k` to retrieve more candidates
4. Try `RecursiveChunker` instead of `FixedSizeChunker`
5. Check that your documents actually contain the answer

---

## Module File Map

```
agentexplorr/rag/
├── __init__.py              # Module exports and RAG overview docstring
├── document_processor.py    # Load PDF, Markdown, Text, HTML, CSV files
├── chunking.py              # FixedSize, Recursive, and Semantic chunkers
├── embeddings.py            # sentence-transformers wrapper
├── vector_stores/
│   ├── __init__.py          # ChromaDB and FAISS exports
│   ├── chroma_store.py      # ChromaDB vector store
│   └── faiss_store.py       # FAISS vector store
├── retriever.py             # Hybrid retriever with RRF re-ranking
├── rag_pipeline.py          # End-to-end RAG pipeline with Ollama
├── evaluation.py            # Context relevancy, faithfulness, answer relevancy
└── README.md                # This file — learning guide
```

---

*Built with love as part of [AgentExplorr](https://github.com/agentexplorr) — the open-source AI/ML playground. No commercial APIs required.*
