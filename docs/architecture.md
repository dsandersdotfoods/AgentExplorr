# Architecture

## System Overview

AgentExplorr is organized as a Python monorepo with six independent modules, each exploring a different area of modern AI/ML. All modules share a common core for configuration, logging, and utilities.

```
                            ┌─────────────────────┐
                            │     AgentExplorr     │
                            │   (pyproject.toml)   │
                            └──────────┬──────────┘
                                       │
                 ┌─────────────────────┼─────────────────────┐
                 │                     │                     │
          ┌──────┴──────┐    ┌────────┴────────┐   ┌───────┴───────┐
          │    Core     │    │   Notebooks     │   │    Tests      │
          │ config/log  │    │  (interactive)  │   │  (pytest)     │
          └──────┬──────┘    └─────────────────┘   └───────────────┘
                 │
    ┌────────────┼────────────┬────────────┬────────────┬────────────┐
    │            │            │            │            │            │
┌───┴───┐  ┌────┴────┐  ┌────┴────┐  ┌────┴────┐  ┌───┴────┐  ┌───┴────┐
│Agents │  │   MCP   │  │  LLM    │  │Classical│  │  RAG   │  │Prompt  │
│       │  │         │  │Training │  │  ML/DL  │  │        │  │  Eng.  │
│LangGr.│  │ Servers │  │ LoRA    │  │ sklearn │  │ChromaDB│  │Jinja2  │
│Ollama │  │ Clients │  │ QLoRA   │  │ PyTorch │  │ FAISS  │  │Few-shot│
│Tools  │  │ Tools   │  │ HF PEFT │  │ MLflow  │  │Sentence│  │  CoT   │
└───────┘  └─────────┘  └─────────┘  └─────────┘  └────────┘  └────────┘
```

## Module Dependencies

```
prompt_engineering  ──►  (standalone, no inter-module deps)
rag                 ──►  core, prompt_engineering
agents              ──►  core, prompt_engineering
mcp                 ──►  core
llm_training        ──►  core
classical_ml        ──►  core
```

## Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Package Manager | uv | Fast, modern Python dependency management |
| Build System | hatchling | PEP 517 compliant build backend |
| LLM Inference | Ollama | Local, open source LLM server |
| Agent Framework | LangGraph | Stateful agent orchestration |
| Fine-tuning | HF Transformers + PEFT | LoRA/QLoRA parameter-efficient fine-tuning |
| Classical ML | scikit-learn | Traditional ML pipelines |
| Deep Learning | PyTorch | Neural network training |
| Vector Store | ChromaDB + FAISS | Embedding storage and similarity search |
| Embeddings | sentence-transformers | Local text embeddings |
| Experiment Tracking | MLflow | Log parameters, metrics, artifacts |
| Linting | Ruff | Fast Python linter + formatter |
| Type Checking | mypy (strict) | Static type analysis |
| Testing | pytest | Unit and integration tests |
| CI/CD | GitHub Actions | Automated lint, typecheck, test |
| Containerization | Docker + Compose | Reproducible environments |

## Data Flow

### RAG Pipeline
```
Documents → Chunking → Embedding → Vector Store → Retrieval → LLM → Answer
  (PDF)     (split)   (sentence-   (ChromaDB/    (top-k     (Ollama)
             text)     transformers) FAISS)       similar)
```

### Agent Loop
```
User Query → LLM (Think) → Tool Selection → Tool Execution → Observation → LLM (Think) → ...
                  ↑                                               │
                  └───────────────────────────────────────────────┘
```

### Fine-tuning Pipeline
```
HF Dataset → Format → Tokenize → LoRA Config → SFTTrainer → Evaluate → Merge & Save
(Alpaca)    (instruction  (pad/    (r, alpha,   (training    (perplexity, (adapter +
             format)      truncate) targets)    loop)        ROUGE)       base model)
```
