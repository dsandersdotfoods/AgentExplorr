<p align="center">
  <h1 align="center">AgentExplorr</h1>
  <p align="center">
    <strong>A comprehensive AI/ML playground exploring agents, MCP, LLM training, RAG, and classical ML — all open source.</strong>
  </p>
  <p align="center">
    <a href="https://github.com/dsandersdotfoods/AgentExplorr/actions"><img src="https://img.shields.io/github/actions/workflow/status/dsandersdotfoods/AgentExplorr/ci.yml?label=CI&style=flat-square" alt="CI"></a>
    <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue?style=flat-square" alt="Python"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="License"></a>
    <a href="https://docs.astral.sh/ruff/"><img src="https://img.shields.io/badge/code%20style-ruff-000000?style=flat-square" alt="Ruff"></a>
    <a href="https://mypy-lang.org/"><img src="https://img.shields.io/badge/types-mypy%20strict-blue?style=flat-square" alt="mypy"></a>
    <a href="https://docs.astral.sh/uv/"><img src="https://img.shields.io/badge/pkg-uv-blueviolet?style=flat-square" alt="uv"></a>
  </p>
</p>

---

## What Is This?

AgentExplorr is a hands-on playground for learning and experimenting with modern AI/ML — from autonomous agents to fine-tuning large language models. Every module is heavily documented with explanations, learning resources, and links to videos and papers.

**100% open source. No commercial API keys required.**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            AgentExplorr                                    │
├──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┤
│  Agents  │   MCP    │   LLM    │Classical │   RAG    │  Prompt  │  Core    │
│          │          │ Training │  ML/DL   │          │   Eng.   │          │
│ LangGraph│ Servers  │ LoRA     │ sklearn  │ ChromaDB │ Jinja2   │ Config   │
│ Ollama   │ Clients  │ QLoRA    │ PyTorch  │ FAISS    │ Few-shot │ Logging  │
│ Tools    │ Tools    │ HF PEFT  │ MLflow   │ sentence-│ CoT      │ Utils    │
│ ReAct    │ JSON-RPC │ TRL      │ CNN      │ transform│ Struct.  │ Pydantic │
│ Multi-ag.│ stdio    │ Datasets │ Transf.  │ Hybrid   │ Output   │ structlog│
└──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘
```

## Modules

### 1. AI Agents
Build autonomous agents that reason and use tools.
- **ReAct Agent** — Reasoning + Acting loop via LangGraph
- **Tool-Calling Agent** — LLM decides which tools to use
- **Multi-Agent System** — Supervisor routes tasks to specialists
- **Custom Tools** — DuckDuckGo search, calculator, web scraper

📁 `src/agentexplorr/agents/` · 📖 [README](src/agentexplorr/agents/README.md)

### 2. Model Context Protocol (MCP)
The open standard for LLM-tool communication.
- **Filesystem Server** — File operations as MCP tools
- **Database Server** — SQL queries with safety checks
- **API Server** — Open-Meteo weather API wrapper
- **MCP Client** — Connect to and use MCP servers
- **Code Executor** — Sandboxed Python execution

📁 `src/agentexplorr/mcp/` · 📖 [README](src/agentexplorr/mcp/README.md)

### 3. LLM Fine-tuning
Train language models on custom data.
- **LoRA** — Low-Rank Adaptation (parameter-efficient fine-tuning)
- **QLoRA** — 4-bit quantized LoRA (fits on consumer GPUs)
- **Data Preparation** — Load & format Hugging Face datasets
- **Evaluation** — Perplexity, BLEU, ROUGE metrics

📁 `src/agentexplorr/llm_training/` · 📖 [README](src/agentexplorr/llm_training/README.md)

### 4. Classical ML & Deep Learning
Fundamentals of machine learning.
- **Classification** — sklearn pipelines + GridSearchCV (Wine dataset)
- **Regression** — Ridge/Lasso/ElasticNet (California Housing)
- **Clustering** — KMeans, DBSCAN, silhouette analysis (Iris)
- **CNN** — PyTorch convolutional neural network (CIFAR-10)
- **Transformer** — Built from scratch! (attention, positional encoding)
- **MLflow** — Experiment tracking and model registry

📁 `src/agentexplorr/classical_ml/` · 📖 [README](src/agentexplorr/classical_ml/README.md)

### 5. RAG (Retrieval Augmented Generation)
Augment LLMs with external knowledge.
- **Document Processing** — PDF, text, markdown ingestion
- **Chunking** — Fixed-size, recursive, semantic strategies
- **Embeddings** — sentence-transformers (local, no API)
- **Vector Stores** — ChromaDB + FAISS implementations
- **Hybrid Retrieval** — Multi-store search with re-ranking
- **Evaluation** — Context relevancy, faithfulness, answer quality

📁 `src/agentexplorr/rag/` · 📖 [README](src/agentexplorr/rag/README.md)

### 6. Prompt Engineering
The art and science of crafting effective prompts.
- **Templates** — Jinja2 prompt templates with variable injection
- **Few-Shot Learning** — Teaching by example (in-context learning)
- **Chain of Thought** — Step-by-step reasoning (zero-shot, few-shot, self-consistency)
- **Structured Output** — Parse LLM responses into Pydantic models

📁 `src/agentexplorr/prompt_engineering/` · 📖 [README](src/agentexplorr/prompt_engineering/README.md)

## Quick Start

### Prerequisites
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (package manager)
- [Ollama](https://ollama.ai/) (local LLM inference)

### Setup

```bash
# Clone
git clone https://github.com/dsandersdotfoods/AgentExplorr.git
cd AgentExplorr

# Install everything
make install

# Pull an LLM model
ollama pull llama3.2

# Configure
cp .env.example .env

# Verify
make lint && make typecheck && make test
```

### Interactive Exploration

```bash
make notebook  # Launch JupyterLab with interactive walkthroughs
```

### Docker (Optional)

```bash
make docker-up  # Starts app + MLflow + ChromaDB
# MLflow UI:  http://localhost:5000
# ChromaDB:   http://localhost:8000
```

## Tech Stack

| Category | Tools |
|----------|-------|
| **LLM Inference** | Ollama (llama3.2, mistral, phi3) |
| **Agent Framework** | LangGraph + LangChain |
| **Fine-tuning** | Hugging Face Transformers, PEFT, TRL |
| **Classical ML** | scikit-learn, PyTorch |
| **Vector Stores** | ChromaDB, FAISS |
| **Embeddings** | sentence-transformers |
| **Experiment Tracking** | MLflow |
| **MCP** | Official Python SDK |
| **Package Manager** | uv |
| **Linting** | Ruff |
| **Type Checking** | mypy (strict mode) |
| **Testing** | pytest + coverage |
| **CI/CD** | GitHub Actions |
| **Containerization** | Docker + Compose |

## Open Source Datasets

| Dataset | Source | Used For |
|---------|--------|----------|
| [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) | Hugging Face | LLM fine-tuning |
| Wine Quality | sklearn | Classification |
| California Housing | sklearn | Regression |
| Iris | sklearn | Clustering |
| CIFAR-10 | torchvision | CNN training |

## Project Structure

```
AgentExplorr/
├── src/agentexplorr/          # Source code (6 modules + core)
│   ├── core/                  # Config, logging, utilities
│   ├── agents/                # AI agents (LangGraph + Ollama)
│   ├── mcp/                   # Model Context Protocol
│   ├── llm_training/          # LoRA/QLoRA fine-tuning
│   ├── classical_ml/          # sklearn + PyTorch + MLflow
│   ├── rag/                   # Retrieval Augmented Generation
│   └── prompt_engineering/    # Templates, few-shot, CoT
├── notebooks/                 # Interactive Jupyter walkthroughs
├── tests/                     # Comprehensive test suite
├── configs/                   # YAML configuration files
├── scripts/                   # Dataset download, utilities
├── docs/                      # Architecture, guides
├── docker/                    # Dockerfile + docker-compose
└── .github/workflows/         # CI/CD pipeline
```

## Development

```bash
make help          # Show all available commands
make lint          # Run linter
make format        # Auto-fix + format
make typecheck     # Run mypy (strict)
make test          # Run tests with coverage
make test-fast     # Skip slow/GPU tests
make all           # Lint + typecheck + test
```

## Learning Path

**Recommended order** (builds on previous knowledge):

1. **Prompt Engineering** → Foundations, no GPU needed
2. **RAG** → Applied NLP, builds on prompting
3. **Agents** → Combines prompting + tools
4. **MCP** → Protocol for tool-using AI
5. **Classical ML** → ML fundamentals
6. **LLM Training** → Advanced, requires GPU

Each module's `README.md` contains concept explanations, code examples, paper references, and video links.

## License

MIT — see [LICENSE](LICENSE)
