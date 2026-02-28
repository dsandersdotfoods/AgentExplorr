# Getting Started

## Prerequisites

- **Python 3.11+** — [Download](https://www.python.org/downloads/)
- **uv** — [Install](https://docs.astral.sh/uv/getting-started/installation/)
- **Ollama** — [Install](https://ollama.ai/download) (for local LLM inference)
- **Git** — [Download](https://git-scm.com/downloads)

## Quick Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/AgentExplorr.git
cd AgentExplorr
```

### 2. Install Dependencies

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies
make install

# Or install specific modules only:
uv sync --extra agents    # Just agents
uv sync --extra rag       # Just RAG
uv sync --extra training  # Just fine-tuning
```

### 3. Set Up Ollama

```bash
# Install Ollama (macOS/Linux)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model (llama3.2 is small and fast for development)
ollama pull llama3.2

# Verify it's running
curl http://localhost:11434/api/tags
```

### 4. Configure Environment

```bash
cp .env.example .env
# Edit .env with your settings (all defaults work out of the box)
```

### 5. Verify Installation

```bash
# Run linting
make lint

# Run type checking
make typecheck

# Run tests
make test
```

## Module Walkthroughs

Each module has its own README with detailed explanations:

| Module | Start Here | What You'll Learn |
|--------|-----------|------------------|
| Prompt Engineering | `src/agentexplorr/prompt_engineering/README.md` | Templates, few-shot, CoT, structured output |
| RAG | `src/agentexplorr/rag/README.md` | Document processing, embeddings, vector stores |
| Agents | `src/agentexplorr/agents/README.md` | ReAct, tool-calling, multi-agent systems |
| MCP | `src/agentexplorr/mcp/README.md` | MCP servers, clients, protocol basics |
| LLM Training | `src/agentexplorr/llm_training/README.md` | LoRA, QLoRA, fine-tuning pipelines |
| Classical ML | `src/agentexplorr/classical_ml/README.md` | sklearn, PyTorch, MLflow |

## Interactive Notebooks

Launch JupyterLab to explore interactively:

```bash
make notebook
# Opens http://localhost:8888
```

## Docker Setup (Optional)

```bash
# Build and start all services
make docker-build
make docker-up

# Services:
#   App:     http://localhost:8888 (JupyterLab)
#   MLflow:  http://localhost:5000 (experiment tracking)
#   Chroma:  http://localhost:8000 (vector store)

# Stop services
make docker-down
```

## Learning Path (Recommended Order)

1. **Prompt Engineering** — Foundations, no GPU needed
2. **RAG** — Applied NLP, builds on prompting
3. **Agents** — Combines prompting + tools
4. **MCP** — Protocol for tool-using AI
5. **Classical ML** — Fundamentals of ML
6. **LLM Training** — Advanced, requires GPU

## Troubleshooting

### Ollama not running
```bash
ollama serve  # Start the Ollama server
```

### Import errors
```bash
uv sync --all-extras --dev  # Reinstall all dependencies
```

### Tests failing
```bash
make test-fast  # Skip slow/integration/GPU tests
```
