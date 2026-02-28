# =============================================================================
# AgentExplorr — Makefile
# =============================================================================
# Common commands for development, testing, and deployment.
#
# Usage:
#   make install      — Install all dependencies (first time setup)
#   make lint         — Run linter (ruff)
#   make test         — Run test suite
#   make all          — Lint + typecheck + test (CI equivalent)
#
# LEARNING RESOURCE:
#   - Makefile tutorial: https://makefiletutorial.com/
#   - VIDEO: "Makefiles for Python" — https://www.youtube.com/watch?v=x_6Cdi8WvUE
# =============================================================================

.PHONY: help install install-dev lint format typecheck test test-fast \
        notebook clean docker-build docker-up docker-down pre-commit-install \
        pre-commit-run download-data all

# Default target — show help
help: ## Show this help message
	@echo "AgentExplorr — Available commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

install: ## Install all dependencies (first time setup)
	uv sync --all-extras --dev
	@echo "\n✅ All dependencies installed. Run 'make pre-commit-install' for git hooks."

install-dev: ## Install with pre-commit hooks
	uv sync --all-extras --dev
	uv run pre-commit install
	@echo "\n✅ Dependencies + pre-commit hooks installed."

# ---------------------------------------------------------------------------
# Code Quality
# ---------------------------------------------------------------------------

lint: ## Run linter (ruff check + format check)
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/

format: ## Auto-fix lint issues and format code
	uv run ruff check --fix src/ tests/
	uv run ruff format src/ tests/

typecheck: ## Run static type checker (mypy)
	uv run mypy src/

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------

test: ## Run full test suite with coverage
	uv run pytest

test-fast: ## Run tests excluding slow/integration tests
	uv run pytest -m "not slow and not integration and not gpu"

# ---------------------------------------------------------------------------
# Notebooks
# ---------------------------------------------------------------------------

notebook: ## Launch JupyterLab
	uv run jupyter lab --notebook-dir=notebooks/

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

download-data: ## Download open source datasets
	uv run python scripts/download_datasets.py

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------

docker-build: ## Build Docker images
	docker compose -f docker/docker-compose.yml build

docker-up: ## Start all services (app + MLflow + ChromaDB)
	docker compose -f docker/docker-compose.yml up -d

docker-down: ## Stop all services
	docker compose -f docker/docker-compose.yml down

# ---------------------------------------------------------------------------
# Pre-commit
# ---------------------------------------------------------------------------

pre-commit-install: ## Install pre-commit git hooks
	uv run pre-commit install

pre-commit-run: ## Run pre-commit on all files
	uv run pre-commit run --all-files

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

clean: ## Remove build artifacts, caches, and temp files
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov/
	rm -rf mlruns/ chroma_data/ faiss_index/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Cleaned."

# ---------------------------------------------------------------------------
# CI — run everything
# ---------------------------------------------------------------------------

all: lint typecheck test ## Run lint + typecheck + test (CI equivalent)
