"""
Shared Test Fixtures — Pytest Configuration
=============================================

WHAT ARE FIXTURES?
  Fixtures are reusable test setup/teardown functions. Instead of repeating
  setup code in every test, you define it once as a fixture and pytest
  automatically injects it into tests that request it.

HOW FIXTURES WORK:
  1. Decorate a function with @pytest.fixture
  2. Name it as a parameter in your test function
  3. Pytest calls the fixture, passes the return value to your test
  4. After the test, the fixture cleans up (if using yield)

SCOPE:
  - "function" (default) — new fixture for each test
  - "session" — one fixture for the entire test run
  - "module" — one fixture per test file

LEARNING RESOURCES:
  - Pytest fixtures: https://docs.pytest.org/en/stable/how-to/fixtures.html
  - VIDEO: "Pytest Tutorial" — https://www.youtube.com/watch?v=cHYq1MRoyI0
"""

from __future__ import annotations

import tempfile
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agentexplorr.core.config import Settings


@pytest.fixture
def test_settings() -> Settings:
    """Provide test-specific settings that don't depend on .env files.

    WHY OVERRIDE SETTINGS IN TESTS?
      Tests should be deterministic and not depend on the developer's
      local environment. By providing explicit test settings, we ensure
      tests produce the same results everywhere (local, CI, etc.).
    """
    return Settings(
        ollama_base_url="http://localhost:11434",
        ollama_model="llama3.2",
        hf_token="test-token",
        embedding_model="all-MiniLM-L6-v2",
        mlflow_tracking_uri="http://localhost:5000",
        mlflow_experiment_name="test-experiment",
        log_level="DEBUG",
        environment="development",
    )


@pytest.fixture
def tmp_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory that's cleaned up after the test.

    Uses Python's tempfile module to create a directory that's
    automatically deleted when the test finishes. Perfect for:
    - Testing file I/O
    - Creating temp databases
    - Saving model checkpoints
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_documents() -> list[str]:
    """Provide sample documents for testing RAG and text processing.

    These are short, diverse texts covering different topics.
    Useful for testing chunking, embedding, and retrieval.
    """
    return [
        (
            "Machine learning is a subset of artificial intelligence that enables "
            "systems to learn from data. Instead of being explicitly programmed, "
            "ML algorithms build mathematical models from training data to make "
            "predictions or decisions."
        ),
        (
            "Neural networks are inspired by biological neural networks in the brain. "
            "They consist of layers of interconnected nodes (neurons) that process "
            "information. Deep learning uses neural networks with many layers."
        ),
        (
            "Retrieval Augmented Generation (RAG) combines the power of large language "
            "models with external knowledge retrieval. Instead of relying solely on "
            "the model's training data, RAG fetches relevant documents and includes "
            "them in the prompt context."
        ),
        (
            "Transfer learning is a technique where a model trained on one task is "
            "reused as the starting point for a different task. Fine-tuning is a "
            "common form of transfer learning where a pre-trained model is further "
            "trained on domain-specific data."
        ),
    ]


@pytest.fixture
def mock_llm() -> MagicMock:
    """Provide a mock LLM for testing without API calls.

    WHY MOCK THE LLM?
      Real LLM calls are:
      - Slow (seconds per call)
      - Expensive (API costs)
      - Non-deterministic (different outputs each time)
      - Require network access (fails in CI without API keys)

      Mocking the LLM lets us test our logic in isolation,
      fast and deterministically.
    """
    mock = MagicMock()
    mock.invoke.return_value = MagicMock(content="This is a mock LLM response.")
    return mock


@pytest.fixture
def sample_yaml_config(tmp_dir: Path) -> Path:
    """Create a sample YAML config file for testing config loading."""
    config_path = tmp_dir / "test_config.yaml"
    config_path.write_text(
        """
model:
  name: test-model
  temperature: 0.7
  max_tokens: 512

training:
  learning_rate: 0.0002
  batch_size: 16
  epochs: 3
"""
    )
    return config_path
