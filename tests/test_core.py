"""
Tests for Core Utilities
==========================

Test configuration loading, logging setup, and utility functions.
"""

from __future__ import annotations

from pathlib import Path

from agentexplorr.core.config import Settings
from agentexplorr.core.logging import get_logger
from agentexplorr.core.utils import ensure_directory, load_yaml_config, timer


class TestSettings:
    """Tests for Pydantic Settings configuration."""

    def test_default_values(self) -> None:
        """Test that defaults are sensible."""
        settings = Settings()
        assert settings.ollama_base_url == "http://localhost:11434"
        assert settings.ollama_model == "llama3.2"
        assert settings.log_level == "INFO"

    def test_custom_values(self) -> None:
        """Test overriding defaults."""
        settings = Settings(ollama_model="mistral", log_level="DEBUG")
        assert settings.ollama_model == "mistral"
        assert settings.log_level == "DEBUG"

    def test_environment_field(self) -> None:
        """Test environment validation."""
        settings = Settings(environment="production")
        assert settings.environment == "production"


class TestLogging:
    """Tests for structured logging setup."""

    def test_get_logger(self) -> None:
        """Test logger creation."""
        logger = get_logger("test_module")
        assert logger is not None


class TestUtils:
    """Tests for utility functions."""

    def test_load_yaml_config(self, sample_yaml_config: Path) -> None:
        """Test YAML config loading."""
        config = load_yaml_config(sample_yaml_config)
        assert config["model"]["name"] == "test-model"
        assert config["training"]["learning_rate"] == 0.0002

    def test_load_yaml_missing_file(self) -> None:
        """Test that missing YAML raises FileNotFoundError."""
        import pytest

        with pytest.raises(FileNotFoundError):
            load_yaml_config("/nonexistent/config.yaml")

    def test_ensure_directory(self, tmp_dir: Path) -> None:
        """Test directory creation."""
        new_dir = tmp_dir / "subdir" / "nested"
        result = ensure_directory(new_dir)
        assert result.exists()
        assert result.is_dir()

    def test_timer(self, capsys: object) -> None:
        """Test timer context manager prints elapsed time."""
        import io
        import sys

        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()

        with timer("test operation"):
            pass  # Instant operation

        output = buffer.getvalue()
        sys.stdout = old_stdout

        assert "test operation" in output
        assert "s" in output  # Contains seconds
