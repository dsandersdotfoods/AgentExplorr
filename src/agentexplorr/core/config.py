"""
Centralized Configuration via Pydantic Settings
================================================

WHY PYDANTIC SETTINGS?
  Pydantic Settings gives us type-safe configuration that reads from environment
  variables AND .env files automatically. If someone sets OLLAMA_BASE_URL="not-a-url",
  Pydantic will raise a clear validation error at startup — not at runtime when
  you're debugging why your agent isn't working.

HOW IT WORKS:
  1. Define fields with types and defaults in a BaseSettings subclass
  2. Pydantic auto-reads matching env vars (case-insensitive)
  3. You can also load from a .env file via model_config

LEARNING RESOURCES:
  - Pydantic Settings docs: https://docs.pydantic.dev/latest/concepts/pydantic_settings/
  - 12-Factor App (config): https://12factor.net/config
  - VIDEO: "Pydantic V2 Tutorial" — https://www.youtube.com/watch?v=502XOB0u8OY
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings loaded from environment variables.

    All settings can be overridden by setting the corresponding environment
    variable. For example, to change the Ollama model:

        export OLLAMA_MODEL=mistral

    Or add it to your .env file:

        OLLAMA_MODEL=mistral
    """

    # --- Pydantic Settings config ---
    # This tells Pydantic where to find the .env file and how to read it.
    # `env_file = ".env"` means it looks for a .env file in the current directory.
    # `case_sensitive = False` means OLLAMA_MODEL and ollama_model both work.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Don't fail on unknown env vars
    )

    # --- Ollama (Local LLM Server) ---
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for the Ollama API server",
    )
    ollama_model: str = Field(
        default="llama3.2",
        description="Default Ollama model for agents and RAG",
    )

    # --- Hugging Face ---
    hf_token: str = Field(
        default="",
        description="Hugging Face API token (for gated models/datasets)",
    )

    # --- Embeddings ---
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence-transformers model for embeddings",
    )

    # --- MLflow ---
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5000",
        description="MLflow tracking server URI",
    )
    mlflow_experiment_name: str = Field(
        default="agentexplorr",
        description="Default MLflow experiment name",
    )

    # --- Application ---
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level",
    )
    environment: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Runtime environment",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get cached application settings.

    WHY CACHE?
      We use @lru_cache so the Settings object is created once and reused.
      This avoids re-reading .env files on every function call, and ensures
      all modules see the exact same configuration.

    Usage:
        from agentexplorr.core.config import get_settings

        settings = get_settings()
        print(settings.ollama_model)  # "llama3.2"
    """
    return Settings()
