"""
Structured Logging with structlog
==================================

WHY STRUCTURED LOGGING?
  Traditional logging outputs unstructured strings like:
    "Processing document doc123 with 5 pages"

  Structured logging outputs key-value pairs:
    event="processing_document" doc_id="doc123" pages=5

  This makes logs searchable, filterable, and parseable by tools like
  Datadog, ELK, or even simple jq commands. In production ML systems,
  structured logging is essential for debugging model behavior.

HOW IT WORKS:
  1. structlog wraps Python's stdlib logging
  2. Each log event is a dict of key-value pairs
  3. Processors transform the event dict (add timestamps, format output, etc.)
  4. The final processor renders it (console or JSON)

LEARNING RESOURCES:
  - structlog docs: https://www.structlog.org/en/stable/
  - Python logging HOWTO: https://docs.python.org/3/howto/logging.html
  - VIDEO: "Structured Logging in Python" — https://www.youtube.com/watch?v=Y5eyEgyHLLo
"""

from __future__ import annotations

import logging
import sys

import structlog

from agentexplorr.core.config import get_settings


def setup_logging() -> None:
    """Configure structlog with rich console output for development.

    This sets up:
      - Timestamps on every log line
      - Log level (INFO, DEBUG, etc.)
      - Logger name (which module emitted the log)
      - Pretty colors in development (via rich)
      - JSON output in production (for log aggregation)
    """
    settings = get_settings()

    # Choose renderer based on environment
    # Development: colorful, human-readable output
    # Production: JSON lines for machine parsing
    if settings.environment == "development":
        renderer = structlog.dev.ConsoleRenderer(colors=True)
    else:
        renderer = structlog.processors.JSONRenderer()  # type: ignore[assignment]

    structlog.configure(
        processors=[
            # Add context vars bound via logger.bind(key=value)
            structlog.contextvars.merge_contextvars,
            # Add log level name (INFO, DEBUG, etc.)
            structlog.stdlib.add_log_level,
            # Add logger name
            structlog.stdlib.add_logger_name,
            # Add ISO-8601 timestamp
            structlog.processors.TimeStamper(fmt="iso"),
            # If in development, add pretty stack traces
            structlog.dev.set_exc_info,
            # Final rendering
            renderer,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Also configure stdlib logging so third-party libraries' logs are captured
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level),
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger for the given module.

    Usage:
        from agentexplorr.core.logging import get_logger

        logger = get_logger(__name__)
        logger.info("processing_document", doc_id="doc123", pages=5)
        logger.error("embedding_failed", model="all-MiniLM-L6-v2", error=str(e))

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        A structlog BoundLogger instance.
    """
    setup_logging()
    return structlog.get_logger(name)
