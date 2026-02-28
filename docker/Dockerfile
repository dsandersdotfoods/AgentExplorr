# =============================================================================
# AgentExplorr — CPU Docker Image
# =============================================================================
# Multi-stage build for a lean production image.
#
# Usage:
#   docker build -f docker/Dockerfile -t agentexplorr .
#   docker run -it --rm agentexplorr python -c "import agentexplorr; print('Ready!')"
#
# LEARNING RESOURCE:
#   - Docker best practices: https://docs.docker.com/build/building/best-practices/
#   - VIDEO: "Docker in 100 Seconds" — https://www.youtube.com/watch?v=Gjnup-PuquQ
# =============================================================================

# Stage 1: Build — install dependencies in an isolated layer
FROM python:3.12-slim AS builder

# Install uv (the package manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files first (cache layer — only rebuilds when deps change)
COPY pyproject.toml uv.lock* ./

# Install all extras except GPU-specific packages
RUN uv sync --all-extras --no-dev --no-install-project

# Copy source code
COPY src/ ./src/
COPY configs/ ./configs/

# Install the project itself
RUN uv sync --all-extras --no-dev

# Stage 2: Runtime — slim image with only what's needed
FROM python:3.12-slim AS runtime

WORKDIR /app

# Copy the virtual environment from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY --from=builder /app/configs /app/configs

# Add venv to PATH
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src:$PYTHONPATH"

# Non-root user for security
RUN useradd --create-home appuser
USER appuser

CMD ["python", "-m", "agentexplorr"]
