# Contributing

## Development Setup

```bash
git clone https://github.com/yourusername/AgentExplorr.git
cd AgentExplorr
make install-dev  # Installs deps + pre-commit hooks
```

## Code Standards

- **Type hints** on all functions (mypy strict mode)
- **Docstrings** on all public functions (Google style)
- **Ruff** for linting and formatting (configured in pyproject.toml)
- **Tests** for all new functionality

## Pre-commit Hooks

Pre-commit hooks run automatically on `git commit`:

```bash
# Manual run on all files
make pre-commit-run
```

## Running Tests

```bash
make test        # Full suite with coverage
make test-fast   # Exclude slow/integration tests
```

## Adding a New Module

1. Create the package under `src/agentexplorr/`
2. Add a `README.md` with learning resources
3. Add tests under `tests/test_<module>/`
4. Add a notebook under `notebooks/`
5. Update `pyproject.toml` with new dependencies
6. Update the main `README.md`
