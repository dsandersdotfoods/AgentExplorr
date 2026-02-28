"""
Shared Utility Functions
========================

Small, reusable helpers used across the project. Each function is
intentionally simple — complex logic belongs in its respective module.

DESIGN PRINCIPLE:
  Keep utils lean. If a utility grows beyond ~20 lines, it probably
  deserves its own module. Bloated utils.py files are an anti-pattern.

LEARNING RESOURCES:
  - Context managers: https://docs.python.org/3/library/contextlib.html
  - pathlib (modern file paths): https://docs.python.org/3/library/pathlib.html
  - VIDEO: "Python Context Managers" — https://www.youtube.com/watch?v=2VIAL2sMpWs
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

import yaml


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file and return it as a dictionary.

    WHY YAML?
      YAML is widely used for ML configs because it supports:
      - Comments (JSON doesn't!)
      - Multi-line strings (for prompts)
      - Anchors & references (DRY configs)

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed YAML content as a dictionary.

    Raises:
        FileNotFoundError: If the YAML file doesn't exist.
        yaml.YAMLError: If the YAML is malformed.

    Example:
        >>> config = load_yaml_config("configs/training.yaml")
        >>> print(config["learning_rate"])
        0.0002
    """
    path = Path(path)
    if not path.exists():
        msg = f"Config file not found: {path}"
        raise FileNotFoundError(msg)

    with open(path) as f:
        # safe_load prevents arbitrary Python object execution
        # NEVER use yaml.load() with untrusted input!
        data: dict[str, Any] = yaml.safe_load(f) or {}
    return data


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if it doesn't exist, return the Path.

    Useful for creating output directories before saving model checkpoints,
    datasets, or other artifacts.

    Args:
        path: Directory path to ensure exists.

    Returns:
        The Path object (for chaining).

    Example:
        >>> output_dir = ensure_directory("outputs/experiment_01")
        >>> model.save(output_dir / "model.pt")
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


@contextmanager
def timer(label: str = "Operation") -> Generator[None, None, None]:
    """Context manager that times a block of code and prints the duration.

    HOW CONTEXT MANAGERS WORK:
      A context manager is anything that implements __enter__ and __exit__.
      The @contextmanager decorator lets us write them as generators:
      - Code before `yield` runs on __enter__ (start timing)
      - Code after `yield` runs on __exit__ (stop timing)

    Usage:
        >>> with timer("Training epoch 1"):
        ...     model.train(epoch=1)
        ⏱ Training epoch 1: 42.31s

    Args:
        label: Human-readable description of what's being timed.
    """
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"⏱ {label}: {elapsed:.2f}s")
