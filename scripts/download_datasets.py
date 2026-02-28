"""
Download Open Source Datasets
==============================

This script downloads all datasets used in the AgentExplorr project.
All datasets are free, open source, and publicly available.

Usage:
  python scripts/download_datasets.py
  # or
  make download-data

DATASETS DOWNLOADED:
  1. Alpaca (52K instruction-following examples) — for LLM fine-tuning
  2. Wine Quality — for classification (via sklearn)
  3. California Housing — for regression (via sklearn)
  4. Iris — for clustering (via sklearn)
  5. CIFAR-10 — for CNN training (via torchvision)

LEARNING RESOURCES:
  - Hugging Face Datasets: https://huggingface.co/docs/datasets/
  - sklearn datasets: https://scikit-learn.org/stable/datasets.html
  - VIDEO: "Hugging Face Datasets Tutorial" — https://www.youtube.com/watch?v=_BZearw7f0w
"""

from __future__ import annotations

import sys
from pathlib import Path


def download_alpaca() -> None:
    """Download the Alpaca instruction-following dataset from Hugging Face.

    WHAT IS ALPACA?
      Stanford Alpaca is a dataset of 52K instruction-following examples
      generated using GPT-3.5. It's one of the most popular datasets for
      fine-tuning LLMs on instruction following.

    Paper: https://arxiv.org/abs/2303.16200
    """
    try:
        from datasets import load_dataset

        print("Downloading Alpaca dataset...")
        dataset = load_dataset("tatsu-lab/alpaca", split="train")
        output_dir = Path("data/alpaca")
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(output_dir))
        print(f"  Saved {len(dataset)} examples to {output_dir}")
    except ImportError:
        print("  Skipping Alpaca (install 'datasets' package: uv sync --extra training)")
    except Exception as e:
        print(f"  Error downloading Alpaca: {e}")


def download_sklearn_datasets() -> None:
    """Download sklearn built-in datasets (they're bundled, just verify access).

    These datasets are included with scikit-learn — no download needed.
    We just verify they load correctly.
    """
    try:
        from sklearn.datasets import (
            fetch_california_housing,
            load_iris,
            load_wine,
        )

        print("Verifying sklearn datasets...")

        wine = load_wine()
        print(f"  Wine Quality: {wine.data.shape[0]} samples, {wine.data.shape[1]} features")

        iris = load_iris()
        print(f"  Iris: {iris.data.shape[0]} samples, {iris.data.shape[1]} features")

        housing = fetch_california_housing()
        print(f"  California Housing: {housing.data.shape[0]} samples, {housing.data.shape[1]} features")

    except ImportError:
        print("  Skipping sklearn datasets (install 'scikit-learn': uv sync --extra ml)")


def download_cifar10() -> None:
    """Download CIFAR-10 image classification dataset.

    WHAT IS CIFAR-10?
      10 classes of 32x32 color images: airplane, automobile, bird, cat,
      deer, dog, frog, horse, ship, truck. 60K images total.

    Paper: https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf
    """
    try:
        from torchvision.datasets import CIFAR10

        print("Downloading CIFAR-10...")
        data_dir = Path("data/cifar10")
        data_dir.mkdir(parents=True, exist_ok=True)
        dataset = CIFAR10(root=str(data_dir), download=True, train=True)
        print(f"  CIFAR-10: {len(dataset)} training images")
    except ImportError:
        print("  Skipping CIFAR-10 (install 'torchvision': uv sync --extra ml)")
    except Exception as e:
        print(f"  Error downloading CIFAR-10: {e}")


def main() -> None:
    """Download all datasets."""
    print("=" * 60)
    print("AgentExplorr — Downloading Open Source Datasets")
    print("=" * 60)
    print()

    download_sklearn_datasets()
    print()
    download_alpaca()
    print()
    download_cifar10()
    print()

    print("=" * 60)
    print("Done! All available datasets downloaded to data/")
    print("=" * 60)


if __name__ == "__main__":
    main()
