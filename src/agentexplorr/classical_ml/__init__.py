"""
Classical Machine Learning & Deep Learning Module
===================================================

This module is the heart of the AgentExplorr ML playground. It covers the full
spectrum of machine learning — from classical algorithms (Random Forests, Ridge
Regression, KMeans) to modern deep learning architectures (CNNs, Transformers).

WHY THIS MODULE EXISTS:
    Every ML/AI engineer needs strong fundamentals. LLMs and RAG are built ON TOP
    of these concepts. If you don't understand gradient descent, you can't debug
    fine-tuning. If you don't understand attention, you can't reason about prompt
    engineering at a deep level.

    This module gives you WORKING code for every major ML paradigm, with
    educational comments explaining the "why" behind every design choice.

MODULE STRUCTURE:
    classical_ml/
    ├── __init__.py                  ← You are here
    ├── README.md                    ← Complete learning guide & roadmap
    ├── experiment_tracking.py       ← MLflow experiment tracking wrapper
    ├── pipelines/
    │   ├── classification.py        ← Classification (Random Forest, GridSearchCV)
    │   ├── regression.py            ← Regression (Ridge, Lasso, ElasticNet)
    │   └── clustering.py            ← Clustering (KMeans, DBSCAN)
    └── deep_learning/
        ├── cnn.py                   ← Convolutional Neural Network (CIFAR-10)
        ├── transformer.py           ← Transformer from scratch (Attention Is All You Need)
        └── training_loop.py         ← Reusable PyTorch training loop

LEARNING ROADMAP (suggested order):
    1. classification.py   — Start here. sklearn pipelines are the bread & butter.
    2. regression.py       — Understand bias-variance tradeoff & regularization.
    3. clustering.py       — Unsupervised learning: when you have no labels.
    4. cnn.py              — Your first neural network. Understand convolutions.
    5. training_loop.py    — How PyTorch training actually works under the hood.
    6. transformer.py      — THE architecture behind GPT, BERT, and all modern LLMs.
    7. experiment_tracking  — Professional ML: track everything with MLflow.

LEARNING RESOURCES:
    - Scikit-learn User Guide: https://scikit-learn.org/stable/user_guide.html
    - PyTorch Tutorials: https://pytorch.org/tutorials/
    - Stanford CS229 (Andrew Ng): https://cs229.stanford.edu/
    - Fast.ai (Practical Deep Learning): https://course.fast.ai/
    - VIDEO: "Machine Learning Roadmap" — https://www.youtube.com/watch?v=pHiMN_ez9GQ
    - VIDEO: "But what is a neural network?" (3Blue1Brown) — https://www.youtube.com/watch?v=aircAruvnKk
    - BOOK: "Hands-On ML with Scikit-Learn, Keras & TensorFlow" (Aurélien Géron)
    - BOOK: "Deep Learning" (Goodfellow, Bengio, Courville) — https://www.deeplearningbook.org/
"""

from __future__ import annotations

# --- Deep Learning Imports ---
# PyTorch-based neural network architectures and training utilities.
from agentexplorr.classical_ml.deep_learning.cnn import SimpleCNN
from agentexplorr.classical_ml.deep_learning.training_loop import Trainer
from agentexplorr.classical_ml.deep_learning.transformer import MiniTransformer

# --- Experiment Tracking ---
from agentexplorr.classical_ml.experiment_tracking import ExperimentTracker

# --- Pipeline Imports ---
# These are the classical ML pipelines wrapping sklearn functionality
# into clean, reusable classes with built-in evaluation.
from agentexplorr.classical_ml.pipelines.classification import ClassificationPipeline
from agentexplorr.classical_ml.pipelines.clustering import ClusteringPipeline
from agentexplorr.classical_ml.pipelines.regression import RegressionPipeline

__all__ = [
    # Classical ML Pipelines
    "ClassificationPipeline",
    "RegressionPipeline",
    "ClusteringPipeline",
    # Deep Learning
    "SimpleCNN",
    "MiniTransformer",
    "Trainer",
    # Experiment Tracking
    "ExperimentTracker",
]
