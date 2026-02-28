"""
Deep Learning Implementations from Scratch Using PyTorch
==========================================================

This sub-module contains educational, from-scratch implementations of the most
important deep learning architectures. Every class is heavily commented to explain
not just WHAT the code does, but WHY each design choice was made.

MODULE STRUCTURE:
    deep_learning/
    ├── __init__.py          ← You are here
    ├── transformer.py       ← Transformer from scratch (the resume highlight!)
    ├── cnn.py               ← Convolutional Neural Network for CIFAR-10
    └── training_loop.py     ← Reusable PyTorch training loop with best practices

WHY BUILD FROM SCRATCH?
    Using ``nn.TransformerEncoder`` or a pre-built ResNet is fine for production,
    but it teaches you NOTHING about how these architectures actually work.

    Building from scratch forces you to understand:
      - How attention computes relationships between tokens
      - Why we scale dot products by sqrt(d_k)
      - How convolutions learn spatial features
      - Why residual connections prevent vanishing gradients
      - How layer normalization stabilizes training

    These are the questions interviewers ask, and the understanding you need
    to debug real ML systems.

LEARNING RESOURCES:
    - "Attention Is All You Need" (Vaswani et al., 2017):
      https://arxiv.org/abs/1706.03762
    - "Deep Residual Learning" (He et al., 2015):
      https://arxiv.org/abs/1512.03385
    - PyTorch Tutorials: https://pytorch.org/tutorials/
    - 3Blue1Brown "Neural Networks" series:
      https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
    - Andrej Karpathy "Neural Networks: Zero to Hero":
      https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ
"""

from __future__ import annotations

from agentexplorr.classical_ml.deep_learning.cnn import SimpleCNN
from agentexplorr.classical_ml.deep_learning.training_loop import Trainer
from agentexplorr.classical_ml.deep_learning.transformer import MiniTransformer

__all__ = [
    "SimpleCNN",
    "MiniTransformer",
    "Trainer",
]
