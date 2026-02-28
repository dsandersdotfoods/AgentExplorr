"""
Math Foundations for Machine Learning
======================================

This module teaches the MATH behind machine learning — the stuff that
textbooks assume you know but rarely explain well. Every concept is
implemented from scratch with worked examples so you can run the code
and see the numbers for yourself.

WHY THIS MODULE EXISTS:
    ML frameworks (PyTorch, sklearn) hide the math behind clean APIs.
    That's great for productivity, but terrible for understanding.

    When something goes wrong (loss explodes, gradients vanish, model
    doesn't converge), you need to understand the math to debug it.
    This module gives you that understanding.

WHAT'S COVERED:
    1. backpropagation.py    — Chain rule, computational graphs, gradient flow
    2. optimizers.py         — SGD, Momentum, Adam, AdamW update rules
    3. loss_functions.py     — MSE, Cross-Entropy, NLL with full derivations
    4. activations.py        — ReLU, GELU, Sigmoid, Softmax with derivatives
    5. linear_algebra_essentials.py — Dot products, matrix ops, SVD, eigenvalues

HOW TO USE THIS MODULE:
    Each file is both a library AND a tutorial. You can:
    1. Read the docstrings to learn the math
    2. Import the classes to use them
    3. Run each file directly (`python -m agentexplorr.foundations.backpropagation`)
       to see interactive demos with worked examples

LEARNING PATH (RECOMMENDED ORDER):
    1. linear_algebra_essentials.py   (foundation for everything)
    2. activations.py                 (needed for neural networks)
    3. loss_functions.py              (how we measure "wrong")
    4. backpropagation.py             (how gradients flow)
    5. optimizers.py                  (how weights update)

    After this module, go to:
    - classical_ml/deep_learning/transformer.py  (apply everything)
    - classical_ml/deep_learning/cnn.py          (apply to images)
    - llm_training/fine_tune_lora.py             (apply to LLMs)

PREREQUISITES:
    - Basic Python (functions, classes)
    - High school algebra (variables, equations)
    - That's it. We explain everything else.
"""

from agentexplorr.foundations.activations import (
    ActivationFunction,
    gelu,
    leaky_relu,
    relu,
    sigmoid,
    softmax,
    tanh,
)
from agentexplorr.foundations.backpropagation import ComputationalGraph, Node
from agentexplorr.foundations.linear_algebra_essentials import (
    LinearAlgebraTeacher,
)
from agentexplorr.foundations.loss_functions import (
    LossFunction,
    binary_cross_entropy,
    cross_entropy_loss,
    huber_loss,
    mean_squared_error,
)
from agentexplorr.foundations.optimizers import SGD, Adam, AdamW, Momentum

__all__ = [
    "SGD",
    "ActivationFunction",
    "Adam",
    "AdamW",
    "ComputationalGraph",
    "LinearAlgebraTeacher",
    "LossFunction",
    "Momentum",
    "Node",
    "binary_cross_entropy",
    "cross_entropy_loss",
    "gelu",
    "huber_loss",
    "leaky_relu",
    "mean_squared_error",
    "relu",
    "sigmoid",
    "softmax",
    "tanh",
]
