# Math Foundations for Machine Learning

> The math they assume you know — explained from scratch, with code you can run.

## Why This Module Exists

ML frameworks hide the math behind clean APIs. `loss.backward()` computes
gradients. `optimizer.step()` updates weights. But **what's actually happening?**

When things go wrong — loss explodes, gradients vanish, model doesn't converge —
you need to understand the underlying math to debug it. This module gives you
that understanding by implementing everything from scratch with numpy.

## What You'll Learn

| File | Concept | Difficulty |
|------|---------|------------|
| `linear_algebra_essentials.py` | Vectors, matrices, norms, SVD — the math behind LoRA | Beginner |
| `activations.py` | ReLU, GELU, Sigmoid, Softmax — with derivatives | Beginner |
| `loss_functions.py` | MSE, Cross-Entropy, NLL — why each exists | Intermediate |
| `backpropagation.py` | Chain rule, computational graphs, gradient flow | Intermediate |
| `optimizers.py` | SGD, Momentum, Adam, AdamW — update rules derived | Intermediate |

## Recommended Learning Path

```
Start here                                              Apply here
    │                                                       │
    ▼                                                       ▼
┌───────────────────────┐                    ┌──────────────────────────┐
│ 1. Linear Algebra     │                    │ transformer.py           │
│    Vectors, matrices, │                    │ (self-attention uses     │
│    dot products, SVD  │──────┐             │  dot products, matrix    │
└───────────────────────┘      │             │  multiplications, norms) │
                               │             └──────────────────────────┘
┌───────────────────────┐      │             ┌──────────────────────────┐
│ 2. Activations        │      │             │ cnn.py                   │
│    ReLU, GELU,        │──────┤             │ (ReLU, BatchNorm,        │
│    Sigmoid, Softmax   │      │             │  softmax for classes)    │
└───────────────────────┘      │             └──────────────────────────┘
                               │             ┌──────────────────────────┐
┌───────────────────────┐      │             │ fine_tune_lora.py        │
│ 3. Loss Functions     │──────┤             │ (cross-entropy loss,     │
│    MSE, Cross-Entropy │      │             │  low-rank approximation  │
│    NLL, Huber         │      │             │  = SVD connection)       │
└───────────────────────┘      │             └──────────────────────────┘
                               │             ┌──────────────────────────┐
┌───────────────────────┐      │             │ training_loop.py         │
│ 4. Backpropagation    │──────┤             │ (loss.backward() does    │
│    Chain rule, comp.  │      │             │  exactly what we built   │
│    graphs, gradients  │      │             │  manually here)          │
└───────────────────────┘      │             └──────────────────────────┘
                               │             ┌──────────────────────────┐
┌───────────────────────┐      │             │ Any training script      │
│ 5. Optimizers         │──────┘             │ (optimizer.step() does   │
│    SGD, Momentum,     │                    │  exactly what we built   │
│    Adam, AdamW        │                    │  manually here)          │
└───────────────────────┘                    └──────────────────────────┘
```

## Quick Start

### Run Any File Directly
Each file includes a `__main__` block with worked examples:

```bash
# See backpropagation in action
python -m agentexplorr.foundations.backpropagation

# Compare optimizer convergence
python -m agentexplorr.foundations.optimizers

# Understand cross-entropy loss
python -m agentexplorr.foundations.loss_functions

# Explore activation functions
python -m agentexplorr.foundations.activations

# Linear algebra for ML
python -m agentexplorr.foundations.linear_algebra_essentials
```

### Import and Use
```python
from agentexplorr.foundations import (
    # Activations
    relu, sigmoid, gelu, softmax,
    # Loss functions
    cross_entropy_loss, mean_squared_error,
    # Optimizers
    Adam, AdamW, SGD,
    # Backpropagation
    ComputationalGraph, Node,
    # Linear algebra
    LinearAlgebraTeacher,
)

# Example: compute cross-entropy loss and its gradient
import numpy as np
predictions = np.array([0.7, 0.2, 0.1])  # softmax probabilities
targets = np.array([1, 0, 0])            # one-hot labels
loss = cross_entropy_loss(predictions, targets)
print(f"Loss: {loss}")
```

## The Big Picture

Here's how all the math connects:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        NEURAL NETWORK TRAINING                       │
│                                                                      │
│  1. FORWARD PASS (linear_algebra + activations)                     │
│     input → [W₁x + b₁] → [ReLU] → [W₂x + b₂] → [Softmax] → pred │
│                                                                      │
│  2. COMPUTE LOSS (loss_functions)                                    │
│     loss = CrossEntropy(predictions, targets)                        │
│     "How wrong is the model?"                                        │
│                                                                      │
│  3. BACKWARD PASS (backpropagation)                                  │
│     Compute ∂loss/∂W for every weight using the chain rule          │
│     pred ← [∂Softmax] ← [∂W₂] ← [∂ReLU] ← [∂W₁] ← input        │
│                                                                      │
│  4. UPDATE WEIGHTS (optimizers)                                      │
│     W = W - lr * ∂loss/∂W  (SGD)                                   │
│     ...or Adam/AdamW for smarter updates                             │
│                                                                      │
│  5. REPEAT until loss is small enough                                │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Equations Cheat Sheet

| Concept | Formula | File |
|---------|---------|------|
| Dot product | a·b = Σ aᵢbᵢ = \|a\|\|b\|cos(θ) | `linear_algebra_essentials.py` |
| Matrix multiply | (m×n) @ (n×p) → (m×p) | `linear_algebra_essentials.py` |
| L2 norm | \|\|x\|\|₂ = √(Σxᵢ²) | `linear_algebra_essentials.py` |
| SVD | A = UΣV^T | `linear_algebra_essentials.py` |
| ReLU | f(x) = max(0, x) | `activations.py` |
| Sigmoid | σ(x) = 1/(1+e^(-x)) | `activations.py` |
| GELU | f(x) = x·Φ(x) | `activations.py` |
| Softmax | p(xᵢ) = e^xᵢ / Σe^xⱼ | `activations.py` |
| MSE | L = (1/n)Σ(ŷ-y)² | `loss_functions.py` |
| Cross-Entropy | L = -Σ y·log(ŷ) | `loss_functions.py` |
| Chain rule | ∂L/∂w = ∂L/∂y · ∂y/∂w | `backpropagation.py` |
| SGD update | w = w - lr·∇L | `optimizers.py` |
| Adam update | w = w - lr·m̂/(√v̂+ε) | `optimizers.py` |

## Learning Resources

### Videos (Watch in This Order)
1. [3Blue1Brown — Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) — Best visual intro to linear algebra
2. [3Blue1Brown — Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) — Gradient descent, backpropagation visualized
3. [3Blue1Brown — Attention in Transformers](https://www.youtube.com/watch?v=eMlx5fFNoYc) — Connects everything to transformers
4. [StatQuest — Gradient Descent](https://www.youtube.com/watch?v=sDv4f4s2SB8) — Clear, step-by-step
5. [StatQuest — Cross-Entropy](https://www.youtube.com/watch?v=6ArSys5qHAU) — Best explanation of entropy → cross-entropy
6. [Andrej Karpathy — Micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0) — Build an autograd engine from scratch (what we do in backpropagation.py)

### Textbooks (Free Online)
- [Deep Learning Book](https://www.deeplearningbook.org/) — Goodfellow, Bengio, Courville (Chapters 2-8 cover everything here)
- [Mathematics for Machine Learning](https://mml-book.github.io/) — Deisenroth, Faisal, Ong (free PDF)
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) — Michael Nielsen (free, interactive)

### Papers
- [Adam Optimizer](https://arxiv.org/abs/1412.6980) — Kingma & Ba, 2014
- [AdamW (Decoupled Weight Decay)](https://arxiv.org/abs/1711.05101) — Loshchilov & Hutter, 2017
- [LoRA](https://arxiv.org/abs/2106.09685) — Hu et al., 2021 (SVD connection)
- [Batch Normalization](https://arxiv.org/abs/1502.03167) — Ioffe & Szegedy, 2015
- [GELU Activation](https://arxiv.org/abs/1606.08415) — Hendrycks & Gimpel, 2016
- [Xavier Initialization](https://proceedings.mlr.press/v9/glorot10a.html) — Glorot & Bengio, 2010
