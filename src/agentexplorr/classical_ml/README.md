# Classical ML & Deep Learning

> From decision trees to transformers — the fundamentals of machine learning.

## What You'll Learn

| File | Concept | Difficulty |
|------|---------|-----------|
| `pipelines/classification.py` | sklearn pipelines + GridSearchCV | Beginner |
| `pipelines/regression.py` | Ridge/Lasso/ElasticNet regression | Beginner |
| `pipelines/clustering.py` | KMeans, DBSCAN, silhouette analysis | Beginner |
| `deep_learning/cnn.py` | PyTorch CNN for CIFAR-10 | Intermediate |
| `deep_learning/transformer.py` | Transformer from scratch! | Advanced |
| `deep_learning/training_loop.py` | Reusable PyTorch trainer | Intermediate |
| `experiment_tracking.py` | MLflow experiment tracking | Intermediate |

## Architecture

```
Classical ML & Deep Learning
├── Pipelines (sklearn)          ← Traditional ML
│   ├── Classification           Wine Quality dataset
│   ├── Regression               California Housing
│   └── Clustering               Iris dataset
├── Deep Learning (PyTorch)      ← Neural Networks
│   ├── CNN                      Image classification (CIFAR-10)
│   ├── Transformer              Built from scratch!
│   └── Training Loop            Reusable trainer
└── Experiment Tracking          ← MLflow
```

## Quick Start

### Classification Pipeline
```python
from agentexplorr.classical_ml.pipelines.classification import ClassificationPipeline

pipeline = ClassificationPipeline()
metrics = pipeline.train_on_wine()
print(f"Accuracy: {metrics['accuracy']:.4f}")
```

### Transformer from Scratch
```python
import torch
from agentexplorr.classical_ml.deep_learning.transformer import MiniTransformer

model = MiniTransformer(vocab_size=10000, d_model=256, n_heads=4, n_layers=4)
input_ids = torch.randint(0, 10000, (2, 32))
logits = model(input_ids)  # (2, 32, 10000)
print(f"Parameters: {model.count_parameters():,}")
```

### Training with the Trainer
```python
from agentexplorr.classical_ml.deep_learning.training_loop import Trainer

trainer = Trainer(model, optimizer, loss_fn, device="cuda")
history = trainer.fit(train_loader, val_loader, epochs=20, patience=5)
```

## Key Concepts

### Bias-Variance Tradeoff
- **Bias**: Model is too simple → underfits → misses patterns
- **Variance**: Model is too complex → overfits → memorizes noise
- **Sweet spot**: Regularization (L1/L2), cross-validation, early stopping

### The Transformer Architecture
The most important architecture in modern AI. See `deep_learning/transformer.py` for a complete, from-scratch implementation with detailed explanations of:
- Self-attention and the Q, K, V mechanism
- Positional encoding (sinusoidal)
- Layer normalization and residual connections
- Multi-head attention

## Learning Resources

### Videos
- [3Blue1Brown — Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) — The best visual intro
- [Andrej Karpathy — Let's Build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) — Build a transformer from scratch
- [StatQuest — Machine Learning](https://www.youtube.com/playlist?list=PLblh5JKOoLUICTaGLRoHQDuF_7q2GfuJF) — Clear explanations
- [PyTorch in 100 Seconds](https://www.youtube.com/watch?v=ORMx45xqWkA) — Quick overview

### Papers
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — The Transformer paper
- [Deep Residual Learning](https://arxiv.org/abs/1512.03385) — ResNets (skip connections)
- [Batch Normalization](https://arxiv.org/abs/1502.03167) — Stabilizing training

### Courses
- [CS231n](https://cs231n.github.io/) — Stanford's CNN course (free)
- [Fast.ai](https://www.fast.ai/) — Practical deep learning (free)
- [sklearn User Guide](https://scikit-learn.org/stable/user_guide.html) — Comprehensive reference
