# Classical ML & Deep Learning

> From decision trees to transformers -- the fundamentals every ML engineer must master.

This module is the educational backbone of AgentExplorr. Every file contains working
implementations with heavy inline comments explaining the **why** behind every design choice.

---

## Module Map

| File | What You'll Learn | Difficulty | Time |
|------|-------------------|-----------|------|
| `pipelines/classification.py` | sklearn Pipelines, GridSearchCV, confusion matrices | Beginner | 30 min |
| `pipelines/regression.py` | Regularization (Ridge/Lasso/ElasticNet), bias-variance | Beginner | 30 min |
| `pipelines/clustering.py` | Unsupervised learning, KMeans, DBSCAN, silhouette score | Beginner | 30 min |
| `deep_learning/cnn.py` | Convolutions, pooling, batch norm, dropout (CIFAR-10) | Intermediate | 45 min |
| `deep_learning/training_loop.py` | PyTorch training loop, early stopping, LR scheduling | Intermediate | 30 min |
| `deep_learning/transformer.py` | **Self-attention, positional encoding, the full Transformer** | Advanced | 90 min |
| `experiment_tracking.py` | MLflow experiment tracking, model registry | Intermediate | 20 min |

**Suggested reading order:** Classification -> Regression -> Clustering -> CNN -> Training Loop -> Transformer -> Experiment Tracking

---

## Architecture Overview

```
classical_ml/
├── __init__.py                      # Module entry point & learning roadmap
├── README.md                        # You are here
├── experiment_tracking.py           # MLflow wrapper for experiment management
│
├── pipelines/                       # Classical ML (scikit-learn)
│   ├── __init__.py
│   ├── classification.py            # RandomForest + GridSearchCV on Wine dataset
│   ├── regression.py                # Ridge/Lasso/ElasticNet on California Housing
│   └── clustering.py                # KMeans/DBSCAN on Iris dataset
│
└── deep_learning/                   # Neural Networks (PyTorch)
    ├── __init__.py
    ├── cnn.py                       # SimpleCNN for CIFAR-10 (32x32 RGB images)
    ├── transformer.py               # MiniTransformer from scratch (GPT-style)
    └── training_loop.py             # Reusable Trainer with all best practices
```

---

## Quick Start Examples

### 1. Classification Pipeline

```python
from agentexplorr.classical_ml.pipelines.classification import ClassificationPipeline

# Create pipeline
clf = ClassificationPipeline(random_state=42)

# Load Wine Quality dataset
X_train, X_test, y_train, y_test = clf.load_data(test_size=0.2)

# Build sklearn Pipeline (StandardScaler + RandomForestClassifier)
clf.build_pipeline(n_estimators=100)

# Tune hyperparameters with GridSearchCV (5-fold cross-validation)
tuning_results = clf.tune_hyperparameters(X_train, y_train, cv=5)
print(f"Best CV Score: {tuning_results['best_cv_score']:.4f}")

# Evaluate on held-out test set
results = clf.evaluate(X_test, y_test)
print(f"Test Accuracy: {results['accuracy']:.4f}")
print(f"Test F1-Score: {results['f1_score']:.4f}")

# Feature importances
importances = clf.get_feature_importances()
for name, score in list(importances.items())[:5]:
    print(f"  {name}: {score:.4f}")
```

### 2. Regression Pipeline (with Regularization Comparison)

```python
from agentexplorr.classical_ml.pipelines.regression import RegressionPipeline

# Compare Ridge, Lasso, and ElasticNet
for model_name in ["ridge", "lasso", "elasticnet"]:
    reg = RegressionPipeline(model_name=model_name, alpha=1.0)
    X_train, X_test, y_train, y_test = reg.load_data()
    reg.build_pipeline()
    cv_results = reg.fit_with_cross_validation(X_train, y_train, cv=5)
    results = reg.evaluate(X_test, y_test)
    print(f"{model_name}: R2={results['r2']:.4f}, RMSE={results['rmse']:.4f}")

# Compare different alpha values for Lasso (feature selection)
lasso = RegressionPipeline(model_name="lasso", alpha=0.1)
lasso.feature_names = reg.feature_names
lasso.compare_alphas(X_train, y_train, X_test, y_test)
```

### 3. Clustering Pipeline (with Elbow Method)

```python
from agentexplorr.classical_ml.pipelines.clustering import ClusteringPipeline

# KMeans clustering
clust = ClusteringPipeline(algorithm="kmeans", n_clusters=3)
X, y_true = clust.load_data()
clust.build_pipeline()
labels = clust.fit(X)
results = clust.evaluate(X)
print(f"Silhouette Score: {results['silhouette_score']:.4f}")

# Find optimal K using elbow method
elbow = clust.elbow_method(X, k_range=range(2, 11))
print(f"Optimal K: {elbow['optimal_k']}")

# PCA visualization (reduce 4D -> 2D)
pca_2d = clust.pca_reduce(X, n_components=2)
```

### 4. CNN for Image Classification

```python
import torch
from agentexplorr.classical_ml.deep_learning.cnn import SimpleCNN, create_cifar10_dataloaders

# Create model
model = SimpleCNN(num_classes=10, dropout_rate=0.5)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Test forward pass
x = torch.randn(8, 3, 32, 32)  # batch of 8 RGB images
logits = model(x)               # (8, 10) -- one score per class
probs, classes = model.predict(x)
```

### 5. Transformer from Scratch (THE Resume Highlight)

```python
import torch
from agentexplorr.classical_ml.deep_learning.transformer import MiniTransformer

# Build a small GPT-style Transformer
model = MiniTransformer(
    vocab_size=10000,  # Vocabulary size
    d_model=256,       # Embedding dimension
    n_heads=4,         # Attention heads
    n_layers=4,        # Transformer blocks
    max_seq_len=512,   # Max sequence length
)
print(f"Parameters: {model.count_parameters():,}")

# Forward pass: token IDs -> next-token logits
tokens = torch.randint(0, 10000, (2, 50))  # batch=2, seq_len=50
logits = model(tokens)                      # (2, 50, 10000)

# Autoregressive generation
prompt = torch.randint(0, 10000, (1, 5))
generated = model.generate(prompt, max_new_tokens=20, temperature=0.8, top_k=50)
```

### 6. Training Any Model

```python
import torch.nn as nn
from agentexplorr.classical_ml.deep_learning.training_loop import Trainer

# Works with ANY nn.Module
model = SimpleCNN(num_classes=10)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    loss_fn=criterion,
    device="cuda" if torch.cuda.is_available() else "cpu",
    grad_clip_value=1.0,  # Prevents exploding gradients
)

history = trainer.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=50,
    patience=5,           # Early stopping: stop after 5 epochs without improvement
    checkpoint_dir="./checkpoints",
)
```

### 7. Experiment Tracking with MLflow

```python
from agentexplorr.classical_ml.experiment_tracking import ExperimentTracker

tracker = ExperimentTracker(experiment_name="wine_classification")

with tracker.start_run(run_name="random_forest_baseline") as run_id:
    # Log hyperparameters
    tracker.log_params({"n_estimators": 100, "max_depth": 10, "model": "RandomForest"})

    # Train your model...
    clf.tune_hyperparameters(X_train, y_train)
    results = clf.evaluate(X_test, y_test)

    # Log metrics
    tracker.log_metrics({"accuracy": results["accuracy"], "f1_score": results["f1_score"]})

    # Log the model for deployment
    tracker.log_model(clf.pipeline, registered_model_name="wine_classifier")

# Compare all runs to find the best model
tracker.compare_runs(metric_key="accuracy", n_top=5)
```

---

## Core Concepts

### The Bias-Variance Tradeoff

The single most important concept in machine learning:

- **Bias** = error from overly simplistic assumptions. High bias -> underfitting.
- **Variance** = error from sensitivity to training data. High variance -> overfitting.
- **Total Error = Bias^2 + Variance + Irreducible Noise**

| Problem | Symptom | Solution |
|---------|---------|----------|
| High Bias (underfitting) | Train AND test error are high | More complex model, more features |
| High Variance (overfitting) | Train error low, test error high | Regularization, more data, simpler model |

Regularization techniques in this module:
- **Ridge (L2)**: Shrinks all coefficients toward zero -> reduces variance
- **Lasso (L1)**: Drives some coefficients to exactly zero -> feature selection
- **Dropout**: Randomly disables neurons during training -> ensemble effect
- **Early stopping**: Stops training before overfitting occurs

### Attention Mechanism (Transformers)

The key innovation behind GPT, BERT, Claude, and all modern LLMs:

```
For each token, compute: "How relevant is every other token to me?"

Query (Q):  "What am I looking for?"
Key (K):    "What information do I contain?"
Value (V):  "Here is my actual content."

Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
```

Why this matters:
- RNNs process tokens sequentially (slow, forgets long-range dependencies)
- Transformers process ALL tokens in parallel (fast, global context)

See `deep_learning/transformer.py` for the complete from-scratch implementation.

### The ML Pipeline Pattern

Always use sklearn Pipelines instead of manual preprocessing:

```
BAD (data leakage risk):
  scaler.fit(X_train)
  X_train_scaled = scaler.transform(X_train)
  X_test_scaled = scaler.transform(X_test)  # What if you forget this?
  model.fit(X_train_scaled, y_train)

GOOD (Pipeline handles everything):
  pipe = Pipeline([("scaler", StandardScaler()), ("model", RandomForest())])
  pipe.fit(X_train, y_train)      # Scaler AND model fitted correctly
  pipe.predict(X_test)            # Scaler transforms, model predicts
```

---

## Learning Resources

### Videos (Start Here)

| Topic | Resource | Duration |
|-------|----------|----------|
| Neural Networks Intro | [3Blue1Brown: But what is a neural network?](https://www.youtube.com/watch?v=aircAruvnKk) | 19 min |
| Gradient Descent | [3Blue1Brown: Gradient Descent](https://www.youtube.com/watch?v=IHZwWFHWa-w) | 21 min |
| Transformers | [3Blue1Brown: Attention in Transformers](https://www.youtube.com/watch?v=eMlx5fFNoYc) | 27 min |
| Build GPT | [Andrej Karpathy: Let's build GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY) | 120 min |
| Random Forests | [StatQuest: Random Forest](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ) | 10 min |
| Bias-Variance | [StatQuest: Bias-Variance Tradeoff](https://www.youtube.com/watch?v=EuBBz3bI-aA) | 7 min |
| Regularization | [StatQuest: Ridge, Lasso, ElasticNet](https://www.youtube.com/watch?v=Q81RR3yKn30) | 21 min |
| Cross-Validation | [StatQuest: Cross Validation](https://www.youtube.com/watch?v=fSytzGwwBVw) | 6 min |
| KMeans | [StatQuest: KMeans Clustering](https://www.youtube.com/watch?v=4b5d3muPQmA) | 9 min |
| sklearn Pipelines | [Data School: sklearn Pipeline Tutorial](https://www.youtube.com/watch?v=41yGHJEMoRQ) | 20 min |

### Papers (For Deep Understanding)

| Paper | Why It Matters |
|-------|---------------|
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017) | Introduced the Transformer -- the foundation of all modern LLMs |
| [Deep Residual Learning](https://arxiv.org/abs/1512.03385) (He et al., 2015) | Residual connections that enabled training 100+ layer networks |
| [Batch Normalization](https://arxiv.org/abs/1502.03167) (Ioffe & Szegedy, 2015) | Stabilized and accelerated deep network training |
| [Random Forests](https://link.springer.com/article/10.1023/A:1010933404324) (Breiman, 2001) | The ensemble method that works on everything |
| [ImageNet Classification with Deep CNNs](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) (Krizhevsky et al., 2012) | AlexNet -- started the deep learning revolution |

### Online Courses (Free)

| Course | Platform | Focus |
|--------|----------|-------|
| [CS231n: CNNs for Visual Recognition](https://cs231n.stanford.edu/) | Stanford | Computer Vision |
| [CS224n: NLP with Deep Learning](https://web.stanford.edu/class/cs224n/) | Stanford | NLP & Transformers |
| [Fast.ai Practical Deep Learning](https://course.fast.ai/) | fast.ai | Hands-on DL |
| [Machine Learning (Andrew Ng)](https://www.coursera.org/learn/machine-learning) | Coursera | ML Foundations |

### Documentation

| Resource | URL |
|----------|-----|
| scikit-learn User Guide | https://scikit-learn.org/stable/user_guide.html |
| PyTorch Tutorials | https://pytorch.org/tutorials/ |
| MLflow Documentation | https://mlflow.org/docs/latest/index.html |
| The Illustrated Transformer | https://jalammar.github.io/illustrated-transformer/ |
| The Annotated Transformer | https://nlp.seas.harvard.edu/annotated-transformer/ |

### Books

| Book | Author | Best For |
|------|--------|----------|
| *Hands-On ML with Scikit-Learn, Keras & TensorFlow* | Aurelien Geron | Practical ML |
| *Deep Learning* ([free online](https://www.deeplearningbook.org/)) | Goodfellow, Bengio, Courville | Theory |
| *Pattern Recognition and Machine Learning* | Christopher Bishop | Mathematical depth |

---

## Running the Demos

Each file includes a `__main__` block you can run directly:

```bash
# Classification
python -m agentexplorr.classical_ml.pipelines.classification

# Regression
python -m agentexplorr.classical_ml.pipelines.regression

# Clustering
python -m agentexplorr.classical_ml.pipelines.clustering

# CNN
python -m agentexplorr.classical_ml.deep_learning.cnn

# Transformer
python -m agentexplorr.classical_ml.deep_learning.transformer

# Training Loop
python -m agentexplorr.classical_ml.deep_learning.training_loop
```

---

## Interview Prep

This module covers the most commonly asked ML interview topics:

1. **"Explain the bias-variance tradeoff."** -> See `regression.py`
2. **"How does regularization work?"** -> See `regression.py` (Ridge/Lasso/ElasticNet)
3. **"Explain the attention mechanism."** -> See `transformer.py` (Q, K, V analogy)
4. **"What is a confusion matrix?"** -> See `classification.py`
5. **"How does cross-validation work?"** -> See `classification.py` and `regression.py`
6. **"Explain how a CNN works."** -> See `cnn.py` (convolution, pooling, activation)
7. **"What is the difference between supervised and unsupervised learning?"** -> Compare `classification.py` vs `clustering.py`
8. **"How do you prevent overfitting?"** -> See `training_loop.py` (early stopping, dropout, regularization)
9. **"Explain the Transformer architecture."** -> See `transformer.py` (the entire file!)
10. **"How do you track experiments in production?"** -> See `experiment_tracking.py` (MLflow)
