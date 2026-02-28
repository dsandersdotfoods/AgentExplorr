"""
Classical ML Pipelines
=======================

This sub-package contains production-style sklearn pipelines for the three
fundamental ML paradigms:

    1. **Classification** — Predicting discrete labels (spam/not-spam, disease/healthy)
    2. **Regression** — Predicting continuous values (house price, temperature)
    3. **Clustering** — Discovering groups in unlabeled data (customer segments)

WHY PIPELINES?
    A raw ML workflow often looks like:
        scaler.fit_transform(X_train)  →  model.fit(X_scaled)  →  model.predict(X_test)

    The problem? You'll forget to transform X_test, or you'll fit the scaler on
    test data (data leakage!), or you'll lose track of which preprocessing steps
    you applied. sklearn Pipelines solve all of this:

        pipe = Pipeline([("scaler", StandardScaler()), ("model", RandomForest())])
        pipe.fit(X_train, y_train)    # Scaler + model fitted together
        pipe.predict(X_test)          # Scaler transforms, then model predicts

    Pipelines are also required for GridSearchCV — you can tune hyperparameters
    across ALL steps (including preprocessing) in one call.

LEARNING RESOURCES:
    - sklearn Pipelines: https://scikit-learn.org/stable/modules/compose.html
    - ColumnTransformer: https://scikit-learn.org/stable/modules/compose.html#columntransformer-for-heterogeneous-data
    - VIDEO: "sklearn Pipelines" — https://www.youtube.com/watch?v=41yGHJEMoRQ
    - VIDEO: "Data Leakage in ML" — https://www.youtube.com/watch?v=h6p0GGpKLms
"""

from __future__ import annotations

from agentexplorr.classical_ml.pipelines.classification import ClassificationPipeline
from agentexplorr.classical_ml.pipelines.clustering import ClusteringPipeline
from agentexplorr.classical_ml.pipelines.regression import RegressionPipeline

__all__ = [
    "ClassificationPipeline",
    "RegressionPipeline",
    "ClusteringPipeline",
]
