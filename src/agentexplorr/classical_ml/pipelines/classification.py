"""
Classification Pipeline
========================

A production-quality classification pipeline built on scikit-learn, designed to
teach you every step of the ML workflow — from raw data to evaluated predictions.

WHAT IS CLASSIFICATION?
    Classification is the task of predicting a DISCRETE label for an input.
    Examples:
      - Email → spam or not spam (binary classification)
      - Image → cat, dog, or bird (multi-class classification)
      - Wine → quality class 0, 1, or 2 (what we do here)

    The model learns a decision boundary that separates different classes in
    feature space. Different algorithms draw different kinds of boundaries:
      - Logistic Regression: linear boundaries (hyperplanes)
      - Decision Trees: axis-aligned rectangular regions
      - Random Forests: ensemble of trees → smoother, more robust boundaries
      - SVMs: maximum-margin boundaries, optionally non-linear via kernels

WHY RANDOM FOREST AS DEFAULT?
    Random Forest is the "Swiss Army knife" of classical ML:
      1. Works well out-of-the-box with minimal tuning
      2. Handles both numerical and categorical features
      3. Robust to outliers and noisy data
      4. Provides feature importance for free
      5. Rarely overfits (thanks to bagging + random feature selection)

    It works by training many decision trees on random subsets of data and
    features, then averaging their predictions. This "wisdom of crowds"
    approach dramatically reduces variance compared to a single tree.

THE PIPELINE ARCHITECTURE:
    ┌─────────────────┐     ┌──────────────┐     ┌──────────────────────┐
    │ ColumnTransformer│ ──→ │ StandardScaler│ ──→ │ RandomForestClassifier│
    │ (preprocessing)  │     │ (numeric cols)│     │ (or any classifier)   │
    │                  │     │ OneHotEncoder │     │                       │
    │                  │     │ (categ. cols) │     │                       │
    └─────────────────┘     └──────────────┘     └──────────────────────┘

    Why a Pipeline?
      - Prevents data leakage (scaler is fit ONLY on training data)
      - Reproducible (same transforms applied to train and test)
      - GridSearchCV works across ALL steps (tune scaler + model together)

DATASET: Wine Quality (sklearn.datasets.load_wine)
    - 178 samples, 13 features, 3 classes
    - Features: alcohol, malic acid, ash, magnesium, color intensity, etc.
    - Task: predict the cultivar (grape variety) from chemical analysis

LEARNING RESOURCES:
    - sklearn Classification Guide: https://scikit-learn.org/stable/supervised_learning.html
    - Random Forests Explained: https://scikit-learn.org/stable/modules/ensemble.html#forests-of-randomized-trees
    - GridSearchCV docs: https://scikit-learn.org/stable/modules/grid_search.html
    - Confusion Matrix Guide: https://scikit-learn.org/stable/modules/model_evaluation.html#confusion-matrix
    - VIDEO: "Random Forest Algorithm Clearly Explained" — https://www.youtube.com/watch?v=J4Wdy0Wc_xQ
    - VIDEO: "Confusion Matrix, Precision, Recall, F1" — https://www.youtube.com/watch?v=Kdsp6soqA7o
    - VIDEO: "sklearn Pipeline Tutorial" — https://www.youtube.com/watch?v=41yGHJEMoRQ
    - PAPER: Breiman (2001) "Random Forests" — https://link.springer.com/article/10.1023/A:1010933404324
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from agentexplorr.core import get_logger

logger = get_logger(__name__)


class ClassificationPipeline:
    """End-to-end classification pipeline with preprocessing, tuning & evaluation.

    This class wraps sklearn's Pipeline, ColumnTransformer, and GridSearchCV into
    a clean, reusable interface. It's designed to be educational — every method
    has detailed comments explaining what's happening and why.

    Attributes:
        pipeline: The sklearn Pipeline (preprocessing + model).
        grid_search: The GridSearchCV object (after tuning).
        results: Dictionary of evaluation metrics (after evaluation).
        feature_names: Names of input features.
        target_names: Names of target classes.

    Example:
        >>> clf = ClassificationPipeline()
        >>> X_train, X_test, y_train, y_test = clf.load_data()
        >>> clf.build_pipeline()
        >>> clf.tune_hyperparameters(X_train, y_train)
        >>> results = clf.evaluate(X_test, y_test)
        >>> print(results["accuracy"])
        0.9722...
    """

    def __init__(self, random_state: int = 42) -> None:
        """Initialize the classification pipeline.

        Args:
            random_state: Seed for reproducibility. Always set this!
                In ML, reproducibility is critical. If you can't reproduce your
                results, you can't debug them, and reviewers/colleagues can't
                verify them. The random_state parameter controls:
                  - Train/test split randomness
                  - Random Forest's bootstrap sampling
                  - Random Forest's feature selection at each split
        """
        self.random_state: int = random_state
        self.pipeline: Pipeline | None = None
        self.grid_search: GridSearchCV | None = None
        self.results: dict[str, Any] = {}
        self.feature_names: list[str] = []
        self.target_names: list[str] = []

        logger.info(
            "classification_pipeline_initialized",
            random_state=random_state,
        )

    def load_data(
        self,
        test_size: float = 0.2,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load the Wine Quality dataset and split into train/test sets.

        WHY TRAIN/TEST SPLIT?
            We NEVER evaluate a model on the same data it was trained on.
            That would be like giving a student the exact exam questions to
            study — of course they'll score well, but it doesn't measure
            real understanding.

            The test set simulates "unseen data" — data the model has never
            seen during training. This gives us an honest estimate of how
            the model will perform in production.

        WHAT IS test_size=0.2?
            We hold out 20% of data for testing, use 80% for training.
            Common splits are 80/20 or 70/30. With very large datasets
            (>100k samples), even 90/10 is fine.

        Args:
            test_size: Fraction of data to reserve for testing (0.0 to 1.0).

        Returns:
            Tuple of (X_train, X_test, y_train, y_test) — the classic ML split.
        """
        # load_wine returns a Bunch object with .data, .target, .feature_names, etc.
        wine = load_wine()

        # Store metadata for later use in evaluation and visualization
        self.feature_names = list(wine.feature_names)
        self.target_names = list(wine.target_names)

        # stratify=y ensures each class is proportionally represented in both
        # train and test sets. Without this, you might get unlucky and put all
        # samples of one class into the test set, making evaluation misleading.
        X_train, X_test, y_train, y_test = train_test_split(
            wine.data,
            wine.target,
            test_size=test_size,
            random_state=self.random_state,
            stratify=wine.target,  # Preserve class distribution in both splits
        )

        logger.info(
            "data_loaded",
            dataset="wine_quality",
            n_samples=len(wine.data),
            n_features=len(self.feature_names),
            n_classes=len(self.target_names),
            train_size=len(X_train),
            test_size=len(X_test),
        )

        return X_train, X_test, y_train, y_test

    def build_pipeline(
        self,
        numeric_features: list[int] | None = None,
        categorical_features: list[int] | None = None,
        n_estimators: int = 100,
        max_depth: int | None = None,
    ) -> Pipeline:
        """Build the sklearn Pipeline with preprocessing and classifier.

        THE PIPELINE HAS TWO STAGES:

        Stage 1: Preprocessing (ColumnTransformer)
            - Numeric features → StandardScaler
                StandardScaler transforms each feature to have mean=0, std=1:
                    x_scaled = (x - mean) / std
                WHY? Many algorithms (SVM, logistic regression, KNN) are sensitive
                to feature scale. If "alcohol" ranges 11-15 and "magnesium" ranges
                70-160, the model might think magnesium is more important just
                because its values are larger.

            - Categorical features → OneHotEncoder
                Converts categories like ["red", "white", "rosé"] into binary
                columns: [1,0,0], [0,1,0], [0,0,1].
                WHY? ML models work with numbers. You can't feed "red" into a
                mathematical formula. One-hot encoding preserves the fact that
                categories are NOT ordered (red is not "less than" white).

            NOTE: Wine dataset is all-numeric, so we default to scaling everything.
            But the pipeline is built to handle mixed data types — you'll encounter
            this in real-world datasets constantly.

        Stage 2: Model (RandomForestClassifier)
            An ensemble of decision trees, each trained on a random bootstrap
            sample with a random subset of features. Predictions are made by
            majority vote across all trees.

        Args:
            numeric_features: Column indices for numeric features (default: all).
            categorical_features: Column indices for categorical features (default: none).
            n_estimators: Number of trees in the forest.
                More trees = better performance but slower training.
                100 is a good default; 500+ for production.
            max_depth: Maximum depth of each tree (None = unlimited).
                Deeper trees = more complex, risk overfitting.
                Shallower trees = simpler, risk underfitting.

        Returns:
            The constructed sklearn Pipeline.
        """
        # Default: treat all features as numeric (Wine dataset is all-numeric)
        if numeric_features is None:
            numeric_features = list(range(13))  # Wine has 13 features
        if categorical_features is None:
            categorical_features = []

        # --- Build the preprocessing step ---
        # ColumnTransformer applies different transformations to different columns.
        # This is essential for real-world data where you have a mix of numeric
        # and categorical features.
        transformers: list[tuple[str, Any, list[int]]] = []

        if numeric_features:
            transformers.append(("num", StandardScaler(), numeric_features))

        if categorical_features:
            # handle_unknown="ignore" prevents errors if test data has unseen categories
            transformers.append(
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
            )

        preprocessor = ColumnTransformer(
            transformers=transformers,
            # "passthrough" keeps any columns not listed above unchanged.
            # "drop" would remove them. We want to keep everything.
            remainder="passthrough" if not numeric_features else "drop",
        )

        # --- Build the full pipeline ---
        # Pipeline chains preprocessing → model into a single object.
        # When you call pipeline.fit(X, y), it calls:
        #   1. preprocessor.fit_transform(X)  — learn scaling params from training data
        #   2. classifier.fit(X_scaled, y)    — train the model on scaled data
        # When you call pipeline.predict(X_new), it calls:
        #   1. preprocessor.transform(X_new)  — apply SAME scaling (no re-fitting!)
        #   2. classifier.predict(X_scaled)   — make predictions
        self.pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("classifier", RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=self.random_state,
                    # n_jobs=-1 uses all CPU cores for parallel tree training.
                    # Random Forest is "embarrassingly parallel" — each tree
                    # is independent, so we can train them simultaneously.
                    n_jobs=-1,
                )),
            ]
        )

        logger.info(
            "pipeline_built",
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_numeric=len(numeric_features),
            n_categorical=len(categorical_features),
        )

        return self.pipeline

    def tune_hyperparameters(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        param_grid: dict[str, list[Any]] | None = None,
        cv: int = 5,
        scoring: str = "f1_weighted",
    ) -> dict[str, Any]:
        """Find the best hyperparameters using GridSearchCV.

        WHAT IS HYPERPARAMETER TUNING?
            Model parameters (like tree split thresholds) are LEARNED from data.
            Hyperparameters (like n_estimators, max_depth) are SET BY YOU.

            How do you know which values are best? You try many combinations
            and pick the one that performs best on held-out validation data.

        WHAT IS GRID SEARCH?
            GridSearchCV tries EVERY combination of hyperparameters you specify.
            "Grid" because it forms a grid of all possible combinations.
            "CV" because it uses Cross-Validation to evaluate each combination.

            Example: if you specify n_estimators=[50,100,200] and max_depth=[5,10,None],
            GridSearch tries all 3x3=9 combinations.

        WHAT IS CROSS-VALIDATION (cv=5)?
            Instead of a single train/validation split, we split training data
            into 5 folds. We train on 4 folds and validate on the 5th, rotating
            the validation fold 5 times. The final score is the average.

            This gives a much more reliable estimate than a single split, at the
            cost of 5x more training time.

            ┌──────┬──────┬──────┬──────┬──────┐
            │ Val  │Train │Train │Train │Train │  Fold 1
            │Train │ Val  │Train │Train │Train │  Fold 2
            │Train │Train │ Val  │Train │Train │  Fold 3
            │Train │Train │Train │ Val  │Train │  Fold 4
            │Train │Train │Train │Train │ Val  │  Fold 5
            └──────┴──────┴──────┴──────┴──────┘

        WHY f1_weighted SCORING?
            Accuracy can be misleading with imbalanced classes. If 95% of emails
            are not spam, a model that ALWAYS predicts "not spam" gets 95% accuracy.
            F1-score (harmonic mean of precision and recall) penalizes this.
            "weighted" means we weight each class's F1 by its support (# samples).

        Args:
            X_train: Training feature matrix.
            y_train: Training labels.
            param_grid: Hyperparameter search space. Keys use sklearn's double-
                underscore notation: "classifier__n_estimators" means "the
                n_estimators parameter of the step named 'classifier'".
            cv: Number of cross-validation folds.
            scoring: Metric to optimize. Common options:
                "accuracy", "f1_weighted", "precision_weighted", "recall_weighted"

        Returns:
            Dictionary with best parameters and best CV score.
        """
        if self.pipeline is None:
            self.build_pipeline()

        # Default parameter grid — a reasonable search space for Random Forest
        if param_grid is None:
            param_grid = {
                # Double underscore syntax: step_name__parameter_name
                # "classifier__n_estimators" = n_estimators of the "classifier" step
                "classifier__n_estimators": [50, 100, 200],
                "classifier__max_depth": [5, 10, 20, None],
                "classifier__min_samples_split": [2, 5, 10],
                # min_samples_split: minimum samples required to split a node.
                # Higher values → simpler trees → less overfitting
            }

        # GridSearchCV wraps the pipeline and tries all param combinations
        self.grid_search = GridSearchCV(
            estimator=self.pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            # Return the mean and std of scores across folds
            return_train_score=True,
            # n_jobs=-1 parallelizes across folds AND param combinations
            n_jobs=-1,
            # verbose=1 prints progress (useful for large grids)
            verbose=1,
        )

        logger.info(
            "tuning_started",
            n_combinations=np.prod([len(v) for v in param_grid.values()]),
            cv_folds=cv,
            scoring=scoring,
        )

        # This is where the heavy lifting happens — fitting all combinations
        self.grid_search.fit(X_train, y_train)

        # After fitting, grid_search.best_estimator_ is the pipeline re-fitted
        # on ALL training data with the best hyperparameters.
        # grid_search.best_params_ is the winning combination.
        # grid_search.best_score_ is the average CV score of the winning combo.
        best_params = self.grid_search.best_params_
        best_score = self.grid_search.best_score_

        logger.info(
            "tuning_completed",
            best_score=round(best_score, 4),
            best_params=best_params,
        )

        return {
            "best_params": best_params,
            "best_cv_score": best_score,
        }

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> ClassificationPipeline:
        """Train the pipeline without hyperparameter tuning.

        Use this when you already know your hyperparameters, or when you want
        a quick baseline before tuning.

        Args:
            X_train: Training feature matrix.
            y_train: Training labels.

        Returns:
            self (for method chaining: pipeline.fit(X, y).evaluate(X_test, y_test))
        """
        if self.pipeline is None:
            self.build_pipeline()

        assert self.pipeline is not None  # For type checker
        self.pipeline.fit(X_train, y_train)

        logger.info("pipeline_fitted", n_samples=len(X_train))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions using the trained pipeline.

        If hyperparameter tuning was performed, uses the best estimator.
        Otherwise, uses the base pipeline.

        Args:
            X: Feature matrix to predict on.

        Returns:
            Array of predicted class labels.
        """
        # Prefer the tuned model if available
        model = (
            self.grid_search.best_estimator_
            if self.grid_search is not None
            else self.pipeline
        )
        if model is None:
            raise RuntimeError("Pipeline not fitted. Call fit() or tune_hyperparameters() first.")

        predictions: np.ndarray = model.predict(X)
        return predictions

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> dict[str, Any]:
        """Evaluate the model on test data with comprehensive metrics.

        CLASSIFICATION METRICS EXPLAINED:

        1. ACCURACY = correct predictions / total predictions
           Simple but misleading with imbalanced classes.

        2. PRECISION = true positives / (true positives + false positives)
           "Of all items predicted as positive, how many actually are?"
           High precision = few false alarms.
           Important when false positives are costly (e.g., spam filter
           blocking important emails).

        3. RECALL (Sensitivity) = true positives / (true positives + false negatives)
           "Of all actual positives, how many did we catch?"
           High recall = few missed positives.
           Important when false negatives are costly (e.g., missing a cancer
           diagnosis).

        4. F1-SCORE = 2 * (precision * recall) / (precision + recall)
           Harmonic mean of precision and recall. Balances both.
           F1 = 1.0 is perfect. F1 = 0.0 is worst.

        5. CONFUSION MATRIX
           An N x N matrix (N = number of classes) showing:
             - Rows = actual classes
             - Columns = predicted classes
             - Diagonal = correct predictions
             - Off-diagonal = errors

           Example (3 classes):
                         Predicted
                       0    1    2
           Actual  0 [45    2    0]   ← Class 0: 45 correct, 2 confused with 1
                   1 [ 1   50    3]   ← Class 1: 50 correct, 1→0, 3→2
                   2 [ 0    1   76]   ← Class 2: 76 correct, 1 confused with 1

        Args:
            X_test: Test feature matrix (NEVER used during training).
            y_test: True test labels.

        Returns:
            Dictionary containing all evaluation metrics.
        """
        y_pred = self.predict(X_test)

        # Calculate each metric
        accuracy = accuracy_score(y_test, y_pred)

        # For multi-class: average="weighted" weights by class support (# samples)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        # Confusion matrix: rows = actual, columns = predicted
        conf_matrix = confusion_matrix(y_test, y_pred)

        # classification_report gives a nice per-class breakdown
        report = classification_report(
            y_test,
            y_pred,
            target_names=self.target_names if self.target_names else None,
            zero_division=0,
        )

        self.results = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": conf_matrix,
            "classification_report": report,
        }

        logger.info(
            "evaluation_completed",
            accuracy=round(accuracy, 4),
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1_score=round(f1, 4),
        )

        # Print the full classification report for readability
        print("\n" + "=" * 60)
        print("CLASSIFICATION RESULTS")
        print("=" * 60)
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print("\nPer-Class Report:")
        print(report)
        print("Confusion Matrix:")
        print(conf_matrix)
        print("=" * 60)

        return self.results

    def get_feature_importances(self) -> dict[str, float]:
        """Extract feature importances from the Random Forest.

        WHAT ARE FEATURE IMPORTANCES?
            Random Forest measures how much each feature contributes to reducing
            impurity (Gini or entropy) across all trees. Features that appear
            near the top of many trees and produce large impurity reductions
            are deemed "important".

            This is a powerful tool for:
              - Understanding which features drive predictions
              - Feature selection (drop unimportant features to simplify the model)
              - Domain insight (which chemical properties distinguish wine types?)

            CAUTION: Feature importance can be misleading for correlated features.
            If two features are highly correlated, the importance gets "split"
            between them, making both look less important than they are.
            Use permutation importance for a more robust alternative:
            https://scikit-learn.org/stable/modules/permutation_importance.html

        Returns:
            Dictionary mapping feature names to importance scores (sum to 1.0).

        Raises:
            RuntimeError: If the pipeline hasn't been fitted yet.
        """
        # Get the fitted model (from tuning or direct fit)
        model = (
            self.grid_search.best_estimator_
            if self.grid_search is not None
            else self.pipeline
        )
        if model is None:
            raise RuntimeError("Pipeline not fitted. Call fit() or tune_hyperparameters() first.")

        # Access the classifier step within the pipeline
        # model.named_steps["classifier"] gives us the RandomForestClassifier
        classifier = model.named_steps["classifier"]
        importances = classifier.feature_importances_

        # Pair feature names with their importance scores
        feature_importance_dict: dict[str, float] = {}
        for name, importance in zip(self.feature_names, importances):
            feature_importance_dict[name] = float(round(importance, 4))

        # Sort by importance (most important first)
        feature_importance_dict = dict(
            sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
        )

        logger.info(
            "feature_importances_extracted",
            top_3=list(feature_importance_dict.items())[:3],
        )

        return feature_importance_dict


# ---------------------------------------------------------------------------
# Quick demo — run this file directly to see the full pipeline in action
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("CLASSIFICATION PIPELINE DEMO")
    print("Dataset: Wine Quality (sklearn)")
    print("Model: Random Forest with GridSearchCV")
    print("=" * 60)

    # 1. Initialize
    clf = ClassificationPipeline(random_state=42)

    # 2. Load data
    X_train, X_test, y_train, y_test = clf.load_data(test_size=0.2)

    # 3. Build pipeline
    clf.build_pipeline()

    # 4. Tune hyperparameters (this takes a moment)
    print("\nTuning hyperparameters with GridSearchCV...")
    tuning_results = clf.tune_hyperparameters(X_train, y_train, cv=5)
    print(f"Best CV Score: {tuning_results['best_cv_score']:.4f}")
    print(f"Best Params: {tuning_results['best_params']}")

    # 5. Evaluate on test set
    results = clf.evaluate(X_test, y_test)

    # 6. Feature importances
    print("\nFeature Importances (top 5):")
    importances = clf.get_feature_importances()
    for name, score in list(importances.items())[:5]:
        print(f"  {name}: {score:.4f}")
