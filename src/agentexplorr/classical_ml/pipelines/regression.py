"""
Regression Pipeline
====================

A production-quality regression pipeline demonstrating regularization techniques,
the bias-variance tradeoff, and proper model evaluation for continuous targets.

WHAT IS REGRESSION?
    Regression predicts a CONTINUOUS value (a number on a scale) rather than a
    discrete class. Examples:
      - House price prediction ($150k, $320k, $1.2M)
      - Temperature forecasting (72.3F, 68.1F)
      - Stock price prediction (not recommended, but popular)
      - Salary estimation based on experience

    Mathematically, we're learning a function f(X) ≈ y where:
      - X is a matrix of features (inputs)
      - y is a vector of continuous target values (outputs)

THE BIAS-VARIANCE TRADEOFF:
    This is THE most important concept in machine learning. Every ML engineer
    must understand it deeply.

    BIAS = how far off the model's average prediction is from the true value.
        High bias → model is too simple → UNDERFITTING
        Example: fitting a straight line to curved data

    VARIANCE = how much predictions change with different training data.
        High variance → model is too complex → OVERFITTING
        Example: a polynomial that perfectly fits training data but wiggles
        wildly between training points

    Total Error = Bias² + Variance + Irreducible Noise

    ┌─────────────────────────────────────────┐
    │  Error                                   │
    │  ▲                                       │
    │  │ ╲                           ╱         │
    │  │  ╲   Total Error           ╱          │
    │  │   ╲                      ╱            │
    │  │    ╲                   ╱               │
    │  │     ╲               ╱                 │
    │  │      ╲   ╱╲       ╱                   │
    │  │       ╲╱    ╲   ╱   Variance          │
    │  │  Bias  ╲     ╲╱                       │
    │  │         ╲      Sweet spot             │
    │  │──────────╲────────────────►           │
    │  │    Simple   Model Complexity   Complex │
    │  └─────────────────────────────────────┘

    REGULARIZATION is how we control this tradeoff:
      - Ridge (L2): penalizes large coefficients → reduces variance
      - Lasso (L1): drives some coefficients to exactly zero → feature selection
      - ElasticNet: combines Ridge + Lasso

REGULARIZATION DEEP DIVE:

    Ordinary Least Squares (OLS) minimizes:
        Loss = Σ(y - ŷ)²

    Ridge Regression (L2) minimizes:
        Loss = Σ(y - ŷ)² + α * Σ(w²)
        The α * Σ(w²) term penalizes large weights, pushing them toward zero
        (but never exactly zero). This prevents any single feature from having
        outsized influence, reducing overfitting.

    Lasso Regression (L1) minimizes:
        Loss = Σ(y - ŷ)² + α * Σ|w|
        The absolute value penalty can drive weights to EXACTLY zero, effectively
        removing features. This performs automatic feature selection.

    ElasticNet combines both:
        Loss = Σ(y - ŷ)² + α * (ρ * Σ|w| + (1-ρ)/2 * Σ(w²))
        ρ (l1_ratio) controls the mix: ρ=1 is pure Lasso, ρ=0 is pure Ridge.

DATASET: California Housing (sklearn.datasets.fetch_california_housing)
    - 20,640 samples, 8 features
    - Target: median house value (in $100,000s)
    - Features: median income, house age, avg rooms, latitude, longitude, etc.

LEARNING RESOURCES:
    - sklearn Regression Guide: https://scikit-learn.org/stable/supervised_learning.html#supervised-learning
    - Ridge Regression: https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression
    - Lasso Regression: https://scikit-learn.org/stable/modules/linear_model.html#lasso
    - ElasticNet: https://scikit-learn.org/stable/modules/linear_model.html#elastic-net
    - Cross-validation: https://scikit-learn.org/stable/modules/cross_validation.html
    - VIDEO: "Bias-Variance Tradeoff" (StatQuest) — https://www.youtube.com/watch?v=EuBBz3bI-aA
    - VIDEO: "Regularization (Ridge, Lasso, ElasticNet)" — https://www.youtube.com/watch?v=Q81RR3yKn30
    - VIDEO: "Cross Validation Clearly Explained" — https://www.youtube.com/watch?v=fSytzGwwBVw
    - PAPER: Tibshirani (1996) "Regression Shrinkage and Selection via the Lasso"
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from agentexplorr.core import get_logger

logger = get_logger(__name__)

# Type alias for supported regression model names
ModelName = Literal["ridge", "lasso", "elasticnet"]


class RegressionPipeline:
    """End-to-end regression pipeline with regularization and cross-validation.

    This pipeline supports three regularized linear models (Ridge, Lasso,
    ElasticNet) and includes proper preprocessing, cross-validation, and
    comprehensive evaluation metrics.

    Attributes:
        model_name: Which regularization model is used ("ridge", "lasso", "elasticnet").
        pipeline: The sklearn Pipeline (scaler + model).
        cv_scores: Cross-validation scores from the last fit.
        results: Dictionary of evaluation metrics (after evaluation).

    Example:
        >>> reg = RegressionPipeline(model_name="ridge", alpha=1.0)
        >>> X_train, X_test, y_train, y_test = reg.load_data()
        >>> reg.build_pipeline()
        >>> reg.fit_with_cross_validation(X_train, y_train, cv=5)
        >>> results = reg.evaluate(X_test, y_test)
        >>> print(f"R² = {results['r2']:.4f}")
    """

    # Map of model names to their sklearn classes.
    # This pattern (strategy pattern) lets us swap models without changing code.
    MODEL_REGISTRY: dict[str, type[Ridge | Lasso | ElasticNet]] = {
        "ridge": Ridge,
        "lasso": Lasso,
        "elasticnet": ElasticNet,
    }

    def __init__(
        self,
        model_name: ModelName = "ridge",
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        random_state: int = 42,
    ) -> None:
        """Initialize the regression pipeline.

        Args:
            model_name: Which regularization model to use.
                "ridge": L2 regularization — good default, handles multicollinearity.
                "lasso": L1 regularization — performs feature selection.
                "elasticnet": L1 + L2 — best of both worlds.
            alpha: Regularization strength. Higher = stronger regularization.
                α = 0: no regularization (equivalent to OLS)
                α = 0.01: very light regularization
                α = 1.0: moderate regularization (good default)
                α = 100: very strong regularization (most coefficients → 0)
            l1_ratio: Only for ElasticNet. Mix of L1 vs L2:
                0.0 = pure Ridge (L2)
                0.5 = equal mix (default)
                1.0 = pure Lasso (L1)
            random_state: Seed for reproducibility.
        """
        if model_name not in self.MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Choose from: {list(self.MODEL_REGISTRY.keys())}"
            )

        self.model_name: ModelName = model_name
        self.alpha: float = alpha
        self.l1_ratio: float = l1_ratio
        self.random_state: int = random_state
        self.pipeline: Pipeline | None = None
        self.cv_scores: np.ndarray | None = None
        self.results: dict[str, Any] = {}
        self.feature_names: list[str] = []

        logger.info(
            "regression_pipeline_initialized",
            model=model_name,
            alpha=alpha,
            l1_ratio=l1_ratio if model_name == "elasticnet" else "N/A",
        )

    def load_data(
        self,
        test_size: float = 0.2,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load the California Housing dataset and split into train/test.

        ABOUT THE DATASET:
            The California Housing dataset contains 20,640 instances from the
            1990 U.S. Census. Each instance represents a census block group
            (a geographical unit of ~600-3000 people).

            Features:
              - MedInc: median income in block group (in $10,000s)
              - HouseAge: median house age in block group
              - AveRooms: average number of rooms per household
              - AveBedrms: average number of bedrooms per household
              - Population: block group population
              - AveOccup: average number of household members
              - Latitude: block group latitude
              - Longitude: block group longitude

            Target: median house value (in $100,000s), capped at $5.00001

            IMPORTANT: This dataset has some quirks:
              - Target is capped at $500,001 (truncation bias)
              - Location features (lat/long) encode non-linear spatial patterns
              - Features have very different scales → StandardScaler is essential

        Args:
            test_size: Fraction of data for testing.

        Returns:
            Tuple of (X_train, X_test, y_train, y_test).
        """
        housing = fetch_california_housing()
        self.feature_names = list(housing.feature_names)

        X_train, X_test, y_train, y_test = train_test_split(
            housing.data,
            housing.target,
            test_size=test_size,
            random_state=self.random_state,
        )

        logger.info(
            "data_loaded",
            dataset="california_housing",
            n_samples=len(housing.data),
            n_features=len(self.feature_names),
            train_size=len(X_train),
            test_size=len(X_test),
            target_mean=round(float(np.mean(housing.target)), 4),
            target_std=round(float(np.std(housing.target)), 4),
        )

        return X_train, X_test, y_train, y_test

    def build_pipeline(self) -> Pipeline:
        """Build the preprocessing + regression pipeline.

        WHY STANDARDSCALER FOR REGRESSION?
            Regularization penalizes coefficient magnitudes. If feature scales
            differ wildly (income in $10Ks vs. rooms count in single digits),
            the penalty affects them unequally. StandardScaler ensures all
            features are on the same scale, so regularization treats them fairly.

            Without scaling: the coefficient for "population" (large numbers)
            would be tiny and barely penalized, while "rooms" (small numbers)
            would have a large coefficient and be heavily penalized. This creates
            an unfair bias toward features with large scales.

        Returns:
            The constructed sklearn Pipeline.
        """
        # Build the model with appropriate kwargs
        model_class = self.MODEL_REGISTRY[self.model_name]

        model_kwargs: dict[str, Any] = {"alpha": self.alpha}

        # ElasticNet has an extra parameter: l1_ratio
        if self.model_name == "elasticnet":
            model_kwargs["l1_ratio"] = self.l1_ratio

        # Ridge and ElasticNet don't have random_state; Lasso does for
        # coordinate descent solver randomness
        if self.model_name in ("lasso", "elasticnet"):
            model_kwargs["random_state"] = self.random_state
            # max_iter: coordinate descent may need more iterations for convergence
            model_kwargs["max_iter"] = 10000

        model = model_class(**model_kwargs)

        # Pipeline: scale features THEN apply regularized regression
        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("regressor", model),
            ]
        )

        logger.info(
            "pipeline_built",
            model=self.model_name,
            alpha=self.alpha,
        )

        return self.pipeline

    def fit_with_cross_validation(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        cv: int = 5,
    ) -> dict[str, float]:
        """Fit the pipeline with cross-validation to estimate generalization.

        CROSS-VALIDATION FOR REGRESSION:
            We use negative MSE as the scoring metric (sklearn convention: higher
            is better, so MSE is negated). After cross-validation, we negate
            back to get positive MSE values.

            Cross-validation answers: "How well will this model generalize to
            unseen data?" It's a more reliable estimate than a single
            train/test split, especially with smaller datasets.

        WHY FIT ON FULL TRAINING DATA AFTER CV?
            Cross-validation is for EVALUATION, not for training the final model.
            After we know the model generalizes well, we re-fit on ALL training
            data to get the most information into the final model.

        Args:
            X_train: Training feature matrix.
            y_train: Training target values.
            cv: Number of cross-validation folds.

        Returns:
            Dictionary with mean and std of cross-validation R² scores.
        """
        if self.pipeline is None:
            self.build_pipeline()

        assert self.pipeline is not None  # For type checker

        # Cross-validate using R² (coefficient of determination)
        # R² = 1 - (SS_res / SS_tot)
        #   SS_res = sum of squared residuals (prediction errors)
        #   SS_tot = total sum of squares (variance of y)
        # R² = 1.0 means perfect prediction
        # R² = 0.0 means the model predicts the mean of y (no better than baseline)
        # R² < 0.0 means worse than predicting the mean (very bad!)
        self.cv_scores = cross_val_score(
            self.pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring="r2",
            # n_jobs=-1 parallelizes across folds
            n_jobs=-1,
        )

        cv_mean = float(np.mean(self.cv_scores))
        cv_std = float(np.std(self.cv_scores))

        logger.info(
            "cross_validation_completed",
            cv_folds=cv,
            r2_mean=round(cv_mean, 4),
            r2_std=round(cv_std, 4),
            r2_scores=[round(s, 4) for s in self.cv_scores],
        )

        # After evaluating via CV, fit on the FULL training set
        # (CV only used subsets for training in each fold)
        self.pipeline.fit(X_train, y_train)

        logger.info("pipeline_fitted_on_full_training_data", n_samples=len(X_train))

        return {
            "cv_r2_mean": cv_mean,
            "cv_r2_std": cv_std,
        }

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> RegressionPipeline:
        """Fit the pipeline directly without cross-validation.

        Args:
            X_train: Training feature matrix.
            y_train: Training target values.

        Returns:
            self (for method chaining).
        """
        if self.pipeline is None:
            self.build_pipeline()

        assert self.pipeline is not None
        self.pipeline.fit(X_train, y_train)

        logger.info("pipeline_fitted", n_samples=len(X_train))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions using the trained pipeline.

        Args:
            X: Feature matrix to predict on.

        Returns:
            Array of predicted continuous values.
        """
        if self.pipeline is None:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")

        predictions: np.ndarray = self.pipeline.predict(X)
        return predictions

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> dict[str, Any]:
        """Evaluate the model with comprehensive regression metrics.

        REGRESSION METRICS EXPLAINED:

        1. MSE (Mean Squared Error) = mean((y - ŷ)²)
           Average of squared differences between predictions and true values.
           Penalizes large errors more than small ones (quadratic penalty).
           Units are squared (e.g., $² for price), which is hard to interpret.

        2. RMSE (Root Mean Squared Error) = sqrt(MSE)
           Square root of MSE, back in original units (e.g., $).
           More interpretable than MSE. The most commonly used regression metric.
           "On average, predictions are off by $X."

        3. MAE (Mean Absolute Error) = mean(|y - ŷ|)
           Average of absolute differences. Less sensitive to outliers than MSE.
           "The typical prediction error is $X."
           Use MAE when outliers shouldn't dominate the evaluation.

        4. R² (Coefficient of Determination) = 1 - (SS_res / SS_tot)
           How much variance in y the model explains.
           R² = 1.0: perfect prediction (explains all variance)
           R² = 0.0: model is as good as predicting the mean
           R² < 0.0: model is WORSE than predicting the mean

           Think of R² as "percentage of variance explained":
             R² = 0.75 means the model explains 75% of the variance in y.

        Args:
            X_test: Test feature matrix.
            y_test: True test target values.

        Returns:
            Dictionary of evaluation metrics.
        """
        y_pred = self.predict(X_test)

        # Calculate all metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = float(np.sqrt(mse))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        self.results = {
            "mse": float(mse),
            "rmse": rmse,
            "mae": float(mae),
            "r2": float(r2),
        }

        logger.info(
            "evaluation_completed",
            mse=round(mse, 4),
            rmse=round(rmse, 4),
            mae=round(mae, 4),
            r2=round(r2, 4),
        )

        # Pretty-print results
        print("\n" + "=" * 60)
        print(f"REGRESSION RESULTS ({self.model_name.upper()}, alpha={self.alpha})")
        print("=" * 60)
        print(f"MSE  (Mean Squared Error):    {mse:.4f}")
        print(f"RMSE (Root Mean Squared Err): {rmse:.4f}")
        print(f"MAE  (Mean Absolute Error):   {mae:.4f}")
        print(f"R²   (Coef. Determination):   {r2:.4f}")
        print()
        print(f"Interpretation: The model explains {r2*100:.1f}% of the variance")
        print(f"in housing prices. Average prediction error is ${rmse * 100_000:,.0f}")
        print(f"(target is in $100K units, so RMSE={rmse:.4f} ≈ ${rmse * 100_000:,.0f}).")
        print("=" * 60)

        return self.results

    def get_coefficients(self) -> dict[str, float]:
        """Extract model coefficients (feature weights).

        INTERPRETING COEFFICIENTS:
            For linear models, each coefficient tells you:
              "For a 1-unit increase in this feature (after scaling),
               the predicted target changes by this much."

            With StandardScaler, a "1-unit increase" means a 1-standard-deviation
            increase in the original feature. This makes coefficients comparable
            across features with different scales.

            LASSO SPECIAL: Lasso can drive coefficients to exactly 0.0,
            effectively removing those features. This is automatic feature
            selection! Features with coefficient = 0 are deemed unimportant
            by the model.

        Returns:
            Dictionary mapping feature names to their coefficients.

        Raises:
            RuntimeError: If the pipeline hasn't been fitted.
        """
        if self.pipeline is None:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")

        # Access the regressor step to get coefficients
        regressor = self.pipeline.named_steps["regressor"]
        coefficients = regressor.coef_

        coef_dict: dict[str, float] = {}
        for name, coef in zip(self.feature_names, coefficients):
            coef_dict[name] = float(round(coef, 6))

        # Sort by absolute value (most influential first)
        coef_dict = dict(
            sorted(coef_dict.items(), key=lambda x: abs(x[1]), reverse=True)
        )

        logger.info(
            "coefficients_extracted",
            model=self.model_name,
            n_nonzero=sum(1 for c in coef_dict.values() if abs(c) > 1e-10),
            n_total=len(coef_dict),
        )

        return coef_dict

    def compare_alphas(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        alphas: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        """Compare model performance across different alpha (regularization strength) values.

        UNDERSTANDING THE ALPHA PARAMETER:
            Alpha controls the regularization-fit tradeoff:

            Low alpha (e.g., 0.001):
              → Weak regularization → model fits training data closely
              → Risk of overfitting (high variance)

            High alpha (e.g., 100):
              → Strong regularization → simple model, small coefficients
              → Risk of underfitting (high bias)

            The "sweet spot" alpha balances bias and variance, giving the best
            generalization to unseen data. This method helps you find it.

        Args:
            X_train: Training features.
            y_train: Training targets.
            X_test: Test features.
            y_test: True test targets.
            alphas: List of alpha values to try. Default covers a wide range.

        Returns:
            List of dicts with alpha, train_r2, test_r2, n_nonzero_coefs for each alpha.
        """
        if alphas is None:
            # Logarithmic scale: covers 5 orders of magnitude
            alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

        comparison_results: list[dict[str, Any]] = []

        print("\n" + "=" * 60)
        print(f"ALPHA COMPARISON ({self.model_name.upper()})")
        print("=" * 60)
        print(f"{'Alpha':>10} | {'Train R²':>10} | {'Test R²':>10} | {'Non-zero Coefs':>15}")
        print("-" * 55)

        for alpha in alphas:
            # Create a fresh pipeline for each alpha value
            reg = RegressionPipeline(
                model_name=self.model_name,
                alpha=alpha,
                l1_ratio=self.l1_ratio,
                random_state=self.random_state,
            )
            reg.feature_names = self.feature_names
            reg.build_pipeline()

            assert reg.pipeline is not None
            reg.pipeline.fit(X_train, y_train)

            train_r2 = r2_score(y_train, reg.pipeline.predict(X_train))
            test_r2 = r2_score(y_test, reg.pipeline.predict(X_test))

            # Count non-zero coefficients (relevant for Lasso)
            coefs = reg.pipeline.named_steps["regressor"].coef_
            n_nonzero = int(np.sum(np.abs(coefs) > 1e-10))

            result = {
                "alpha": alpha,
                "train_r2": float(round(train_r2, 4)),
                "test_r2": float(round(test_r2, 4)),
                "n_nonzero_coefs": n_nonzero,
            }
            comparison_results.append(result)

            print(f"{alpha:>10.3f} | {train_r2:>10.4f} | {test_r2:>10.4f} | {n_nonzero:>15}")

        print("=" * 60)

        # Identify the best alpha by test R²
        best = max(comparison_results, key=lambda x: x["test_r2"])
        print(f"Best alpha: {best['alpha']} (Test R² = {best['test_r2']:.4f})")

        return comparison_results


# ---------------------------------------------------------------------------
# Quick demo — run this file directly to see the pipeline in action
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("REGRESSION PIPELINE DEMO")
    print("Dataset: California Housing")
    print("Models: Ridge, Lasso, ElasticNet")
    print("=" * 60)

    # --- Ridge Regression ---
    print("\n--- RIDGE REGRESSION ---")
    ridge = RegressionPipeline(model_name="ridge", alpha=1.0)
    X_train, X_test, y_train, y_test = ridge.load_data()
    ridge.build_pipeline()
    cv_results = ridge.fit_with_cross_validation(X_train, y_train, cv=5)
    print(f"CV R² = {cv_results['cv_r2_mean']:.4f} (+/- {cv_results['cv_r2_std']:.4f})")
    ridge.evaluate(X_test, y_test)

    # --- Lasso Regression (feature selection) ---
    print("\n--- LASSO REGRESSION ---")
    lasso = RegressionPipeline(model_name="lasso", alpha=0.1)
    lasso.feature_names = ridge.feature_names
    lasso.build_pipeline()
    lasso.fit(X_train, y_train)
    lasso.evaluate(X_test, y_test)
    print("\nLasso Coefficients (0 = feature removed):")
    for name, coef in lasso.get_coefficients().items():
        marker = " (REMOVED)" if abs(coef) < 1e-10 else ""
        print(f"  {name}: {coef:.6f}{marker}")

    # --- Compare alphas ---
    print("\n--- ALPHA COMPARISON (Lasso) ---")
    lasso.compare_alphas(X_train, y_train, X_test, y_test)
