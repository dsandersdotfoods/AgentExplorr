"""
Tests for Classical ML Pipelines
=================================

Tests verify pipeline construction and training on small synthetic data.
All tests run fast (<1s) using small datasets.
"""

from __future__ import annotations

from agentexplorr.classical_ml.pipelines.classification import ClassificationPipeline
from agentexplorr.classical_ml.pipelines.regression import RegressionPipeline
from agentexplorr.classical_ml.pipelines.clustering import ClusteringPipeline


class TestClassificationPipeline:
    """Tests for the classification pipeline."""

    def test_pipeline_creation(self) -> None:
        """Test that a pipeline can be created with default settings."""
        pipeline = ClassificationPipeline()
        assert pipeline is not None

    def test_train_on_wine(self) -> None:
        """Test training on the Wine dataset (small, fast)."""
        pipeline = ClassificationPipeline()
        metrics = pipeline.train_on_wine()

        # Should achieve reasonable accuracy on this easy dataset
        assert "accuracy" in metrics
        assert metrics["accuracy"] > 0.7  # Reasonable baseline


class TestRegressionPipeline:
    """Tests for the regression pipeline."""

    def test_pipeline_creation(self) -> None:
        """Test that a pipeline can be created."""
        pipeline = RegressionPipeline()
        assert pipeline is not None

    def test_train_on_housing(self) -> None:
        """Test training on California Housing (subset for speed)."""
        pipeline = RegressionPipeline()
        metrics = pipeline.train_on_housing(max_samples=500)

        assert "r2" in metrics
        assert "rmse" in metrics
        assert metrics["r2"] > 0  # Should explain some variance


class TestClusteringPipeline:
    """Tests for the clustering pipeline."""

    def test_pipeline_creation(self) -> None:
        """Test that a pipeline can be created."""
        pipeline = ClusteringPipeline()
        assert pipeline is not None

    def test_fit_on_iris(self) -> None:
        """Test clustering on Iris dataset."""
        pipeline = ClusteringPipeline(n_clusters=3)
        metrics = pipeline.fit_on_iris()

        assert "silhouette_score" in metrics
        assert metrics["silhouette_score"] > 0  # Better than random
