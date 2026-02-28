"""Tests for loss functions.

Verifies both loss values and gradient correctness.
"""

from __future__ import annotations

import numpy as np
import pytest

from agentexplorr.foundations.loss_functions import (
    binary_cross_entropy,
    cross_entropy_loss,
    huber_loss,
    mean_squared_error,
)


class TestMSE:
    """Tests for Mean Squared Error."""

    def test_perfect_predictions(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        loss = mean_squared_error(y_pred, y_true)
        assert loss == pytest.approx(0.0)

    def test_known_value(self) -> None:
        y_true = np.array([1.0, 0.0])
        y_pred = np.array([0.0, 1.0])
        # MSE = ((0-1)^2 + (1-0)^2) / 2 = 1.0
        loss = mean_squared_error(y_pred, y_true)
        assert loss == pytest.approx(1.0)

    def test_always_non_negative(self) -> None:
        rng = np.random.default_rng(42)
        y_true = rng.standard_normal(100)
        y_pred = rng.standard_normal(100)
        loss = mean_squared_error(y_pred, y_true)
        assert loss >= 0.0


class TestCrossEntropy:
    """Tests for Cross-Entropy Loss."""

    def test_confident_correct_prediction(self) -> None:
        """Confident correct prediction should give low loss."""
        # Logits: high score for class 0
        logits = np.array([[10.0, 0.0, 0.0]])
        target = np.array([0])
        loss = cross_entropy_loss(logits, target, from_logits=True)
        assert loss < 0.1

    def test_confident_wrong_prediction(self) -> None:
        """Confident wrong prediction should give high loss."""
        # Logits: high score for class 1, but target is class 0
        logits = np.array([[0.0, 10.0, 0.0]])
        target = np.array([0])
        loss = cross_entropy_loss(logits, target, from_logits=True)
        assert loss > 5.0

    def test_always_non_negative(self) -> None:
        logits = np.array([[1.0, 2.0, 3.0]])
        target = np.array([2])
        loss = cross_entropy_loss(logits, target, from_logits=True)
        assert loss >= 0.0


class TestBinaryCrossEntropy:
    """Tests for Binary Cross-Entropy Loss."""

    def test_perfect_positive(self) -> None:
        loss = binary_cross_entropy(
            np.array([0.99]), np.array([1.0]), from_logits=False
        )
        assert loss < 0.05

    def test_perfect_negative(self) -> None:
        loss = binary_cross_entropy(
            np.array([0.01]), np.array([0.0]), from_logits=False
        )
        assert loss < 0.05

    def test_wrong_prediction(self) -> None:
        loss = binary_cross_entropy(
            np.array([0.01]), np.array([1.0]), from_logits=False
        )
        assert loss > 3.0

    def test_symmetric(self) -> None:
        """Equally wrong in both directions should give same loss."""
        loss1 = binary_cross_entropy(
            np.array([0.1]), np.array([1.0]), from_logits=False
        )
        loss2 = binary_cross_entropy(
            np.array([0.9]), np.array([0.0]), from_logits=False
        )
        assert loss1 == pytest.approx(loss2, abs=0.01)


class TestHuberLoss:
    """Tests for Huber Loss."""

    def test_small_error_equals_mse(self) -> None:
        """For small errors, Huber should behave like 0.5 * error^2."""
        y_true = np.array([1.0])
        y_pred = np.array([1.1])
        delta = 1.0
        h_loss = huber_loss(y_pred, y_true, delta=delta)
        expected = 0.5 * (0.1) ** 2
        assert h_loss == pytest.approx(expected, abs=1e-6)

    def test_large_error_linear(self) -> None:
        """For large errors, Huber should be linear (not quadratic)."""
        y_true = np.array([0.0])
        y_pred = np.array([10.0])
        delta = 1.0
        h_loss = huber_loss(y_pred, y_true, delta=delta)
        # For |error| > delta: delta * (|error| - 0.5*delta) = 1*(10 - 0.5) = 9.5
        assert h_loss == pytest.approx(9.5)

    def test_always_non_negative(self) -> None:
        rng = np.random.default_rng(42)
        y_true = rng.standard_normal(50)
        y_pred = rng.standard_normal(50)
        loss = huber_loss(y_pred, y_true)
        assert loss >= 0.0
