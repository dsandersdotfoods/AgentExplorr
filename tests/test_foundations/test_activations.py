"""Tests for activation functions.

Verifies both forward computation and gradient correctness using
numerical gradient checking: (f(x+h) - f(x-h)) / (2h) ≈ f'(x).
"""

from __future__ import annotations

import numpy as np
import pytest

from agentexplorr.foundations.activations import (
    gelu,
    leaky_relu,
    relu,
    sigmoid,
    softmax,
    tanh,
)


class TestReLU:
    """Tests for ReLU activation."""

    def test_positive_input(self) -> None:
        x = np.array([1.0, 2.0, 3.0])
        val, grad = relu(x)
        np.testing.assert_array_equal(val, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(grad, [1.0, 1.0, 1.0])

    def test_negative_input(self) -> None:
        x = np.array([-1.0, -2.0, -3.0])
        val, grad = relu(x)
        np.testing.assert_array_equal(val, [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(grad, [0.0, 0.0, 0.0])

    def test_mixed_input(self) -> None:
        x = np.array([-2.0, 0.0, 3.0])
        val, grad = relu(x)
        np.testing.assert_array_equal(val, [0.0, 0.0, 3.0])

    def test_scalar(self) -> None:
        val, grad = relu(np.float64(5.0))
        assert val == 5.0
        assert grad == 1.0


class TestSigmoid:
    """Tests for Sigmoid activation."""

    def test_zero(self) -> None:
        val, grad = sigmoid(np.float64(0.0))
        assert val == pytest.approx(0.5)
        assert grad == pytest.approx(0.25)

    def test_large_positive(self) -> None:
        val, _ = sigmoid(np.float64(100.0))
        assert val == pytest.approx(1.0)

    def test_large_negative(self) -> None:
        val, _ = sigmoid(np.float64(-100.0))
        assert val == pytest.approx(0.0)

    def test_derivative_formula(self) -> None:
        """σ'(x) = σ(x)(1 - σ(x))."""
        x = np.array([-1.0, 0.0, 0.5, 1.0, 2.0])
        val, grad = sigmoid(x)
        expected_grad = val * (1 - val)
        np.testing.assert_allclose(grad, expected_grad)

    def test_numerical_gradient(self) -> None:
        """Verify analytic gradient against numerical gradient."""
        x = np.array([0.5, -0.5, 1.0])
        h = 1e-7
        _, analytic_grad = sigmoid(x)
        numerical_grad = (sigmoid(x + h)[0] - sigmoid(x - h)[0]) / (2 * h)
        np.testing.assert_allclose(analytic_grad, numerical_grad, atol=1e-5)


class TestTanh:
    """Tests for Tanh activation."""

    def test_zero(self) -> None:
        val, grad = tanh(np.float64(0.0))
        assert val == pytest.approx(0.0)
        assert grad == pytest.approx(1.0)

    def test_derivative_formula(self) -> None:
        """tanh'(x) = 1 - tanh²(x)."""
        x = np.array([-1.0, 0.0, 1.0])
        val, grad = tanh(x)
        expected_grad = 1 - val**2
        np.testing.assert_allclose(grad, expected_grad)


class TestLeakyReLU:
    """Tests for Leaky ReLU activation."""

    def test_positive(self) -> None:
        val, grad = leaky_relu(np.array([1.0, 2.0]))
        np.testing.assert_array_equal(val, [1.0, 2.0])
        np.testing.assert_array_equal(grad, [1.0, 1.0])

    def test_negative(self) -> None:
        val, grad = leaky_relu(np.array([-1.0, -2.0]), alpha=0.01)
        np.testing.assert_allclose(val, [-0.01, -0.02])
        np.testing.assert_allclose(grad, [0.01, 0.01])


class TestGELU:
    """Tests for GELU activation."""

    def test_zero(self) -> None:
        val, _ = gelu(np.float64(0.0))
        assert val == pytest.approx(0.0, abs=1e-6)

    def test_large_positive(self) -> None:
        val, _ = gelu(np.float64(10.0))
        assert val == pytest.approx(10.0, abs=0.01)

    def test_numerical_gradient(self) -> None:
        x = np.array([0.5, -0.5, 1.0])
        h = 1e-5
        _, analytic_grad = gelu(x)
        numerical_grad = (gelu(x + h)[0] - gelu(x - h)[0]) / (2 * h)
        np.testing.assert_allclose(analytic_grad, numerical_grad, atol=1e-3)


class TestSoftmax:
    """Tests for Softmax function."""

    def test_sums_to_one(self) -> None:
        x = np.array([1.0, 2.0, 3.0])
        val, _ = softmax(x)
        assert np.sum(val) == pytest.approx(1.0)

    def test_all_positive(self) -> None:
        x = np.array([-10.0, 0.0, 10.0])
        val, _ = softmax(x)
        assert np.all(val > 0)

    def test_preserves_ordering(self) -> None:
        x = np.array([1.0, 3.0, 2.0])
        val, _ = softmax(x)
        assert val[1] > val[2] > val[0]

    def test_numerical_stability(self) -> None:
        """Should not overflow with large values."""
        x = np.array([1000.0, 1001.0, 1002.0])
        val, _ = softmax(x)
        assert np.sum(val) == pytest.approx(1.0)
        assert np.all(np.isfinite(val))

    def test_uniform_input(self) -> None:
        x = np.array([1.0, 1.0, 1.0])
        val, _ = softmax(x)
        np.testing.assert_allclose(val, [1 / 3, 1 / 3, 1 / 3])
