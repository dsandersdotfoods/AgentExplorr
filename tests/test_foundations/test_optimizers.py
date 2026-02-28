"""Tests for optimizers.

Verifies that each optimizer converges on simple problems.
"""

from __future__ import annotations

import numpy as np

from agentexplorr.foundations.optimizers import SGD, Adam, AdamW, Momentum


def quadratic_gradient(params: list[np.ndarray]) -> list[np.ndarray]:
    """Gradient of f(x) = sum(x^2), which is 2x for each param."""
    return [2.0 * p for p in params]


class TestSGD:
    """Tests for vanilla SGD."""

    def test_converges_on_quadratic(self) -> None:
        """SGD should minimize f(x) = x^2 to x ~ 0."""
        optimizer = SGD(lr=0.1)
        params = [np.array([5.0])]
        for _ in range(100):
            grads = quadratic_gradient(params)
            params = optimizer.step(params, grads)
        assert abs(params[0][0]) < 0.01

    def test_learning_rate_affects_speed(self) -> None:
        """Higher LR should converge faster (on smooth problems)."""
        slow = SGD(lr=0.01)
        fast = SGD(lr=0.1)
        p_slow = [np.array([5.0])]
        p_fast = [np.array([5.0])]

        for _ in range(20):
            p_slow = slow.step(p_slow, quadratic_gradient(p_slow))
            p_fast = fast.step(p_fast, quadratic_gradient(p_fast))

        assert abs(p_fast[0][0]) < abs(p_slow[0][0])


class TestMomentum:
    """Tests for SGD with Momentum."""

    def test_converges(self) -> None:
        optimizer = Momentum(lr=0.01, beta=0.9)
        params = [np.array([5.0])]
        for _ in range(200):
            params = optimizer.step(params, quadratic_gradient(params))
        assert abs(params[0][0]) < 0.1

    def test_reduces_loss(self) -> None:
        """Momentum should reduce loss over many steps."""
        optimizer = Momentum(lr=0.01, beta=0.9)
        params = [np.array([5.0])]
        initial_val = abs(params[0][0])
        for _ in range(100):
            params = optimizer.step(params, quadratic_gradient(params))
        assert abs(params[0][0]) < initial_val


class TestAdam:
    """Tests for Adam optimizer."""

    def test_converges(self) -> None:
        optimizer = Adam(lr=0.1)
        params = [np.array([5.0])]
        for _ in range(200):
            params = optimizer.step(params, quadratic_gradient(params))
        assert abs(params[0][0]) < 0.1

    def test_multidimensional(self) -> None:
        """Should work on multi-dimensional problems."""
        optimizer = Adam(lr=0.1)
        params = [np.array([3.0, -4.0, 2.0])]
        for _ in range(200):
            grads = quadratic_gradient(params)
            params = optimizer.step(params, grads)
        np.testing.assert_allclose(params[0], [0.0, 0.0, 0.0], atol=0.1)


class TestAdamW:
    """Tests for AdamW optimizer."""

    def test_converges(self) -> None:
        optimizer = AdamW(lr=0.1, weight_decay=0.01)
        params = [np.array([5.0])]
        for _ in range(200):
            params = optimizer.step(params, quadratic_gradient(params))
        assert abs(params[0][0]) < 0.1

    def test_weight_decay_shrinks_params(self) -> None:
        """Weight decay should push parameters toward zero."""
        no_decay = Adam(lr=0.01)
        with_decay = AdamW(lr=0.01, weight_decay=0.1)

        p_no = [np.array([5.0])]
        p_wd = [np.array([5.0])]

        for _ in range(50):
            # Use zero gradient — only weight decay should act
            zero_grad = [np.array([0.0])]
            p_no = no_decay.step(p_no, zero_grad)
            p_wd = with_decay.step(p_wd, zero_grad)

        # With_decay should shrink due to weight decay
        assert abs(p_wd[0][0]) < abs(p_no[0][0])
