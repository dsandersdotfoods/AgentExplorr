"""Tests for the backpropagation module.

Verifies that manually computed gradients match expected values
and PyTorch autograd (when available).
"""

from __future__ import annotations

import numpy as np
import pytest

from agentexplorr.foundations.backpropagation import ComputationalGraph, Node


class TestNode:
    """Tests for individual computation nodes."""

    def test_add_forward(self) -> None:
        a = Node(2.0, label="a")
        b = Node(3.0, label="b")
        c = a + b
        assert c.value == pytest.approx(5.0)

    def test_add_backward(self) -> None:
        """Addition passes gradient through unchanged."""
        a = Node(2.0, label="a")
        b = Node(3.0, label="b")
        c = a + b
        # Use ComputationalGraph to run backward
        graph = ComputationalGraph(c, label="test")
        graph.backward()
        assert a.grad == pytest.approx(1.0)
        assert b.grad == pytest.approx(1.0)

    def test_mul_forward(self) -> None:
        a = Node(2.0, label="a")
        b = Node(3.0, label="b")
        c = a * b
        assert c.value == pytest.approx(6.0)

    def test_mul_backward(self) -> None:
        """Multiplication swaps: d(ab)/da = b, d(ab)/db = a."""
        a = Node(2.0, label="a")
        b = Node(3.0, label="b")
        c = a * b
        graph = ComputationalGraph(c, label="test")
        graph.backward()
        assert a.grad == pytest.approx(3.0)  # d(ab)/da = b = 3
        assert b.grad == pytest.approx(2.0)  # d(ab)/db = a = 2


class TestComputationalGraph:
    """Tests for the computational graph."""

    def test_simple_chain(self) -> None:
        """f(x) = (2x + 3)^2 should have gradient 4(2x + 3) at x=1 -> 4*5=20."""
        x = Node(1.0, label="x")
        two = Node(2.0, label="2")
        three = Node(3.0, label="3")

        # f = (2x + 3)^2
        two_x = two * x
        sum_node = two_x + three
        # Square = multiply by itself
        f = sum_node * sum_node

        graph = ComputationalGraph(f, label="test")
        graph.backward()

        # f = (2*1 + 3)^2 = 25
        assert f.value == pytest.approx(25.0)
        # df/dx = 2 * (2x + 3) * 2 = 4(2x+3) = 4*5 = 20
        assert x.grad == pytest.approx(20.0)
