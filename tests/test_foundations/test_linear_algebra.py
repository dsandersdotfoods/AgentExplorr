"""Tests for linear algebra essentials.

Verifies core linear algebra operations and their ML interpretations.
"""

from __future__ import annotations

import numpy as np
import pytest


class TestLinearAlgebraTeacher:
    """Tests for the LinearAlgebraTeacher utility class."""

    def test_dot_product_orthogonal(self) -> None:
        """Orthogonal vectors should have zero dot product."""
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        result = np.dot(a, b)
        assert result == pytest.approx(0.0)

    def test_dot_product_parallel(self) -> None:
        """Parallel vectors should have maximum dot product."""
        a = np.array([1.0, 0.0])
        b = np.array([2.0, 0.0])
        result = np.dot(a, b)
        assert result == pytest.approx(2.0)

    def test_matrix_multiply_shapes(self) -> None:
        """(m×n) @ (n×p) should give (m×p)."""
        A = np.random.default_rng(42).standard_normal((3, 4))
        B = np.random.default_rng(42).standard_normal((4, 5))
        C = A @ B
        assert C.shape == (3, 5)

    def test_l1_norm(self) -> None:
        x = np.array([3.0, -4.0])
        assert np.linalg.norm(x, ord=1) == pytest.approx(7.0)

    def test_l2_norm(self) -> None:
        x = np.array([3.0, 4.0])
        assert np.linalg.norm(x, ord=2) == pytest.approx(5.0)

    def test_cosine_similarity_identical(self) -> None:
        """Identical vectors should have cosine similarity of 1."""
        a = np.array([1.0, 2.0, 3.0])
        cos_sim = np.dot(a, a) / (np.linalg.norm(a) ** 2)
        assert cos_sim == pytest.approx(1.0)

    def test_cosine_similarity_opposite(self) -> None:
        """Opposite vectors should have cosine similarity of -1."""
        a = np.array([1.0, 2.0])
        b = -a
        cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        assert cos_sim == pytest.approx(-1.0)

    def test_svd_reconstruction(self) -> None:
        """A = U @ diag(S) @ V^T should reconstruct the original matrix."""
        rng = np.random.default_rng(42)
        A = rng.standard_normal((5, 3))
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        reconstructed = U @ np.diag(s) @ Vt
        np.testing.assert_allclose(reconstructed, A, atol=1e-10)

    def test_low_rank_approximation(self) -> None:
        """Truncated SVD should give best rank-k approximation."""
        rng = np.random.default_rng(42)
        # Create a rank-2 matrix (should be perfectly reconstructed with k=2)
        a = rng.standard_normal((5, 2))
        b = rng.standard_normal((2, 4))
        A = a @ b  # rank-2 matrix

        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        # Keep only top 2 singular values
        A_approx = U[:, :2] @ np.diag(s[:2]) @ Vt[:2, :]
        np.testing.assert_allclose(A_approx, A, atol=1e-10)

    def test_eigendecomposition(self) -> None:
        """Eigenvalues of a symmetric positive definite matrix should be positive."""
        rng = np.random.default_rng(42)
        A = rng.standard_normal((4, 4))
        A = A @ A.T  # Make symmetric positive semi-definite
        eigenvalues, _ = np.linalg.eigh(A)
        assert np.all(eigenvalues >= -1e-10)
