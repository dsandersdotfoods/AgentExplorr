"""
Linear Algebra Essentials for Machine Learning
=================================================

This module teaches the linear algebra that underpins ALL of machine learning.
Every neural network, every attention mechanism, every embedding lookup — it
all comes down to the operations in this file.

    MACHINE LEARNING IS MATRIX MULTIPLICATION WITH EXTRA STEPS.

    A neural network "forward pass" is literally:
        x -> W1 @ x + b1 -> activation -> W2 @ ... -> output

WHAT'S COVERED:
    1. Vectors and Dot Products — the atom of ML computation
    2. Matrix Multiplication   — the core operation of neural networks
    3. Norms (L1, L2)          — measuring magnitude, regularization
    4. Cosine Similarity       — comparing embeddings
    5. Eigendecomposition      — PCA, understanding variance
    6. SVD                     — the factorization behind LoRA
    7. Broadcasting            — numpy/PyTorch's implicit expansion

NOTATION GUIDE:
    - Vectors: lowercase (a, b, x)  |  Matrices: uppercase (W, A, U)
    - ||x|| = norm (length)  |  a . b = dot product  |  A^T = transpose

LEARNING RESOURCES:
    - VIDEO SERIES (ESSENTIAL): 3Blue1Brown "Essence of Linear Algebra"
      https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab

    - COURSE: MIT 18.06 Linear Algebra (Gilbert Strang)
      https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/

    - COURSE: Stanford CS229 — Linear Algebra Review
      https://cs229.stanford.edu/section/cs229-linalg.pdf

    - PAPER: "LoRA: Low-Rank Adaptation of Large Language Models"
      (Hu et al., 2021) — https://arxiv.org/abs/2106.09685

    - PAPER: "Attention Is All You Need" (Vaswani et al., 2017)
      https://arxiv.org/abs/1706.03762
      (Attention scores are dot products — see Section 3.2.1)

    - BLOG: "The Matrix Calculus You Need For Deep Learning"
      (Parr & Howard) — https://explained.ai/matrix-calculus/
"""

from __future__ import annotations

import numpy as np

from agentexplorr.core.logging import get_logger

logger = get_logger(__name__)


class LinearAlgebraTeacher:
    """Interactive teacher for the linear algebra behind machine learning.

    Provides static methods that DEMONSTRATE each concept with real numbers.
    Every method prints formatted output with clear explanations.

    Usage:
        LinearAlgebraTeacher.dot_product_demo()       # single demo
        LinearAlgebraTeacher.run_all_demos()           # all in sequence
    """

    # =====================================================================
    # 1. VECTORS AND DOT PRODUCTS
    # =====================================================================

    @staticmethod
    def dot_product_demo() -> dict[str, np.ndarray]:
        """Demonstrate the dot product — the fundamental atom of ML computation.

        THE MATH:
            Algebraic:  a . b = a1*b1 + a2*b2 + ... + an*bn
            Geometric:  a . b = |a| * |b| * cos(theta)

        WHY THIS MATTERS:
            - Attention scores: Q . K^T gives similarity between query and key
            - Logits: output layer computes W . x (each row is a class template)
            - Embeddings: similar words have high dot product

        Returns:
            Dictionary with computed values for inspection.
        """
        print("\n" + "=" * 70)
        print("  1. VECTORS AND DOT PRODUCTS")
        print("     The fundamental atom of ML computation")
        print("=" * 70)

        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])

        elementwise = a * b
        dot_manual = np.sum(elementwise)
        dot_numpy = np.dot(a, b)

        print(f"\n  a = {a}")
        print(f"  b = {b}")
        print(f"  Element-wise: a * b = {elementwise}")
        print(f"  Dot product:  sum(a * b) = {dot_manual}")

        # Geometric interpretation
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        cos_theta = dot_numpy / (norm_a * norm_b)
        theta_degrees = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

        print(f"\n  |a| = {norm_a:.4f},  |b| = {norm_b:.4f}")
        print(f"  cos(theta) = {cos_theta:.4f}")
        print(f"  theta = {theta_degrees:.2f} degrees (small = similar direction)")

        # Special cases
        print("\n  --- Special Cases ---")
        c, d = np.array([1.0, 0.0]), np.array([0.0, 1.0])
        print(f"  Perpendicular: {c} . {d} = {np.dot(c, d)}  (= 0)")
        e, f = np.array([1.0, 0.0]), np.array([-1.0, 0.0])
        print(f"  Opposite:      {e} . {f} = {np.dot(e, f)}  (< 0)")

        # ML connection
        print("\n  --- ML Connection: Attention-like Scoring ---")
        query = np.array([0.9, 0.1, 0.8, 0.2])
        key_relevant = np.array([0.85, 0.15, 0.75, 0.25])
        key_irrelevant = np.array([0.1, 0.9, 0.2, 0.7])

        print(f"  query          = {query}")
        print(f"  key_relevant   = {key_relevant}")
        print(f"  key_irrelevant = {key_irrelevant}")
        print(f"  score(relevant)   = {np.dot(query, key_relevant):.4f}  (HIGH)")
        print(f"  score(irrelevant) = {np.dot(query, key_irrelevant):.4f}  (LOW)")
        print("  -> Dot product naturally measures 'relevance'. This IS attention.")

        logger.info("dot_product_demo_complete", dot_product=float(dot_numpy))

        return {
            "a": a, "b": b, "dot_product": dot_numpy,
            "angle_degrees": theta_degrees, "cos_theta": cos_theta,
        }

    # =====================================================================
    # 2. MATRIX MULTIPLICATION
    # =====================================================================

    @staticmethod
    def matrix_multiply_as_neural_layer() -> dict[str, np.ndarray]:
        """Demonstrate matrix multiplication as the core neural network operation.

        THE MATH:
            (m x n) @ (n x p) = (m x p).  Inner dimensions must match.
            C[i,j] = dot product of row i of A with column j of B.

        WHY THIS IS NEURAL NETWORKS:
            y = W @ x + b  — each ROW of W is a "pattern detector" whose
            dot product with x measures how much x matches that pattern.

        COST: O(m*n*p) — why GPUs matter (they parallelize this).

        Returns:
            Dictionary with matrices and results for inspection.
        """
        print("\n" + "=" * 70)
        print("  2. MATRIX MULTIPLICATION")
        print("     The core operation of neural networks")
        print("=" * 70)

        A = np.array([[1, 2], [3, 4], [5, 6]])
        B = np.array([[7, 8, 9], [10, 11, 12]])
        C = A @ B

        print(f"\n  A (3x2):\n{_indent(A)}")
        print(f"\n  B (2x3):\n{_indent(B)}")
        print(f"\n  C = A @ B (3x3):\n{_indent(C)}")
        print(f"  Shape: ({A.shape[0]}x{A.shape[1]}) @ "
              f"({B.shape[0]}x{B.shape[1]}) = ({C.shape[0]}x{C.shape[1]})")

        # Neural network interpretation
        print("\n  --- Neural Network: y = W @ x + b ---")
        W = np.array([[0.5, -0.3, 0.8], [-0.2, 0.7, 0.1]])
        b = np.array([0.1, -0.1])
        x = np.array([1.0, 0.5, 0.8])
        y = W @ x + b

        print(f"  W (2 neurons x 3 features):\n{_indent(W)}")
        print(f"  b = {b},  x = {x}")
        print(f"  Neuron 0: {W[0]} . {x} + {b[0]} = {y[0]:.2f}")
        print(f"  Neuron 1: {W[1]} . {x} + {b[1]} = {y[1]:.2f}")

        # Batch processing
        print("\n  --- Batch Processing ---")
        X_batch = np.array([[1.0, 0.5, 0.8], [0.2, 0.9, 0.4], [0.7, 0.3, 0.6]])
        Y_batch = X_batch @ W.T + b
        print("  3 samples processed simultaneously:")
        print(f"  Input  (3x3):\n{_indent(X_batch)}")
        print(f"  Output (3x2):\n{_indent(np.round(Y_batch, 4))}")

        # Computational cost
        m, n = 4096, 4096
        print(f"\n  --- Cost: ({m}x{n}) @ ({n}x1) = {m*n:,} ops ---")
        print(f"  GPU at 300 TFLOPS: ~{m*n/300e12*1e6:.1f} microseconds")

        logger.info("matrix_multiply_demo_complete", output_shape=str(C.shape))
        return {"W": W, "b": b, "x": x, "y": y, "batch_output": Y_batch}

    # =====================================================================
    # 3. NORMS AND REGULARIZATION
    # =====================================================================

    @staticmethod
    def norms_and_regularization() -> dict[str, float]:
        """Demonstrate L1 and L2 norms and their connection to regularization.

        THE MATH:
            L1: ||x||_1 = SUM(|xi|)     (Manhattan distance)
            L2: ||x||_2 = sqrt(SUM(xi^2)) (Euclidean distance)

        REGULARIZATION:
            L1 -> sparse models (weights become exactly 0). WHY: L1 unit ball
            has corners at axes; optimal point often lands on a corner.
            L2 -> small weights (never exactly 0). WHY: L2 unit ball is smooth.

        Returns:
            Dictionary with norm values for inspection.
        """
        print("\n" + "=" * 70)
        print("  3. NORMS — Measuring Vector Magnitude")
        print("     L1 vs L2 and why they matter for regularization")
        print("=" * 70)

        x = np.array([3.0, -4.0, 0.0, 1.0, -2.0])
        l1 = np.linalg.norm(x, ord=1)
        l2 = np.linalg.norm(x, ord=2)

        print(f"\n  x = {x}")
        print(f"  L1: |3|+|-4|+|0|+|1|+|-2| = {l1}")
        print(f"  L2: sqrt(9+16+0+1+4) = {l2:.4f}")

        # Regularization simulation
        print("\n  --- L1 vs L2 Regularization ---")
        w = np.array([0.5, -0.3, 0.01, -0.8, 0.02])
        lr, lam = 0.1, 0.5

        l1_grad = np.sign(w) * lam       # Constant push toward 0
        l2_grad = 2 * w * lam            # Proportional push toward 0
        w_l1 = w - lr * l1_grad
        w_l2 = w - lr * l2_grad

        print(f"  Starting weights: {w}")
        print(f"  After 1 L1 step:  {w_l1}")
        print("    -> Small weights (0.01) moved as much as large ones!")
        print(f"  After 1 L2 step:  {np.round(w_l2, 4)}")
        print("    -> Small weights barely moved (proportional to size)")

        # Multiple steps
        print("\n  --- After 10 Steps (regularization only) ---")
        w_l1_final, w_l2_final = w.copy(), w.copy()
        for _ in range(10):
            w_l1_final -= lr * np.sign(w_l1_final) * lam
            w_l1_final = np.where(np.abs(w_l1_final) < 1e-10, 0.0, w_l1_final)
            w_l2_final -= lr * 2 * w_l2_final * lam

        print(f"  L1 result: {w_l1_final}")
        print(f"  L2 result: {np.round(w_l2_final, 6)}")
        print(f"  L1 zeros: {np.sum(w_l1_final == 0)}/{len(w)} (sparse!)")
        print(f"  L2 zeros: {np.sum(w_l2_final == 0)}/{len(w)} (never zero)")
        print("\n  KEY: L1 -> sparsity (feature selection)")
        print("       L2 -> small weights (weight decay)")

        logger.info("norms_demo_complete", l1=float(l1), l2=float(l2))
        return {"l1_norm": float(l1), "l2_norm": float(l2)}

    # =====================================================================
    # 4. COSINE SIMILARITY
    # =====================================================================

    @staticmethod
    def cosine_similarity_demo() -> dict[str, float]:
        """Demonstrate cosine similarity — comparing directions, not magnitudes.

        THE MATH:
            cos(theta) = (a . b) / (||a|| * ||b||)
            Range: [-1, +1].  +1 = same direction, 0 = orthogonal, -1 = opposite.

        WHY THIS MATTERS:
            Embeddings are compared this way in RAG, sentence-transformers, and
            contrastive learning. Magnitude-invariant: a long document and short
            document about the same topic get high similarity.

        Returns:
            Dictionary with similarity values for inspection.
        """
        print("\n" + "=" * 70)
        print("  4. COSINE SIMILARITY")
        print("     Comparing directions, not magnitudes")
        print("=" * 70)

        def cosine_sim(v1: np.ndarray, v2: np.ndarray) -> float:
            """Compute cosine similarity between two vectors."""
            dot = np.dot(v1, v2)
            norms = np.linalg.norm(v1) * np.linalg.norm(v2)
            return float(dot / norms) if norms > 0 else 0.0

        a = np.array([1.0, 2.0, 3.0])
        b = np.array([2.0, 4.0, 6.0])
        print(f"\n  a = {a},  b = {b} (= 2*a)")
        sim_par = cosine_sim(a, b)
        print(f"  cosine_sim = {sim_par:.4f}  (1.0 = identical direction)")

        c, d = np.array([1.0, 0.0]), np.array([0.0, 1.0])
        sim_perp = cosine_sim(c, d)
        print(f"  {c} vs {d}: {sim_perp:.4f}  (perpendicular)")

        e, f = np.array([1.0, 1.0]), np.array([-1.0, -1.0])
        sim_opp = cosine_sim(e, f)
        print(f"  {e} vs {f}: {sim_opp:.4f}  (opposite)")

        # Simulated word embeddings
        # Dimensions loosely: [royalty, femininity, age, power]
        print("\n  --- Simulated Word Embeddings ---")
        embeddings = {
            "king":   np.array([0.9, 0.1, 0.5, 0.9]),
            "queen":  np.array([0.9, 0.9, 0.5, 0.8]),
            "man":    np.array([0.1, 0.1, 0.5, 0.3]),
            "woman":  np.array([0.1, 0.9, 0.5, 0.3]),
            "child":  np.array([0.0, 0.5, 0.1, 0.1]),
            "apple":  np.array([0.0, 0.0, 0.3, 0.0]),
        }
        for word, emb in embeddings.items():
            print(f"    {word:8s} = {emb}")

        print("\n  Pairwise similarities:")
        for w1, w2 in [("king","queen"), ("king","man"), ("queen","woman"),
                       ("king","apple"), ("man","woman")]:
            print(f"    sim({w1:8s}, {w2:8s}) = {cosine_sim(embeddings[w1], embeddings[w2]):.4f}")

        # Word analogy
        analogy = embeddings["king"] - embeddings["man"] + embeddings["woman"]
        sim_analogy = cosine_sim(analogy, embeddings["queen"])
        print("\n  --- Word Analogy: king - man + woman = ? ---")
        print(f"  Result vector: {analogy}")
        print(f"  Closest to 'queen': cosine_sim = {sim_analogy:.4f}")

        logger.info("cosine_similarity_demo_complete", analogy_sim=float(sim_analogy))
        return {
            "sim_parallel": sim_par, "sim_perpendicular": sim_perp,
            "sim_opposite": sim_opp, "analogy_similarity": sim_analogy,
        }

    # =====================================================================
    # 5. EIGENVALUES AND EIGENVECTORS
    # =====================================================================

    @staticmethod
    def eigendecomposition_demo() -> dict[str, np.ndarray]:
        """Demonstrate eigendecomposition — principal directions of a matrix.

        THE MATH:
            A @ v = lambda * v
            Eigenvector v is a direction that only SCALES (by lambda) when
            multiplied by A. Eigenvalue lambda is the scale factor.

        WHY THIS MATTERS — PCA:
            1. Compute covariance matrix of your data
            2. Eigenvectors = directions of max variance
            3. Eigenvalues = how much variance in each direction
            4. Keep top-k eigenvectors -> dimensionality reduction

        Returns:
            Dictionary with eigenvalues, eigenvectors for inspection.
        """
        print("\n" + "=" * 70)
        print("  5. EIGENVALUES AND EIGENVECTORS")
        print("     What stays the same under transformation")
        print("=" * 70)

        A = np.array([[4.0, 2.0], [1.0, 3.0]])
        eigenvalues, eigenvectors = np.linalg.eig(A)

        print(f"\n  A:\n{_indent(A)}")
        print(f"  Eigenvalues:  {eigenvalues}")
        print(f"  Eigenvectors:\n{_indent(eigenvectors)}")

        # Verify A @ v = lambda * v
        print("\n  --- Verification: A @ v = lambda * v ---")
        for i in range(len(eigenvalues)):
            v = eigenvectors[:, i]
            lam = eigenvalues[i]
            Av = A @ v
            lam_v = lam * v
            print(f"  v{i} = [{v[0]:.4f}, {v[1]:.4f}], lambda = {lam:.4f}")
            print(f"    A@v     = [{Av[0]:.4f}, {Av[1]:.4f}]")
            print(f"    lambda*v = [{lam_v[0]:.4f}, {lam_v[1]:.4f}]  Match: {np.allclose(Av, lam_v)}")

        # PCA on synthetic data
        print("\n  --- PCA: Finding Principal Directions ---")
        np.random.seed(42)
        n_points = 200
        t = np.random.randn(n_points)
        noise = np.random.randn(n_points) * 0.3
        data = np.column_stack([2 * t + noise, 1.5 * t + noise])
        data_centered = data - data.mean(axis=0)

        cov = np.cov(data_centered.T)
        pca_vals, pca_vecs = np.linalg.eigh(cov)
        # Sort descending
        idx = np.argsort(pca_vals)[::-1]
        pca_vals, pca_vecs = pca_vals[idx], pca_vecs[:, idx]

        var_explained = pca_vals / pca_vals.sum()
        print(f"  {n_points} data points, covariance:\n{_indent(np.round(cov, 4))}")
        print(f"  Eigenvalues: {np.round(pca_vals, 4)}")
        print(f"  PC1 direction: [{pca_vecs[0,0]:.4f}, {pca_vecs[1,0]:.4f}]")
        print(f"  Variance explained: PC1={var_explained[0]*100:.1f}%, "
              f"PC2={var_explained[1]*100:.1f}%")
        print(f"  -> Keep only PC1: 2D -> 1D, retaining {var_explained[0]*100:.1f}% info!")

        logger.info("eigendecomposition_demo_complete",
                     var_pc1=float(var_explained[0]))
        return {
            "eigenvalues": eigenvalues, "eigenvectors": eigenvectors,
            "pca_eigenvalues": pca_vals, "variance_explained": var_explained,
        }

    # =====================================================================
    # 6. SVD AND LoRA
    # =====================================================================

    @staticmethod
    def svd_and_lora() -> dict[str, np.ndarray | float]:
        """Demonstrate SVD and its connection to LoRA fine-tuning.

        THE MATH:
            A = U @ Sigma @ V^T  (any matrix!)
            U (m x m): left singular vectors (rotation)
            Sigma (m x n): singular values on diagonal (scaling)
            V^T (n x n): right singular vectors (rotation)

        LOW-RANK APPROXIMATION:
            Keep top k singular values -> best rank-k approximation (Eckart-Young).

        CONNECTION TO LoRA:
            LoRA: Delta_W ~= B @ A, where B is (d x r), A is (r x d), r << d.
            This IS a rank-r approximation. SVD proves it's theoretically sound.
            See: https://arxiv.org/abs/2106.09685

        Returns:
            Dictionary with SVD components and approximation errors.
        """
        print("\n" + "=" * 70)
        print("  6. SINGULAR VALUE DECOMPOSITION (SVD)")
        print("     The factorization behind LoRA")
        print("=" * 70)

        A = np.array([[1., 2., 3.], [4., 5., 6.],
                      [7., 8., 9.], [10., 11., 12.]])

        U, sigma, Vt = np.linalg.svd(A, full_matrices=False)

        print(f"\n  A (4x3):\n{_indent(A)}")
        print(f"  Singular values: {np.round(sigma, 4)}")
        recon_err = np.linalg.norm(A - U @ np.diag(sigma) @ Vt)
        print(f"  Reconstruction error: {recon_err:.2e} (exact!)")

        # Low-rank approximation
        print("\n  --- Low-Rank Approximation ---")
        for k in range(1, len(sigma) + 1):
            A_k = U[:, :k] @ np.diag(sigma[:k]) @ Vt[:k, :]
            err = np.linalg.norm(A - A_k, 'fro')
            energy = np.sum(sigma[:k]**2) / np.sum(sigma**2) * 100
            params_k = A.shape[0] * k + k + k * A.shape[1]
            print(f"  Rank {k}: error={err:.4f}, energy={energy:.2f}%, "
                  f"params={params_k} (vs {A.shape[0]*A.shape[1]} full)")

        # LoRA simulation
        print("\n  --- LoRA Simulation ---")
        d = 64
        np.random.seed(42)
        w_pretrained = np.random.randn(d, d) * 0.1  # noqa: F841

        # Simulate low-rank weight update
        true_rank = 4
        delta_W = (np.random.randn(d, true_rank) * 0.01) @ \
                  (np.random.randn(true_rank, d) * 0.01)

        U_dw, s_dw, Vt_dw = np.linalg.svd(delta_W, full_matrices=False)

        print(f"  Weight matrix: {d}x{d} = {d*d} params")
        print(f"  True delta_W rank: {true_rank}")
        print(f"  Top 8 singular values: {np.round(s_dw[:8], 6)}")
        print(f"  (Sharp drop after rank {true_rank}!)\n")

        for r in [1, 2, 4, 8, 16]:
            B = U_dw[:, :r] @ np.diag(np.sqrt(s_dw[:r]))
            A_lora = np.diag(np.sqrt(s_dw[:r])) @ Vt_dw[:r, :]
            rel_err = np.linalg.norm(delta_W - B @ A_lora) / np.linalg.norm(delta_W)
            lora_params = 2 * d * r
            compress = d * d / lora_params
            print(f"  r={r:2d}: params={lora_params:5d} ({compress:.1f}x compression), "
                  f"rel_error={rel_err:.6f}")

        print(f"\n  At r={true_rank}, error -> 0! LoRA's insight: weight updates")
        print(f"  are low-rank, so B({d}x{true_rank}) @ A({true_rank}x{d}) suffices.")
        print(f"  Compression: {d*d/(2*d*true_rank):.1f}x fewer parameters!")
        print("  Paper: https://arxiv.org/abs/2106.09685")

        logger.info("svd_and_lora_demo_complete", true_rank=true_rank)
        return {
            "U": U, "sigma": sigma, "Vt": Vt,
            "delta_W_rank": true_rank, "reconstruction_error": float(recon_err),
        }

    # =====================================================================
    # 7. BROADCASTING
    # =====================================================================

    @staticmethod
    def broadcasting_demo() -> dict[str, np.ndarray]:
        """Demonstrate broadcasting — numpy/PyTorch's implicit dimension expansion.

        THE RULES (align from the RIGHT):
            1. Equal dimensions -> OK
            2. One dimension is 1 -> stretch it
            3. Different and neither is 1 -> ERROR

            Example: (3, 4, 1) + (1, 4, 5) -> (3, 4, 5)

        ML PATTERNS:
            - Adding bias:   (batch, features) + (features,) -> per-sample bias
            - Normalization:  (batch, features) - (features,) -> centered
            - Attention mask: (batch, heads, seq, seq) + (1, 1, seq, seq)
            - Pairwise dist:  (N, 1, D) - (1, M, D) -> (N, M, D) -> no loops!

        Returns:
            Dictionary with broadcasted results for inspection.
        """
        print("\n" + "=" * 70)
        print("  7. BROADCASTING")
        print("     Numpy/PyTorch's implicit dimension expansion")
        print("=" * 70)

        # Scalar + array
        a = np.array([1.0, 2.0, 3.0])
        print(f"\n  {a} + 10 = {a + 10}")
        print("  Shape: (3,) + () -> (3,)")

        # Outer operation via broadcasting
        col = np.array([[1], [2], [3]])
        row = np.array([[10, 20, 30]])
        outer = col + row
        print(f"\n  Column (3,1) + Row (1,3) -> outer sum (3,3):\n{_indent(outer)}")

        # ML Pattern: bias
        print("\n  --- ML: Adding Bias to a Batch ---")
        batch = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
        bias = np.array([0.1, 0.2, 0.3])
        result = batch + bias
        print(f"  Batch (3x3):\n{_indent(batch)}")
        print(f"  Bias (3,): {bias}")
        print(f"  Result (3x3):\n{_indent(result)}")
        print(f"  Shape: ({batch.shape}) + ({bias.shape},) -> same bias per sample")

        # ML Pattern: normalization
        print("\n  --- ML: Feature-wise Normalization ---")
        mean = batch.mean(axis=0)
        std = batch.std(axis=0)
        normed = (batch - mean) / std
        print(f"  Means: {mean}, Stds: {np.round(std, 4)}")
        print(f"  Normalized:\n{_indent(np.round(normed, 4))}")
        print("  Each column: mean ~0, std ~1")

        # ML Pattern: pairwise distances
        print("\n  --- ML: Pairwise Distances (No Loops!) ---")
        pts_a = np.array([[0., 0.], [1., 0.], [0., 1.]])
        pts_b = np.array([[1., 1.], [2., 2.]])
        diff = pts_a[:, np.newaxis, :] - pts_b[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        print(f"  A: {pts_a.tolist()}")
        print(f"  B: {pts_b.tolist()}")
        print(f"  Distances (3x2):\n{_indent(np.round(dists, 4))}")
        print("  Shape: (3,1,2) - (1,2,2) -> (3,2,2) -> sum -> (3,2)")

        # Common mistakes
        print("\n  --- Common Mistakes ---")
        print("  (3,4) + (3,)   -> ERROR (3 != 4)")
        print("  (3,4) + (4,)   -> OK: (3,4)")
        print("  (3,4) + (3,1)  -> OK: (3,4)")
        print("  TIP: use np.newaxis to add dimensions for alignment")

        logger.info("broadcasting_demo_complete")
        return {"outer_product": outer, "batch_plus_bias": result,
                "pairwise_distances": dists}

    # =====================================================================
    # RUN ALL DEMOS
    # =====================================================================

    @classmethod
    def run_all_demos(cls) -> None:
        """Run all demos in pedagogical order.

        Order: dot products -> matmul -> norms -> cosine sim ->
               eigen -> SVD/LoRA -> broadcasting.
        """
        print("\n" + "#" * 70)
        print("#  LINEAR ALGEBRA ESSENTIALS FOR MACHINE LEARNING")
        print("#")
        print("#  Companion: 3Blue1Brown 'Essence of Linear Algebra'")
        print("#  https://www.youtube.com/playlist?"
              "list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab")
        print("#" * 70)

        cls.dot_product_demo()
        cls.matrix_multiply_as_neural_layer()
        cls.norms_and_regularization()
        cls.cosine_similarity_demo()
        cls.eigendecomposition_demo()
        cls.svd_and_lora()
        cls.broadcasting_demo()

        print("\n" + "=" * 70)
        print("  ALL DEMOS COMPLETE — Key Takeaways:")
        print("=" * 70)
        print("    1. DOT PRODUCTS measure similarity (attention, logits)")
        print("    2. MATRIX MULTIPLY is the core neural network operation")
        print("    3. L1 NORM -> sparsity,  L2 NORM -> smoothness")
        print("    4. COSINE SIMILARITY compares directions (embeddings)")
        print("    5. EIGENVECTORS are principal directions (PCA)")
        print("    6. SVD enables low-rank approximation (LoRA)")
        print("    7. BROADCASTING eliminates loops (efficient GPU code)")
        print()
        print("  Next: activations.py -> loss_functions.py -> backpropagation.py")
        print("  Watch: 3Blue1Brown 'Essence of Linear Algebra'")
        print("  https://www.youtube.com/playlist?"
              "list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab")
        print()


# =========================================================================
# HELPERS
# =========================================================================

def _indent(array: np.ndarray, spaces: int = 6) -> str:
    """Format a numpy array with consistent indentation for pretty printing.

    Args:
        array: The numpy array to format.
        spaces: Number of spaces to indent each line.

    Returns:
        Indented string representation of the array.
    """
    prefix = " " * spaces
    return "\n".join(prefix + line for line in str(array).split("\n"))


# =========================================================================
# RUNNABLE DEMO
# =========================================================================

if __name__ == "__main__":
    # Run all demos:
    #   python -m agentexplorr.foundations.linear_algebra_essentials
    #
    # Or import and run individually:
    #   from agentexplorr.foundations.linear_algebra_essentials import (
    #       LinearAlgebraTeacher,
    #   )
    #   LinearAlgebraTeacher.svd_and_lora()

    LinearAlgebraTeacher.run_all_demos()
