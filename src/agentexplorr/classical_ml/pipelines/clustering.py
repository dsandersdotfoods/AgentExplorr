"""
Clustering Pipeline
====================

An educational clustering pipeline demonstrating unsupervised learning with
KMeans and DBSCAN, including automatic selection of the optimal number of
clusters and PCA-based visualization.

WHAT IS UNSUPERVISED LEARNING?
    In supervised learning (classification, regression), we have labeled data:
    inputs X paired with known outputs y. The model learns the mapping X → y.

    In UNSUPERVISED learning, we have ONLY inputs X — no labels, no "right answers."
    The model must discover structure in the data on its own:
      - Clustering: find groups of similar data points
      - Dimensionality reduction: compress data while preserving structure (PCA, t-SNE)
      - Anomaly detection: find unusual data points
      - Generative models: learn the data distribution (VAEs, GANs)

    Real-world examples:
      - Customer segmentation (group customers by purchasing behavior)
      - Document topic discovery (group documents by content)
      - Image compression (group similar pixels)
      - Anomaly detection in network traffic

KMEANS vs DBSCAN:

    KMEANS:
      How: Assign each point to the nearest of K centroids, then re-compute
           centroids. Repeat until convergence.
      Pros: Fast, simple, works well with spherical clusters
      Cons: Must specify K in advance; assumes clusters are spherical and
            equal-sized; sensitive to outliers
      When: You have a rough idea of how many clusters exist

    DBSCAN (Density-Based Spatial Clustering of Applications with Noise):
      How: Core points have ≥ min_samples neighbors within eps distance.
           Clusters are connected regions of core points. Points not near
           any core point are labeled as noise (-1).
      Pros: Discovers arbitrary-shaped clusters; doesn't need K; finds outliers
      Cons: Sensitive to eps and min_samples; struggles with varying densities
      When: Clusters have irregular shapes; you want outlier detection

    ┌──────────────────────────────────────────────────────┐
    │  KMeans (K=3)          │  DBSCAN                     │
    │                        │                              │
    │    ●●●                 │    ●●●                      │
    │   ●●●●●   ▲▲▲         │   ●●●●●   ▲▲▲▲             │
    │    ●●●   ▲▲▲▲▲        │    ●●●   ▲▲▲▲▲▲            │
    │          ▲▲▲           │          ▲▲▲▲               │
    │  ■■■■                  │  ■■■■      × (noise)       │
    │ ■■■■■■                 │ ■■■■■■                      │
    │  ■■■■                  │  ■■■■                       │
    │                        │                              │
    │ (forced into 3 circles)│ (finds actual shapes)       │
    └──────────────────────────────────────────────────────┘

EVALUATION: SILHOUETTE SCORE
    Since there are no labels, we can't use accuracy/F1. Instead, we use
    internal metrics that measure cluster quality:

    Silhouette Score = (b - a) / max(a, b)
      - a = mean distance from a point to other points in its cluster (cohesion)
      - b = mean distance to the nearest neighboring cluster (separation)
      - Range: [-1, 1]
        - +1: dense, well-separated clusters (perfect)
        -  0: overlapping clusters
        - -1: points assigned to wrong clusters

    The silhouette score is analogous to asking: "Are the clusters tight
    internally and well-separated from each other?"

DATASET: Iris (sklearn.datasets.load_iris)
    - 150 samples, 4 features (sepal/petal length/width), 3 species
    - We IGNORE the labels (pretend we don't know the species) and let
      clustering discover the natural groupings.
    - Then we can compare discovered clusters vs actual species to validate.

LEARNING RESOURCES:
    - sklearn Clustering Guide: https://scikit-learn.org/stable/modules/clustering.html
    - KMeans: https://scikit-learn.org/stable/modules/clustering.html#k-means
    - DBSCAN: https://scikit-learn.org/stable/modules/clustering.html#dbscan
    - Silhouette Score: https://scikit-learn.org/stable/modules/clustering.html#silhouette-coefficient
    - PCA: https://scikit-learn.org/stable/modules/decomposition.html#pca
    - VIDEO: "KMeans Clustering" (StatQuest) — https://www.youtube.com/watch?v=4b5d3muPQmA
    - VIDEO: "DBSCAN Clearly Explained" — https://www.youtube.com/watch?v=RDZUdRSDOok
    - VIDEO: "Silhouette Score Explained" — https://www.youtube.com/watch?v=mtkWR7UleRQ
    - VIDEO: "PCA Main Ideas" (StatQuest) — https://www.youtube.com/watch?v=HMOI_lkzW08
    - PAPER: Lloyd (1982) "Least Squares Quantization in PCM" (original KMeans)
    - PAPER: Ester et al. (1996) "A Density-Based Algorithm for Discovering Clusters"
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from agentexplorr.core import get_logger

logger = get_logger(__name__)

# Type alias for supported clustering algorithms
ClusterAlgorithm = Literal["kmeans", "dbscan"]


class ClusteringPipeline:
    """End-to-end clustering pipeline with evaluation and PCA visualization.

    This pipeline supports both KMeans and DBSCAN clustering, includes an
    elbow method for optimal K selection, silhouette score evaluation, and
    PCA-based visualization of the discovered clusters.

    Attributes:
        algorithm: Which clustering algorithm is used ("kmeans" or "dbscan").
        pipeline: The sklearn Pipeline (scaler + clusterer).
        labels: Cluster assignments after fitting.
        results: Dictionary of evaluation metrics.
        pca_data: 2D PCA projection for visualization.

    Example:
        >>> clust = ClusteringPipeline(algorithm="kmeans", n_clusters=3)
        >>> X, y_true = clust.load_data()
        >>> clust.build_pipeline()
        >>> clust.fit(X)
        >>> results = clust.evaluate(X)
        >>> print(f"Silhouette Score: {results['silhouette_score']:.4f}")
        >>> optimal_k = clust.elbow_method(X, k_range=range(2, 11))
    """

    def __init__(
        self,
        algorithm: ClusterAlgorithm = "kmeans",
        n_clusters: int = 3,
        eps: float = 0.5,
        min_samples: int = 5,
        random_state: int = 42,
    ) -> None:
        """Initialize the clustering pipeline.

        Args:
            algorithm: Clustering algorithm to use.
            n_clusters: Number of clusters for KMeans. Ignored for DBSCAN.
                KMeans REQUIRES you to specify K in advance. Use the elbow
                method (below) to find a good value.
            eps: Maximum distance between two samples to be considered neighbors
                (DBSCAN only). Think of it as the "radius" of each point's
                neighborhood. Too small → everything is noise. Too large →
                everything is one cluster.
            min_samples: Minimum number of points in a neighborhood to form a
                dense region (DBSCAN only). Higher values → more conservative
                clustering (fewer, larger clusters). Lower values → more
                clusters, including tiny ones.
            random_state: Seed for reproducibility (KMeans only).
        """
        self.algorithm: ClusterAlgorithm = algorithm
        self.n_clusters: int = n_clusters
        self.eps: float = eps
        self.min_samples: int = min_samples
        self.random_state: int = random_state
        self.pipeline: Pipeline | None = None
        self.labels: np.ndarray | None = None
        self.results: dict[str, Any] = {}
        self.pca_data: np.ndarray | None = None
        self.feature_names: list[str] = []
        self.target_names: list[str] = []

        logger.info(
            "clustering_pipeline_initialized",
            algorithm=algorithm,
            n_clusters=n_clusters if algorithm == "kmeans" else "N/A",
            eps=eps if algorithm == "dbscan" else "N/A",
            min_samples=min_samples if algorithm == "dbscan" else "N/A",
        )

    def load_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Load the Iris dataset for clustering experiments.

        NOTE: We return the true labels (y) ONLY for validation — to see how
        well our unsupervised clusters match the actual species. In a real
        unsupervised scenario, you wouldn't have these labels.

        Returns:
            Tuple of (X, y_true):
              - X: Feature matrix (150 samples x 4 features)
              - y_true: True species labels (for comparison only, NOT used for training)
        """
        iris = load_iris()
        self.feature_names = list(iris.feature_names)
        self.target_names = list(iris.target_names)

        logger.info(
            "data_loaded",
            dataset="iris",
            n_samples=len(iris.data),
            n_features=len(self.feature_names),
            n_true_clusters=len(self.target_names),
        )

        return iris.data, iris.target

    def build_pipeline(self) -> Pipeline:
        """Build the preprocessing + clustering pipeline.

        WHY SCALE BEFORE CLUSTERING?
            Both KMeans and DBSCAN use distance metrics (Euclidean by default).
            If features have different scales, the high-scale features dominate
            the distance calculation.

            Example: petal length (1-7 cm) vs sepal width (2-4.5 cm).
            Without scaling, petal length differences contribute more to distance,
            so clusters would be determined mostly by petal length alone.

        Returns:
            The constructed sklearn Pipeline.
        """
        # Build the appropriate clustering model
        if self.algorithm == "kmeans":
            clusterer = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                # n_init=10: KMeans is sensitive to initial centroid placement.
                # It runs 10 times with different initializations and picks
                # the best result (lowest inertia). "auto" in newer sklearn
                # versions adaptively chooses based on the init method.
                n_init=10,
                # init="k-means++": smart initialization that spreads initial
                # centroids apart. Much better than random initialization.
                # This is the default, but worth highlighting because it's
                # one of the most impactful improvements to the algorithm.
                init="k-means++",
                # max_iter: maximum iterations of the assign-recompute loop.
                # Usually converges in 10-20 iterations.
                max_iter=300,
            )
        elif self.algorithm == "dbscan":
            clusterer = DBSCAN(
                eps=self.eps,
                min_samples=self.min_samples,
                # metric="euclidean" is the default distance metric.
                # Other options: "manhattan", "cosine", etc.
                metric="euclidean",
                # n_jobs=-1: parallelize distance computations
                n_jobs=-1,
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}. Use 'kmeans' or 'dbscan'.")

        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clusterer", clusterer),
            ]
        )

        logger.info("pipeline_built", algorithm=self.algorithm)

        return self.pipeline

    def fit(self, X: np.ndarray) -> np.ndarray:
        """Fit the clustering pipeline and return cluster labels.

        HOW KMEANS WORKS (step by step):
            1. Initialize K centroids (using k-means++)
            2. ASSIGN: assign each point to the nearest centroid
            3. UPDATE: recompute centroids as the mean of assigned points
            4. Repeat steps 2-3 until centroids stop moving (convergence)

            The algorithm minimizes "inertia" = sum of squared distances
            from each point to its assigned centroid.

        HOW DBSCAN WORKS (step by step):
            1. Pick an unvisited point P
            2. Find all neighbors within eps distance
            3. If #neighbors >= min_samples, P is a "core point" → start a cluster
            4. Expand the cluster by repeating for each core point's neighbors
            5. Points not reachable from any core point → noise (label = -1)

        Args:
            X: Feature matrix (n_samples x n_features).

        Returns:
            Array of cluster labels (integers). For DBSCAN, -1 = noise.
        """
        if self.pipeline is None:
            self.build_pipeline()

        assert self.pipeline is not None

        # Pipeline doesn't support fit_predict directly through the pipeline
        # so we scale first, then cluster.
        # Step 1: Scale the data
        scaler = self.pipeline.named_steps["scaler"]
        X_scaled = scaler.fit_transform(X)

        # Step 2: Cluster the scaled data
        clusterer = self.pipeline.named_steps["clusterer"]
        self.labels = clusterer.fit_predict(X_scaled)

        # Count cluster sizes
        unique_labels = set(self.labels)
        n_clusters = len(unique_labels - {-1})  # Exclude noise label (-1)
        n_noise = int(np.sum(self.labels == -1))

        logger.info(
            "clustering_completed",
            algorithm=self.algorithm,
            n_clusters=n_clusters,
            n_noise=n_noise,
            cluster_sizes={
                str(label): int(np.sum(self.labels == label)) for label in sorted(unique_labels)
            },
        )

        return self.labels

    def evaluate(self, X: np.ndarray) -> dict[str, Any]:
        """Evaluate clustering quality using silhouette score and inertia.

        SILHOUETTE SCORE EXPLAINED:
            For each data point i:
              a(i) = mean distance to all OTHER points in the same cluster
                     (measures how tight/cohesive the cluster is)
              b(i) = mean distance to all points in the NEAREST other cluster
                     (measures how well-separated clusters are)

              silhouette(i) = (b(i) - a(i)) / max(a(i), b(i))

            The score ranges from -1 to +1:
              +1: point is well-matched to its cluster, far from others (ideal)
               0: point is on the boundary between clusters
              -1: point is likely in the wrong cluster

            The overall silhouette score is the mean across all points.

            RULE OF THUMB:
              > 0.7: strong clustering
              0.5 - 0.7: reasonable clustering
              0.25 - 0.5: weak clustering, consider different K or algorithm
              < 0.25: no substantial clustering structure found

        INERTIA (KMeans only):
            Sum of squared distances from each point to its assigned centroid.
            Lower is better, but ALWAYS decreases as K increases (more clusters
            = each point is closer to a centroid). That's why we use the
            elbow method — not raw inertia — to choose K.

        Args:
            X: Feature matrix (must be the same data used for fitting).

        Returns:
            Dictionary with silhouette_score, inertia (KMeans), and cluster stats.
        """
        if self.labels is None:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")

        assert self.pipeline is not None
        scaler = self.pipeline.named_steps["scaler"]
        X_scaled = scaler.transform(X)

        # Silhouette score requires at least 2 clusters
        unique_labels = set(self.labels) - {-1}  # Exclude noise

        if len(unique_labels) < 2:
            logger.warning(
                "insufficient_clusters_for_silhouette",
                n_clusters=len(unique_labels),
            )
            sil_score = -1.0
        else:
            # Filter out noise points for silhouette calculation
            mask = self.labels != -1
            if np.sum(mask) < 2:
                sil_score = -1.0
            else:
                sil_score = float(silhouette_score(X_scaled[mask], self.labels[mask]))

        self.results = {
            "silhouette_score": sil_score,
            "n_clusters": len(unique_labels),
            "n_noise_points": int(np.sum(self.labels == -1)),
            "cluster_sizes": {
                int(label): int(np.sum(self.labels == label)) for label in sorted(unique_labels)
            },
        }

        # Add inertia for KMeans (sum of squared distances to centroids)
        if self.algorithm == "kmeans":
            clusterer = self.pipeline.named_steps["clusterer"]
            self.results["inertia"] = float(clusterer.inertia_)

        logger.info(
            "evaluation_completed",
            silhouette_score=round(sil_score, 4),
            n_clusters=len(unique_labels),
        )

        # Pretty-print results
        print("\n" + "=" * 60)
        print(f"CLUSTERING RESULTS ({self.algorithm.upper()})")
        print("=" * 60)
        print(f"Number of Clusters: {len(unique_labels)}")
        print(f"Silhouette Score:   {sil_score:.4f}")
        if self.algorithm == "kmeans":
            print(f"Inertia:            {self.results['inertia']:.2f}")
        print(f"Noise Points:       {self.results['n_noise_points']}")
        print("\nCluster Sizes:")
        for label in sorted(unique_labels):
            size = int(np.sum(self.labels == label))
            print(f"  Cluster {label}: {size} points")
        print("=" * 60)

        return self.results

    def fit_on_iris(self) -> dict[str, Any]:
        """Convenience method: load Iris dataset, fit clusters, and evaluate.

        Returns:
            Dictionary of evaluation metrics including "silhouette_score".
        """
        X, _y_true = self.load_data()
        self.build_pipeline()
        self.fit(X)
        return self.evaluate(X)

    def elbow_method(
        self,
        X: np.ndarray,
        k_range: range | None = None,
    ) -> dict[str, Any]:
        """Run the elbow method to find the optimal number of clusters for KMeans.

        THE ELBOW METHOD:
            Plot inertia (y-axis) vs K (x-axis). As K increases, inertia
            ALWAYS decreases (more clusters = each point is closer to a centroid).

            The "elbow" is the point where adding more clusters gives diminishing
            returns — the curve bends like an elbow.

            Inertia
            ▲
            │╲
            │ ╲
            │  ╲
            │   ╲
            │    ╲_____  ← Elbow (optimal K)
            │          ╲______
            │                 ╲_____
            │──────────────────────────► K
            2  3  4  5  6  7  8  9  10

            WHY NOT JUST USE K=N (one cluster per point)?
            Because that's just the original data — no simplification.
            We want the FEWEST clusters that still capture meaningful structure.

        IMPLEMENTATION DETAIL:
            We also compute silhouette scores alongside inertia. Sometimes
            the elbow isn't obvious, but the silhouette score has a clear peak.

        Args:
            X: Feature matrix.
            k_range: Range of K values to try. Default: 2 to 10.

        Returns:
            Dictionary with:
              - k_values: list of K values tested
              - inertias: corresponding inertia values
              - silhouette_scores: corresponding silhouette scores
              - optimal_k: the K with the highest silhouette score
        """
        if k_range is None:
            k_range = range(2, 11)

        # Scale the data once (same scaler for all K values)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        k_values: list[int] = []
        inertias: list[float] = []
        silhouette_scores: list[float] = []

        print("\n" + "=" * 60)
        print("ELBOW METHOD: Finding Optimal K")
        print("=" * 60)
        print(f"{'K':>5} | {'Inertia':>12} | {'Silhouette':>12}")
        print("-" * 35)

        for k in k_range:
            # Fit KMeans with this K value
            kmeans = KMeans(
                n_clusters=k,
                random_state=self.random_state,
                n_init=10,
                init="k-means++",
            )
            labels = kmeans.fit_predict(X_scaled)

            # Record inertia and silhouette score
            inertia = float(kmeans.inertia_)
            sil_score = float(silhouette_score(X_scaled, labels))

            k_values.append(k)
            inertias.append(inertia)
            silhouette_scores.append(sil_score)

            print(f"{k:>5} | {inertia:>12.2f} | {sil_score:>12.4f}")

        # The optimal K is the one with the highest silhouette score
        best_idx = int(np.argmax(silhouette_scores))
        optimal_k = k_values[best_idx]

        print("-" * 35)
        print(f"Optimal K = {optimal_k} (Silhouette = {silhouette_scores[best_idx]:.4f})")
        print("=" * 60)

        logger.info(
            "elbow_method_completed",
            optimal_k=optimal_k,
            best_silhouette=round(silhouette_scores[best_idx], 4),
        )

        return {
            "k_values": k_values,
            "inertias": inertias,
            "silhouette_scores": silhouette_scores,
            "optimal_k": optimal_k,
        }

    def pca_reduce(self, X: np.ndarray, n_components: int = 2) -> np.ndarray:
        """Reduce data to 2D using PCA for visualization.

        WHAT IS PCA (Principal Component Analysis)?
            PCA finds the directions (axes) in the data that capture the
            most variance. It projects high-dimensional data onto these
            axes, preserving as much information as possible.

            Think of it like casting a shadow: if you have a 3D object,
            PCA finds the angle that casts the most informative 2D shadow.

            PC1 (first principal component) = direction of maximum variance
            PC2 (second principal component) = direction of second-most variance
                                               (orthogonal to PC1)

            For visualization, we project 4D Iris data → 2D (PC1 vs PC2).
            This lets us see cluster structure on a flat plot, even though
            the actual clustering happened in 4D.

            IMPORTANT: PCA can distort clusters! Two clusters that overlap in
            2D might be well-separated in the original 4D space. The 2D plot
            is an approximation — not the truth.

        Args:
            X: Feature matrix (n_samples x n_features).
            n_components: Number of dimensions to reduce to (2 for 2D plot).

        Returns:
            Reduced data matrix (n_samples x n_components).
        """
        # Scale first (PCA is sensitive to feature scale, just like clustering)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Fit PCA and transform
        pca = PCA(n_components=n_components)
        self.pca_data = pca.fit_transform(X_scaled)

        # explained_variance_ratio_ tells us how much information (variance)
        # each component captures. If PC1 captures 72% and PC2 captures 23%,
        # our 2D plot preserves 95% of the information. That's excellent!
        explained_var = pca.explained_variance_ratio_
        total_explained = float(np.sum(explained_var))

        logger.info(
            "pca_reduction_completed",
            n_components=n_components,
            explained_variance_per_component=[round(float(v), 4) for v in explained_var],
            total_explained_variance=round(total_explained, 4),
        )

        print(f"\nPCA: {n_components} components explain {total_explained * 100:.1f}% of variance")
        for i, var in enumerate(explained_var):
            print(f"  PC{i + 1}: {var * 100:.1f}%")

        return self.pca_data

    def get_cluster_centers(self) -> np.ndarray | None:
        """Get cluster centers (KMeans only).

        Cluster centers are the centroids — the "average point" of each cluster
        in the scaled feature space. They tell you what a "typical" member of
        each cluster looks like.

        Returns:
            Array of shape (n_clusters, n_features) or None for DBSCAN.
        """
        if self.algorithm != "kmeans":
            logger.warning("cluster_centers_only_available_for_kmeans")
            return None

        if self.pipeline is None:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")

        clusterer = self.pipeline.named_steps["clusterer"]
        centers: np.ndarray = clusterer.cluster_centers_

        logger.info(
            "cluster_centers_extracted",
            n_clusters=len(centers),
            center_dims=centers.shape,
        )

        return centers


# ---------------------------------------------------------------------------
# Quick demo — run this file directly to see clustering in action
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("CLUSTERING PIPELINE DEMO")
    print("Dataset: Iris (pretending we don't know the labels)")
    print("Algorithms: KMeans, DBSCAN")
    print("=" * 60)

    # --- KMeans ---
    print("\n--- KMEANS CLUSTERING ---")
    kmeans_pipe = ClusteringPipeline(algorithm="kmeans", n_clusters=3)
    X, y_true = kmeans_pipe.load_data()
    kmeans_pipe.build_pipeline()
    labels = kmeans_pipe.fit(X)
    kmeans_pipe.evaluate(X)

    # Run elbow method to find optimal K
    elbow_results = kmeans_pipe.elbow_method(X, k_range=range(2, 11))

    # PCA visualization data
    pca_2d = kmeans_pipe.pca_reduce(X, n_components=2)
    print(f"\nPCA reduced shape: {pca_2d.shape}")

    # Compare clusters to true labels
    print("\nCluster vs True Label comparison:")
    for cluster_id in range(3):
        cluster_mask = labels == cluster_id
        true_labels_in_cluster = y_true[cluster_mask]
        from collections import Counter

        counts = Counter(true_labels_in_cluster)
        print(f"  Cluster {cluster_id}: {dict(counts)}")

    # --- DBSCAN ---
    print("\n\n--- DBSCAN CLUSTERING ---")
    dbscan_pipe = ClusteringPipeline(algorithm="dbscan", eps=0.8, min_samples=5)
    dbscan_pipe.feature_names = kmeans_pipe.feature_names
    dbscan_pipe.target_names = kmeans_pipe.target_names
    dbscan_pipe.build_pipeline()
    dbscan_labels = dbscan_pipe.fit(X)
    dbscan_pipe.evaluate(X)
