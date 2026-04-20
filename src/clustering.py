"""
Clustering module for malicious client detection.
Implements two-cluster grouping based on cosine similarity of gradients.
"""

import numpy as np
from typing import Tuple, List
from sklearn.metrics.pairwise import cosine_similarity


class TwoMeansClustering:
    """
    Two-cluster grouping based on gradient similarity.
    Uses cosine similarity to group clients into benign and suspicious clusters.
    """

    def __init__(self, convergence_steps: int = 10, seed: int = 42):
        """
        Initialize TwoMeansClustering.

        Args:
            convergence_steps: Maximum iterations for clustering convergence
            seed: Random seed for reproducibility
        """
        self.convergence_steps = convergence_steps
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def cluster(
        self, gradient_vectors: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        Perform two-means clustering on gradient vectors.

        Args:
            gradient_vectors: Array of shape (num_clients, gradient_dim)
                            Each row is the gradient vector of a client

        Returns:
            Tuple of:
            - cluster_assignments: Array of cluster IDs (0 or 1) for each client
            - similarity_matrix: Client-by-client cosine similarity matrix
            - avg_similarity_cluster_0: Average within-cluster similarity for cluster 0
            - avg_similarity_cluster_1: Average within-cluster similarity for cluster 1
        """
        num_clients = gradient_vectors.shape[0]

        # Compute cosine similarity matrix
        similarity_matrix = cosine_similarity(gradient_vectors)
        np.fill_diagonal(similarity_matrix, 0)  # Ignore self-similarity

        # Initialize centroids: pair with smallest similarity (largest angle)
        # Find the pair of clients with smallest cosine similarity
        min_similarity = np.inf
        best_pair = (0, 1)

        for i in range(num_clients):
            for j in range(i + 1, num_clients):
                if similarity_matrix[i, j] < min_similarity:
                    min_similarity = similarity_matrix[i, j]
                    best_pair = (i, j)

        # Initial clustering: assign clients to one of the two starting points
        initial_assignments = np.zeros(num_clients, dtype=int)
        initial_assignments[best_pair[0]] = 0
        initial_assignments[best_pair[1]] = 1

        # K-means iterations
        cluster_assignments = initial_assignments.copy()

        for iteration in range(self.convergence_steps):
            old_assignments = cluster_assignments.copy()

            # Compute centroids as average gradient vectors
            centroid_0 = gradient_vectors[cluster_assignments == 0].mean(axis=0)
            centroid_1 = gradient_vectors[cluster_assignments == 1].mean(axis=0)

            # Assign each client to nearest centroid
            dist_0 = np.linalg.norm(gradient_vectors - centroid_0, axis=1)
            dist_1 = np.linalg.norm(gradient_vectors - centroid_1, axis=1)

            cluster_assignments = (dist_1 < dist_0).astype(int)

            # Check convergence
            if np.array_equal(old_assignments, cluster_assignments):
                break

        # Compute average within-cluster similarity
        avg_similarity_cluster_0 = self._compute_avg_cluster_similarity(
            similarity_matrix, cluster_assignments, 0
        )
        avg_similarity_cluster_1 = self._compute_avg_cluster_similarity(
            similarity_matrix, cluster_assignments, 1
        )

        return cluster_assignments, similarity_matrix, avg_similarity_cluster_0, avg_similarity_cluster_1

    def _compute_avg_cluster_similarity(
        self, similarity_matrix: np.ndarray, cluster_assignments: np.ndarray, cluster_id: int
    ) -> float:
        """
        Compute average within-cluster cosine similarity.

        Args:
            similarity_matrix: Client-by-client cosine similarity matrix
            cluster_assignments: Cluster assignment for each client
            cluster_id: ID of the cluster (0 or 1)

        Returns:
            Average within-cluster similarity
        """
        cluster_members = np.where(cluster_assignments == cluster_id)[0]

        if len(cluster_members) <= 1:
            return 0.0

        # Extract submatrix for this cluster
        cluster_similarity = similarity_matrix[np.ix_(cluster_members, cluster_members)]

        # Compute average (excluding diagonal)
        num_pairs = len(cluster_members)
        total_similarity = np.sum(cluster_similarity)
        num_comparisons = num_pairs * (num_pairs - 1)

        if num_comparisons == 0:
            return 0.0

        avg_similarity = total_similarity / num_comparisons

        return avg_similarity

    def interpret_clusters(
        self,
        cluster_assignments: np.ndarray,
        avg_similarity_cluster_0: float,
        avg_similarity_cluster_1: float,
    ) -> Tuple[np.ndarray, int]:
        """
        Interpret which cluster is benign and which is suspicious.

        Args:
            cluster_assignments: Cluster assignment for each client
            avg_similarity_cluster_0: Average similarity in cluster 0
            avg_similarity_cluster_1: Average similarity in cluster 1

        Returns:
            Tuple of:
            - binary_membership: Binary vector indicating suspicious (1) or benign (0)
            - benign_cluster_id: ID of the cluster deemed benign (higher similarity)
        """
        # Cluster with higher similarity is more likely benign
        if avg_similarity_cluster_0 >= avg_similarity_cluster_1:
            benign_cluster_id = 0
            suspicious_cluster_id = 1
        else:
            benign_cluster_id = 1
            suspicious_cluster_id = 0

        # Create binary membership vector (1 = suspicious, 0 = benign)
        binary_membership = (cluster_assignments == suspicious_cluster_id).astype(int)

        return binary_membership, benign_cluster_id
