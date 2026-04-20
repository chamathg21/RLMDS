"""
Federated learning server implementation.
Handles global aggregation and malicious client detection.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from torch.utils.data import DataLoader

from .clustering import TwoMeansClustering
from .rl_detector import RLDetector


class Server:
    """
    Federated learning server.
    Aggregates client updates and performs malicious client detection.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        clustering_config: dict,
        rl_config: dict,
        test_loader: DataLoader,
    ):
        """
        Initialize Server.

        Args:
            model: Global model
            device: PyTorch device
            clustering_config: Configuration for clustering
            rl_config: Configuration for RL detector
            test_loader: DataLoader for test set (for evaluation)
        """
        self.model = model
        self.device = device
        self.test_loader = test_loader
        self.loss_fn = nn.CrossEntropyLoss()

        # Initialize clustering and detection modules
        self.clustering = TwoMeansClustering(
            convergence_steps=clustering_config.get("convergence_steps", 10),
            seed=clustering_config.get("seed", 42),
        )

        self.rl_detector = RLDetector(
            num_clients=rl_config.get("num_clients", 50),
            num_states=rl_config.get("num_states", 2),
            num_actions=rl_config.get("num_actions", 2),
            learning_rate=rl_config.get("learning_rate", 0.1),
            discount_factor=rl_config.get("discount_factor", 0.9),
            exploration_rate=rl_config.get("exploration_rate", 0.1),
            reward_correct=rl_config.get("reward_correct", 1.0),
            reward_penalty=rl_config.get("reward_penalty", -0.5),
            q_threshold=rl_config.get("q_threshold", 0.5),
            seed=rl_config.get("seed", 42),
        )

        # Track excluded clients
        self.excluded_clients = set()

    def aggregate_weights(
        self, client_weights: Dict[int, dict], participating_clients: Optional[List[int]] = None
    ) -> dict:
        """
        Aggregate weights using FedAvg algorithm.

        Args:
            client_weights: Dictionary mapping client_id to their weights
            participating_clients: List of clients to include (None = all except excluded)

        Returns:
            Aggregated global weights
        """
        if participating_clients is None:
            participating_clients = [
                cid for cid in client_weights.keys() if cid not in self.excluded_clients
            ]

        if not participating_clients:
            # Return current model weights if no valid clients
            return {name: param.data.clone() for name, param in self.model.named_parameters()}

        # Compute average weights
        aggregated_weights = {}
        num_clients = len(participating_clients)

        for client_id in participating_clients:
            if client_id not in client_weights:
                continue

            weights = client_weights[client_id]

            for param_name, param_value in weights.items():
                if param_name not in aggregated_weights:
                    aggregated_weights[param_name] = torch.zeros_like(param_value)

                aggregated_weights[param_name] += param_value / num_clients

        return aggregated_weights

    def set_global_weights(self, weights: dict):
        """Set the global model weights."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in weights:
                    param.copy_(weights[name])

    def detect_malicious_clients(
        self, gradient_vectors: Dict[int, np.ndarray], true_labels: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, dict, dict]:
        """
        Detect malicious clients using clustering and RL.

        Args:
            gradient_vectors: Dictionary mapping client_id to gradient vectors
            true_labels: Ground truth labels (1=malicious, 0=benign) for evaluation

        Returns:
            Tuple of:
            - detection_labels: Detection results for all clients (1=malicious, 0=benign)
            - clustering_stats: Dictionary with clustering statistics
            - rl_metrics: Dictionary with RL metrics
        """
        num_clients = len(gradient_vectors)

        # Stack gradient vectors (in client ID order)
        client_ids = sorted(gradient_vectors.keys())
        gradient_array = np.array([gradient_vectors[cid] for cid in client_ids])

        # Perform clustering
        cluster_assignments, similarity_matrix, avg_sim_0, avg_sim_1 = self.clustering.cluster(
            gradient_array
        )

        binary_membership, benign_cluster_id = self.clustering.interpret_clusters(
            cluster_assignments, avg_sim_0, avg_sim_1
        )

        # Update RL detector
        detection_labels, rl_metrics = self.rl_detector.update(binary_membership, true_labels)

        # Map detection labels back to client IDs
        detection_dict = {client_ids[i]: detection_labels[i] for i in range(len(client_ids))}

        # Update excluded clients
        for client_id, is_malicious in detection_dict.items():
            if is_malicious:
                self.excluded_clients.add(client_id)

        # Clustering statistics
        clustering_stats = {
            "cluster_assignments": cluster_assignments,
            "binary_membership": binary_membership,
            "benign_cluster_id": benign_cluster_id,
            "avg_similarity_cluster_0": avg_sim_0,
            "avg_similarity_cluster_1": avg_sim_1,
            "similarity_matrix": similarity_matrix,
        }

        return detection_labels, clustering_stats, rl_metrics

    def evaluate_on_test_set(self) -> Tuple[float, float]:
        """
        Evaluate global model on test set.

        Returns:
            Tuple of (accuracy, loss)
        """
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0.0

        with torch.no_grad():
            for batch_x, batch_y in self.test_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                logits = self.model(batch_x)
                loss = self.loss_fn(logits, batch_y)

                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == batch_y).sum().item()
                total += batch_y.size(0)
                total_loss += loss.item()

        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / max(len(self.test_loader), 1)

        return accuracy, avg_loss

    def exclude_client(self, client_id: int):
        """Exclude a client from future aggregations."""
        self.excluded_clients.add(client_id)

    def is_client_excluded(self, client_id: int) -> bool:
        """Check if a client is excluded."""
        return client_id in self.excluded_clients

    def get_excluded_clients(self) -> Set[int]:
        """Get the set of excluded clients."""
        return self.excluded_clients.copy()

    def reset_excluded_clients(self):
        """Reset the excluded clients set."""
        self.excluded_clients = set()

    def get_rl_detector(self) -> RLDetector:
        """Get the RL detector."""
        return self.rl_detector

    def get_clustering(self) -> TwoMeansClustering:
        """Get the clustering module."""
        return self.clustering
