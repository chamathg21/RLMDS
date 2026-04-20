"""
RL-based malicious client detector using Q-learning.
"""

import numpy as np
from typing import Dict, Set, Tuple


class RLDetector:
    """
    Reinforcement learning based detector for malicious clients.
    Uses Q-learning with two states (suspicious, benign) to track client suspicion over time.
    """

    def __init__(
        self,
        num_clients: int,
        num_states: int = 2,
        num_actions: int = 2,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        exploration_rate: float = 0.1,
        reward_correct: float = 1.0,
        reward_penalty: float = -0.5,
        q_threshold: float = 0.5,
        seed: int = 42,
    ):
        """
        Initialize RLDetector.

        Args:
            num_clients: Number of clients in the federation
            num_states: Number of RL states (2: suspicious, benign)
            num_actions: Number of RL actions (2: stay, transition)
            learning_rate: Q-learning alpha parameter
            discount_factor: Q-learning gamma parameter
            exploration_rate: Epsilon-greedy exploration rate
            reward_correct: Reward for correct classification
            reward_penalty: Penalty for incorrect classification
            q_threshold: Threshold for classifying as malicious
            seed: Random seed for reproducibility
        """
        self.num_clients = num_clients
        self.num_states = num_states
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.reward_correct = reward_correct
        self.reward_penalty = reward_penalty
        self.q_threshold = q_threshold
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # Initialize Q-tables for each client
        # Q[client_id][state, action]
        self.q_tables = {}
        for client_id in range(num_clients):
            self.q_tables[client_id] = np.zeros((num_states, num_actions))

        # Track current state for each client (1 = suspicious, 0 = benign)
        self.client_states = np.zeros(num_clients, dtype=int)

        # Track detected malicious clients
        self.detected_malicious = set()

        # History of states for analysis
        self.state_history = []

    def update(
        self,
        binary_membership: np.ndarray,
        true_labels: np.ndarray,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Update Q-values based on clustering feedback.

        Args:
            binary_membership: Binary vector from clustering (1=suspicious, 0=benign)
            true_labels: Ground truth labels (1=malicious, 0=benign)

        Returns:
            Tuple of:
            - detection_labels: Current detection classification for each client
            - metrics: Dictionary with Q-value statistics
        """
        metrics = {
            "avg_q_value": [],
            "avg_q_suspicious": [],
            "avg_q_benign": [],
        }

        detection_labels = np.zeros(self.num_clients, dtype=int)

        for client_id in range(self.num_clients):
            # Skip already detected clients
            if client_id in self.detected_malicious:
                detection_labels[client_id] = 1
                continue

            current_state = self.client_states[client_id]
            observation = binary_membership[client_id]  # 1=suspicious, 0=benign

            # Select action using epsilon-greedy
            if self.rng.random() < self.exploration_rate:
                action = self.rng.randint(0, self.num_actions)
            else:
                action = np.argmax(self.q_tables[client_id][current_state])

            # Compute reward based on observation
            if observation == 1:  # Cluster labeled as suspicious
                reward = self.reward_penalty  # Penalize being in suspicious cluster
            else:  # Cluster labeled as benign
                reward = self.reward_correct  # Reward being in benign cluster

            # Compute next state (action: 0=stay, 1=transition)
            next_state = (current_state + action) % self.num_states

            # Q-learning update
            max_q_next = np.max(self.q_tables[client_id][next_state])
            old_q = self.q_tables[client_id][current_state, action]

            self.q_tables[client_id][current_state, action] = old_q + self.learning_rate * (
                reward + self.discount_factor * max_q_next - old_q
            )

            # Update state for next round
            self.client_states[client_id] = next_state

            # Compute detection label based on Q-values
            # Higher Q-values in suspicious state indicate maliciousness
            q_diff = (
                self.q_tables[client_id][1, 0] - self.q_tables[client_id][0, 0]
            )  # Q(suspicious) - Q(benign)

            if q_diff > self.q_threshold:
                detection_labels[client_id] = 1
                self.detected_malicious.add(client_id)
            else:
                detection_labels[client_id] = 0

            # Track metrics
            avg_q = np.mean(self.q_tables[client_id])
            q_suspicious = np.mean(self.q_tables[client_id][1])
            q_benign = np.mean(self.q_tables[client_id][0])

            metrics["avg_q_value"].append(avg_q)
            metrics["avg_q_suspicious"].append(q_suspicious)
            metrics["avg_q_benign"].append(q_benign)

        # Compute aggregate metrics
        metrics["avg_q_value"] = np.mean(metrics["avg_q_value"])
        metrics["avg_q_suspicious"] = np.mean(metrics["avg_q_suspicious"])
        metrics["avg_q_benign"] = np.mean(metrics["avg_q_benign"])

        self.state_history.append(self.client_states.copy())

        return detection_labels, metrics

    def get_detected_clients(self) -> Set[int]:
        """Get the set of detected malicious clients."""
        return self.detected_malicious.copy()

    def reset_detection(self):
        """Reset detected malicious clients (if needed for new experiments)."""
        self.detected_malicious = set()

    def get_q_table(self, client_id: int) -> np.ndarray:
        """Get the Q-table for a specific client."""
        return self.q_tables[client_id].copy()

    def get_client_state(self, client_id: int) -> int:
        """Get the current state of a specific client."""
        return int(self.client_states[client_id])

    def get_state_history(self) -> np.ndarray:
        """Get the history of client states over rounds."""
        return np.array(self.state_history)
