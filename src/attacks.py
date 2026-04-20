"""
Attack implementations for federated learning.
Supports label flipping and backdoor attacks.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from typing import Tuple, List, Set


class PoisonedDataset(Dataset):
    """
    Wrapper dataset that applies poisoning to a subset of samples.
    """

    def __init__(self, base_dataset, poison_indices: np.ndarray, attack_type: str = "labelflip",
                 target_class: int = 0, trigger_size: int = 4, seed: int = 42):
        """
        Initialize PoisonedDataset.

        Args:
            base_dataset: The original dataset to poison
            poison_indices: Indices of samples to poison
            attack_type: Type of attack ("labelflip" or "backdoor")
            target_class: Target class for backdoor attack or label flip
            trigger_size: Size of trigger pattern for backdoor
            seed: Random seed for reproducibility
        """
        self.base_dataset = base_dataset
        self.poison_indices = set(poison_indices)
        self.attack_type = attack_type.lower()
        self.target_class = target_class
        self.trigger_size = trigger_size
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get item with potential poisoning applied.

        Args:
            idx: Index of the sample

        Returns:
            Tuple of (poisoned_image, poisoned_label)
        """
        image, label = self.base_dataset[idx]

        if idx in self.poison_indices:
            if self.attack_type == "labelflip":
                # Label flipping: flip to random incorrect class
                available_labels = [l for l in range(10) if l != label]
                label = self.rng.choice(available_labels)

            elif self.attack_type == "backdoor":
                # Backdoor attack: add trigger and change label
                image = self._add_trigger(image)
                label = self.target_class

        return image, label

    def _add_trigger(self, image: torch.Tensor) -> torch.Tensor:
        """
        Add a backdoor trigger (white square) to the image.

        Args:
            image: Input image tensor (C, H, W)

        Returns:
            Image with trigger added
        """
        image = image.clone()

        # Get image dimensions
        channels, height, width = image.shape

        # Add white square in bottom-right corner
        start_h = max(0, height - self.trigger_size - 1)
        start_w = max(0, width - self.trigger_size - 1)
        end_h = min(height, start_h + self.trigger_size)
        end_w = min(width, start_w + self.trigger_size)

        # Set trigger to white (normalized value = 1.0 for [0, 1] or normalize for [-1, 1])
        image[:, start_h:end_h, start_w:end_w] = 1.0

        return image


class AttackManager:
    """
    Manages attack application for malicious clients in federated learning.
    """

    def __init__(self, num_clients: int, num_malicious: int, seed: int = 42):
        """
        Initialize AttackManager.

        Args:
            num_clients: Total number of clients
            num_malicious: Number of malicious clients
            seed: Random seed for reproducibility
        """
        self.num_clients = num_clients
        self.num_malicious = num_malicious
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # Select which clients are malicious
        all_clients = np.arange(num_clients)
        self.malicious_clients = set(self.rng.choice(all_clients, num_malicious, replace=False))
        self.benign_clients = set(all_clients) - self.malicious_clients

    def get_malicious_clients(self) -> Set[int]:
        """Get the set of malicious client IDs."""
        return self.malicious_clients.copy()

    def get_benign_clients(self) -> Set[int]:
        """Get the set of benign client IDs."""
        return self.benign_clients.copy()

    def is_malicious(self, client_id: int) -> bool:
        """Check if a client is malicious."""
        return client_id in self.malicious_clients

    def apply_attack(
        self,
        client_dataset: Subset,
        client_id: int,
        attack_type: str,
        poisoning_rate: float,
        target_class: int = 0,
        trigger_size: int = 4,
    ) -> Dataset:
        """
        Apply attack to a client's dataset if the client is malicious.

        Args:
            client_dataset: The client's dataset
            client_id: ID of the client
            attack_type: Type of attack ("labelflip", "backdoor", "none")
            poisoning_rate: Fraction of local data to poison
            target_class: Target class for attacks
            trigger_size: Size of trigger for backdoor attack

        Returns:
            Potentially poisoned dataset
        """
        if client_id not in self.malicious_clients or attack_type == "none":
            return client_dataset

        # Select samples to poison
        num_samples = len(client_dataset)
        num_poison = max(1, int(num_samples * poisoning_rate))
        poison_indices = self.rng.choice(num_samples, num_poison, replace=False)

        # Apply poisoning
        poisoned_dataset = PoisonedDataset(
            client_dataset,
            poison_indices=poison_indices,
            attack_type=attack_type,
            target_class=target_class,
            trigger_size=trigger_size,
            seed=self.seed + client_id,  # Vary seed per client for reproducibility
        )

        return poisoned_dataset
