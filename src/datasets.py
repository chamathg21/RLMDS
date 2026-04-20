"""
Dataset handling for federated learning experiments.
Supports MNIST, FashionMNIST, and CIFAR-10.
"""

import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from typing import Tuple, List


class DataDistributor:
    """
    Distributes datasets across multiple clients in a federated learning setup.
    """

    def __init__(self, dataset_name: str, data_dir: str, num_clients: int, seed: int = 42):
        """
        Initialize the data distributor.

        Args:
            dataset_name: Name of dataset ("mnist", "fashionmnist", "cifar10")
            data_dir: Directory to store datasets
            num_clients: Number of clients to distribute data across
            seed: Random seed for reproducibility
        """
        self.dataset_name = dataset_name.lower()
        self.data_dir = data_dir
        self.num_clients = num_clients
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)

        # Load dataset
        self.train_dataset, self.test_dataset = self._load_dataset()

        # Get dataset statistics
        self.num_classes = self._get_num_classes()
        self.img_shape = self.train_dataset[0][0].shape

    def _load_dataset(self) -> Tuple:
        """Load the specified dataset from torchvision."""
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )

        if self.dataset_name == "mnist":
            train_dataset = torchvision.datasets.MNIST(
                root=self.data_dir, train=True, download=True, transform=transform
            )
            test_dataset = torchvision.datasets.MNIST(
                root=self.data_dir, train=False, download=True, transform=transform
            )
        elif self.dataset_name == "fashionmnist":
            train_dataset = torchvision.datasets.FashionMNIST(
                root=self.data_dir, train=True, download=True, transform=transform
            )
            test_dataset = torchvision.datasets.FashionMNIST(
                root=self.data_dir, train=False, download=True, transform=transform
            )
        elif self.dataset_name == "cifar10":
            # For CIFAR10, use different normalization
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            )
            train_dataset = torchvision.datasets.CIFAR10(
                root=self.data_dir, train=True, download=True, transform=transform
            )
            test_dataset = torchvision.datasets.CIFAR10(
                root=self.data_dir, train=False, download=True, transform=transform
            )
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

        return train_dataset, test_dataset

    def _get_num_classes(self) -> int:
        """Get the number of classes in the dataset."""
        if self.dataset_name in ["mnist", "fashionmnist", "cifar10"]:
            return 10
        return 10

    def distribute_data(self) -> List[Subset]:
        """
        Distribute training data among clients using IID (independent and identically distributed) partitioning.

        Returns:
            List of Subset objects, one for each client
        """
        num_samples = len(self.train_dataset)
        indices = np.arange(num_samples)
        self.rng.shuffle(indices)

        # Split into num_clients parts
        client_data_indices = np.array_split(indices, self.num_clients)

        # Create Subset objects for each client
        client_datasets = [
            Subset(self.train_dataset, client_indices.tolist())
            for client_indices in client_data_indices
        ]

        return client_datasets

    def get_test_loader(self, batch_size: int) -> DataLoader:
        """Get a DataLoader for the test dataset."""
        return DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )

    def get_client_loader(
        self, client_dataset: Subset, batch_size: int, shuffle: bool = True
    ) -> DataLoader:
        """Get a DataLoader for a specific client's dataset."""
        return DataLoader(
            client_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0
        )

    def get_dataset_info(self) -> dict:
        """Get information about the dataset."""
        return {
            "name": self.dataset_name,
            "num_classes": self.num_classes,
            "img_shape": self.img_shape,
            "num_train_samples": len(self.train_dataset),
            "num_test_samples": len(self.test_dataset),
            "num_clients": self.num_clients,
            "samples_per_client": len(self.train_dataset) // self.num_clients,
        }
