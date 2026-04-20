"""
Neural network models for federated learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


class SimpleCNN(nn.Module):
    """
    Simple CNN model for MNIST and FashionMNIST.
    Designed to extract last-layer gradients for malicious client detection.
    """

    def __init__(self, num_classes: int = 10, input_channels: int = 1):
        """
        Initialize SimpleCNN.

        Args:
            num_classes: Number of output classes
            input_channels: Number of input channels (1 for grayscale, 3 for RGB)
        """
        super(SimpleCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)

        # Pooling
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Conv layer 1 + activation + pooling
        x = self.pool(F.relu(self.conv1(x)))

        # Conv layer 2 + activation + pooling
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layer 1 + activation + dropout
        x = self.dropout(F.relu(self.fc1(x)))

        # Fully connected layer 2 (output)
        x = self.fc2(x)

        return x

    def get_penultimate_layer(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get the output of the penultimate (last hidden) layer.
        Used for extracting last-layer gradients.

        Args:
            x: Input tensor

        Returns:
            Output of fc1 layer (before final classification layer)
        """
        # Conv layer 1 + activation + pooling
        x = self.pool(F.relu(self.conv1(x)))

        # Conv layer 2 + activation + pooling
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layer 1 + activation + dropout
        x = self.dropout(F.relu(self.fc1(x)))

        return x


class CIFAR10CNN(nn.Module):
    """
    CNN model optimized for CIFAR-10.
    """

    def __init__(self, num_classes: int = 10):
        """Initialize CIFAR10CNN."""
        super(CIFAR10CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)

        self.dropout = nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)

        return x

    def get_penultimate_layer(self, x: torch.Tensor) -> torch.Tensor:
        """Get the output of the penultimate layer."""
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))

        return x


def create_model(dataset_name: str, num_classes: int = 10) -> nn.Module:
    """
    Factory function to create appropriate model based on dataset.

    Args:
        dataset_name: Name of the dataset
        num_classes: Number of output classes

    Returns:
        Neural network model
    """
    dataset_name = dataset_name.lower()

    if dataset_name in ["mnist", "fashionmnist"]:
        return SimpleCNN(num_classes=num_classes, input_channels=1)
    elif dataset_name == "cifar10":
        return CIFAR10CNN(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
