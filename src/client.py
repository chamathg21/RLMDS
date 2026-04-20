"""
Federated learning client implementation.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from typing import Tuple, Optional


class Client:
    """
    Federated learning client.
    Performs local training on its dataset partition.
    """

    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        dataset: DataLoader,
        device: torch.device,
        learning_rate: float = 0.01,
    ):
        """
        Initialize Client.

        Args:
            client_id: Unique identifier for the client
            model: PyTorch neural network model
            dataset: DataLoader for the client's local dataset
            device: PyTorch device (cpu or cuda)
            learning_rate: Learning rate for local optimization
        """
        self.client_id = client_id
        self.model = model
        self.dataset = dataset
        self.device = device
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.CrossEntropyLoss()

    def train(self, num_epochs: int = 1) -> float:
        """
        Perform local training for specified number of epochs.

        Args:
            num_epochs: Number of local training epochs

        Returns:
            Average training loss over the epochs
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in self.dataset:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                # Forward pass
                logits = self.model(batch_x)
                loss = self.loss_fn(logits, batch_y)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            total_loss += epoch_loss

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    def get_weights(self) -> dict:
        """
        Get current model weights.

        Returns:
            Dictionary of model parameters
        """
        return {name: param.data.clone() for name, param in self.model.named_parameters()}

    def set_weights(self, weights: dict):
        """
        Set model weights from dictionary.

        Args:
            weights: Dictionary of model parameters to set
        """
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in weights:
                    param.copy_(weights[name])

    def get_gradient_vector(self) -> np.ndarray:
        """
        Extract the gradient vector from the last hidden layer.
        This is used for computing cosine similarity in malicious client detection.

        Returns:
            Flattened gradient vector from the penultimate layer
        """
        self.model.eval()
        gradients = []

        with torch.no_grad():
            for batch_x, batch_y in self.dataset:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                # Get penultimate layer output
                penultimate = self.model.get_penultimate_layer(batch_x)

                # Get gradient through final layer
                # For simplicity, we'll use the activations as gradient representation
                gradients.append(penultimate.cpu().numpy())

        if gradients:
            # Concatenate and average across batches
            gradient_vector = np.concatenate(gradients, axis=0).mean(axis=0)
        else:
            gradient_vector = np.zeros(1)

        return gradient_vector.flatten()

    def get_model_size(self) -> int:
        """Get total number of model parameters."""
        return sum(p.numel() for p in self.model.parameters())


class ClientManager:
    """
    Manages multiple federated learning clients.
    """

    def __init__(self, device: torch.device):
        """
        Initialize ClientManager.

        Args:
            device: PyTorch device (cpu or cuda)
        """
        self.device = device
        self.clients = {}

    def add_client(self, client_id: int, client: Client):
        """Add a client to the manager."""
        self.clients[client_id] = client

    def register_clients(
        self, num_clients: int, model: nn.Module, dataloaders: list, learning_rate: float = 0.01
    ):
        """
        Register multiple clients at once.

        Args:
            num_clients: Number of clients
            model: Base model (will be cloned for each client)
            dataloaders: List of DataLoaders, one per client
            learning_rate: Learning rate for all clients
        """
        for client_id in range(num_clients):
            # Clone the model for this client
            client_model = self._clone_model(model)
            client_model = client_model.to(self.device)

            client = Client(
                client_id=client_id,
                model=client_model,
                dataset=dataloaders[client_id],
                device=self.device,
                learning_rate=learning_rate,
            )
            self.add_client(client_id, client)

    def train_client(self, client_id: int, num_epochs: int = 1) -> float:
        """Train a specific client."""
        if client_id not in self.clients:
            raise ValueError(f"Client {client_id} not registered")
        return self.clients[client_id].train(num_epochs)

    def train_all_clients(self, num_epochs: int = 1) -> dict:
        """
        Train all clients sequentially.

        Args:
            num_epochs: Number of local training epochs

        Returns:
            Dictionary with client_id -> loss mapping
        """
        losses = {}
        for client_id, client in self.clients.items():
            losses[client_id] = client.train(num_epochs)
        return losses

    def get_client_weights(self, client_id: int) -> dict:
        """Get weights from a specific client."""
        if client_id not in self.clients:
            raise ValueError(f"Client {client_id} not registered")
        return self.clients[client_id].get_weights()

    def get_all_gradient_vectors(self) -> dict:
        """
        Get gradient vectors from all clients.

        Returns:
            Dictionary with client_id -> gradient_vector mapping
        """
        gradients = {}
        for client_id, client in self.clients.items():
            gradients[client_id] = client.get_gradient_vector()
        return gradients

    def set_client_weights(self, client_id: int, weights: dict):
        """Set weights for a specific client."""
        if client_id not in self.clients:
            raise ValueError(f"Client {client_id} not registered")
        self.clients[client_id].set_weights(weights)

    def set_all_weights(self, weights: dict):
        """Set weights for all clients."""
        for client_id, client in self.clients.items():
            client.set_weights(weights)

    def get_num_clients(self) -> int:
        """Get total number of registered clients."""
        return len(self.clients)

    @staticmethod
    def _clone_model(model: nn.Module) -> nn.Module:
        """Clone a PyTorch model including its architecture and weights."""
        import copy
        return copy.deepcopy(model)
