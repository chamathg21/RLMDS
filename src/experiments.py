"""
Experiment runner for federated learning with malicious client detection.
"""

import os
import json
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
import logging

from .datasets import DataDistributor
from .models import create_model
from .attacks import AttackManager
from .client import ClientManager, Client
from .server import Server
from .metrics import MetricsCalculator
from .plotting import PlotGenerator


class ExperimentRunner:
    """
    Orchestrates federated learning experiments with malicious client detection.
    """

    def __init__(
        self,
        config: dict,
        dataset_name: str,
        attack_type: str,
        num_malicious: int,
        poisoning_rate: float,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize ExperimentRunner.

        Args:
            config: Configuration dictionary
            dataset_name: Name of dataset
            attack_type: Type of attack
            num_malicious: Number of malicious clients
            poisoning_rate: Poisoning rate
            logger: Logger instance
        """
        self.config = config
        self.dataset_name = dataset_name
        self.attack_type = attack_type
        self.num_malicious = num_malicious
        self.poisoning_rate = poisoning_rate
        self.logger = logger or self._get_default_logger()

        # Set random seeds
        self.seed = config.get("RANDOM_SEED", 42)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        # Determine device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")

        # Initialize data and models
        self.data_distributor = self._initialize_data()
        self.attack_manager = AttackManager(
            config["NUM_CLIENTS"], num_malicious, seed=self.seed
        )
        self.global_model = create_model(
            dataset_name, num_classes=self.data_distributor.num_classes
        )
        self.global_model = self.global_model.to(self.device)

        # Initialize server
        self.server = self._initialize_server()

        # Initialize clients
        self.client_manager = self._initialize_clients()

        # Metrics tracking
        self.metrics_per_round = []
        self.test_accuracy_per_round = []
        self.detection_labels_per_round = []

    def _get_default_logger(self) -> logging.Logger:
        """Create a default logger."""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def _initialize_data(self) -> DataDistributor:
        """Initialize data distribution."""
        self.logger.info(f"Initializing data for {self.dataset_name}")
        data_distributor = DataDistributor(
            dataset_name=self.dataset_name,
            data_dir=self.config["DATA_DIR"],
            num_clients=self.config["NUM_CLIENTS"],
            seed=self.seed,
        )
        dataset_info = data_distributor.get_dataset_info()
        self.logger.info(f"Dataset info: {dataset_info}")
        return data_distributor

    def _initialize_server(self) -> Server:
        """Initialize federation server."""
        test_loader = self.data_distributor.get_test_loader(
            batch_size=self.config["BATCH_SIZE"]
        )

        clustering_config = {
            "convergence_steps": self.config.get("CLUSTERING_CONVERGENCE_STEPS", 10),
            "seed": self.seed,
        }

        rl_config = {
            "num_clients": self.config["NUM_CLIENTS"],
            "num_states": self.config.get("NUM_RL_STATES", 2),
            "num_actions": self.config.get("NUM_RL_ACTIONS", 2),
            "learning_rate": self.config.get("RL_LEARNING_RATE", 0.1),
            "discount_factor": self.config.get("RL_DISCOUNT_FACTOR", 0.9),
            "exploration_rate": self.config.get("RL_EXPLORATION_RATE", 0.1),
            "reward_correct": self.config.get("RL_REWARD_CORRECT", 1.0),
            "reward_penalty": self.config.get("RL_REWARD_PENALTY", -0.5),
            "q_threshold": self.config.get("Q_VALUE_THRESHOLD", 0.5),
            "seed": self.seed,
        }

        server = Server(
            model=self.global_model,
            device=self.device,
            clustering_config=clustering_config,
            rl_config=rl_config,
            test_loader=test_loader,
        )

        self.logger.info("Server initialized")
        return server

    def _initialize_clients(self) -> ClientManager:
        """Initialize federated clients."""
        # Distribute data
        client_datasets = self.data_distributor.distribute_data()

        # Apply attacks if needed
        poisoned_datasets = []
        for client_id, dataset in enumerate(client_datasets):
            poisoned_dataset = self.attack_manager.apply_attack(
                client_dataset=dataset,
                client_id=client_id,
                attack_type=self.attack_type,
                poisoning_rate=self.poisoning_rate,
                target_class=self.config.get("BACKDOOR_TARGET_CLASS", 0),
                trigger_size=self.config.get("BACKDOOR_TRIGGER_SIZE", 4),
            )
            poisoned_datasets.append(poisoned_dataset)

        # Create dataloaders
        dataloaders = [
            self.data_distributor.get_client_loader(
                dataset, batch_size=self.config["BATCH_SIZE"], shuffle=True
            )
            for dataset in poisoned_datasets
        ]

        # Initialize client manager
        client_manager = ClientManager(device=self.device)
        client_manager.register_clients(
            num_clients=self.config["NUM_CLIENTS"],
            model=self.global_model,
            dataloaders=dataloaders,
            learning_rate=self.config["LEARNING_RATE"],
        )

        self.logger.info(
            f"Initialized {self.config['NUM_CLIENTS']} clients with "
            f"{self.num_malicious} malicious"
        )
        return client_manager

    def run_experiment(self) -> Dict:
        """
        Run the federated learning experiment.

        Returns:
            Dictionary with experiment results
        """
        num_rounds = self.config["NUM_GLOBAL_ROUNDS"]
        local_epochs = self.config["LOCAL_EPOCHS"]

        # Ground truth labels
        true_labels = np.zeros(self.config["NUM_CLIENTS"], dtype=int)
        malicious_clients = self.attack_manager.get_malicious_clients()
        for client_id in malicious_clients:
            true_labels[client_id] = 1

        self.logger.info(f"Starting {num_rounds} rounds of federated learning")
        self.logger.info(f"Malicious clients: {sorted(malicious_clients)}")
        self.logger.info(f"Attack type: {self.attack_type}, Poisoning rate: {self.poisoning_rate}")

        for round_num in range(num_rounds):
            self.logger.info(f"\n--- Round {round_num + 1}/{num_rounds} ---")

            # Client local training
            self.logger.info("Training on clients...")
            self.client_manager.set_all_weights(
                self.server.aggregate_weights({})
            )

            losses = self.client_manager.train_all_clients(num_epochs=local_epochs)
            avg_loss = np.mean(list(losses.values()))
            self.logger.info(f"Average client loss: {avg_loss:.4f}")

            # Server aggregation
            all_weights = {}
            for client_id in range(self.config["NUM_CLIENTS"]):
                all_weights[client_id] = self.client_manager.get_client_weights(client_id)

            participating_clients = [
                c for c in range(self.config["NUM_CLIENTS"])
                if c not in self.server.excluded_clients
            ]

            aggregated_weights = self.server.aggregate_weights(
                all_weights, participating_clients=participating_clients
            )
            self.server.set_global_weights(aggregated_weights)

            # Malicious client detection
            self.logger.info("Running malicious client detection...")
            gradient_vectors = self.client_manager.get_all_gradient_vectors()
            detection_labels, clustering_stats, rl_metrics = self.server.detect_malicious_clients(
                gradient_vectors, true_labels=true_labels
            )

            # Evaluation
            test_acc, test_loss = self.server.evaluate_on_test_set()
            self.test_accuracy_per_round.append(test_acc)
            self.logger.info(f"Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")

            # Compute detection metrics
            detection_metrics = MetricsCalculator.compute_detection_metrics(
                detection_labels, true_labels
            )
            detection_metrics["test_accuracy"] = test_acc
            detection_metrics["test_loss"] = test_loss
            detection_metrics.update(rl_metrics)

            self.metrics_per_round.append(detection_metrics)
            self.detection_labels_per_round.append(detection_labels.copy())

            if (round_num + 1) % self.config["LOG_INTERVAL"] == 0:
                MetricsCalculator.print_metrics_summary(
                    detection_metrics, prefix=f"Round {round_num + 1}: "
                )

        self.logger.info("\nExperiment completed!")
        return self._compile_results()

    def _compile_results(self) -> Dict:
        """Compile experiment results into a structured dictionary."""
        # Extract metrics over rounds
        metric_names = [
            "malicious_detection_acc",
            "benign_detection_acc",
            "overall_detection_acc",
            "precision",
            "recall",
            "f1",
            "false_positive_rate",
            "test_accuracy",
        ]

        results = {
            "config": {
                "dataset": self.dataset_name,
                "attack_type": self.attack_type,
                "num_malicious": self.num_malicious,
                "poisoning_rate": self.poisoning_rate,
                "num_clients": self.config["NUM_CLIENTS"],
                "num_rounds": self.config["NUM_GLOBAL_ROUNDS"],
            },
            "metrics_per_round": {},
        }

        for metric_name in metric_names:
            values = [m.get(metric_name, 0) for m in self.metrics_per_round]
            results["metrics_per_round"][metric_name] = values

        # Final metrics
        final_metrics = self.metrics_per_round[-1] if self.metrics_per_round else {}
        results["final_metrics"] = final_metrics

        return results

    def save_results(self, results_dir: str) -> str:
        """
        Save experiment results to file.

        Args:
            results_dir: Directory for results

        Returns:
            Path to saved file
        """
        os.makedirs(results_dir, exist_ok=True)

        filename = (
            f"{results_dir}/{self.dataset_name.lower()}_"
            f"{self.attack_type.lower()}_{self.num_malicious}mal_"
            f"{int(self.poisoning_rate*100)}poison.json"
        )

        results = self._compile_results()

        # Convert numpy arrays to lists for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj

        serializable_results = convert_to_serializable(results)

        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        self.logger.info(f"Results saved to {filename}")
        return filename

    def generate_plots(self, figures_dir: str) -> str:
        """
        Generate plots for this experiment.

        Args:
            figures_dir: Directory for figures

        Returns:
            Path to saved figure
        """
        plot_gen = PlotGenerator(figures_dir=figures_dir)

        malicious_acc = np.array([
            m["malicious_detection_acc"] for m in self.metrics_per_round
        ])
        benign_acc = np.array([
            m["benign_detection_acc"] for m in self.metrics_per_round
        ])

        if self.attack_type.lower() == "labelflip":
            filename = plot_gen.plot_detection_accuracy_over_rounds(
                malicious_acc,
                benign_acc,
                dataset=self.dataset_name,
                attack_type=self.attack_type,
                num_malicious=self.num_malicious,
                poisoning_rate=self.poisoning_rate,
            )
        elif self.attack_type.lower() == "backdoor":
            model_acc = np.array(self.test_accuracy_per_round)
            filename = plot_gen.plot_model_accuracy_and_detection_single_axis(
                model_acc,
                malicious_acc,
                dataset=self.dataset_name,
                num_malicious=self.num_malicious,
                poisoning_rate=self.poisoning_rate,
            )
        else:
            filename = ""

        self.logger.info(f"Plot saved to {filename}")
        return filename
