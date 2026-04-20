"""
Plotting utilities for visualizing experiment results.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from typing import Dict, List, Optional


class PlotGenerator:
    """
    Generates publication-ready plots for federated learning experiments.
    """

    def __init__(self, figures_dir: str = "figures", dpi: int = 300):
        """
        Initialize PlotGenerator.

        Args:
            figures_dir: Directory to save figures
            dpi: Resolution of saved figures
        """
        self.figures_dir = figures_dir
        self.dpi = dpi

        # Create figures directory if it doesn't exist
        os.makedirs(figures_dir, exist_ok=True)

        # Set matplotlib style
        plt.style.use('seaborn-v0_8-darkgrid')

    def plot_detection_accuracy_over_rounds(
        self,
        malicious_acc: np.ndarray,
        benign_acc: np.ndarray,
        dataset: str,
        attack_type: str,
        num_malicious: int,
        poisoning_rate: float,
        filename: Optional[str] = None,
    ) -> str:
        """
        Plot malicious and benign detection accuracy over rounds.

        Args:
            malicious_acc: Array of malicious detection accuracy per round
            benign_acc: Array of benign detection accuracy per round
            dataset: Dataset name
            attack_type: Type of attack
            num_malicious: Number of malicious clients
            poisoning_rate: Poisoning rate
            filename: Custom filename (if None, auto-generated)

        Returns:
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        rounds = np.arange(len(malicious_acc))

        ax.plot(rounds, malicious_acc, marker='o', linewidth=2, label='Malicious Detection', markersize=4)
        ax.plot(rounds, benign_acc, marker='s', linewidth=2, label='Benign Detection', markersize=4)

        ax.set_xlabel('Global Round', fontsize=12, fontweight='bold')
        ax.set_ylabel('Detection Accuracy', fontsize=12, fontweight='bold')
        ax.set_ylim([0, 1.05])
        ax.set_title(
            f'{dataset.upper()} {attack_type.capitalize()} Attack\n'
            f'{num_malicious} Malicious, {poisoning_rate*100:.0f}% Poisoning',
            fontsize=13, fontweight='bold'
        )
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)

        # Generate filename if not provided
        if filename is None:
            filename = (
                f"{self.figures_dir}/{dataset.lower()}_"
                f"{attack_type.lower()}_{num_malicious}mal_{int(poisoning_rate*100)}poison.png"
            )

        plt.tight_layout()
        plt.savefig(filename, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        return filename

    def plot_model_accuracy_and_detection(
        self,
        model_acc: np.ndarray,
        detection_acc: np.ndarray,
        dataset: str,
        num_malicious: int,
        poisoning_rate: float,
        filename: Optional[str] = None,
    ) -> str:
        """
        Plot both model accuracy and detection accuracy on same figure with dual y-axes.

        Args:
            model_acc: Array of model accuracy per round
            detection_acc: Array of malicious detection accuracy per round
            dataset: Dataset name
            num_malicious: Number of malicious clients
            poisoning_rate: Poisoning rate
            filename: Custom filename

        Returns:
            Path to saved figure
        """
        fig, ax1 = plt.subplots(figsize=(10, 6))

        rounds = np.arange(len(model_acc))
        color = 'tab:blue'
        ax1.set_xlabel('Global Round', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Model Accuracy', color=color, fontsize=12, fontweight='bold')
        line1 = ax1.plot(rounds, model_acc, color=color, marker='o', linewidth=2, label='Model Accuracy', markersize=4)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim([0, 1.05])

        ax2 = ax1.twinx()
        color = 'tab:orange'
        ax2.set_ylabel('Malicious Detection Accuracy', color=color, fontsize=12, fontweight='bold')
        line2 = ax2.plot(rounds, detection_acc, color=color, marker='s', linewidth=2, label='Detection Accuracy', markersize=4)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim([0, 1.05])

        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, fontsize=11, loc='best')

        percent_mal = int(num_malicious / 50 * 100)  # Assuming 50 total clients
        ax1.set_title(
            f'{dataset.upper()} Backdoor Attack\n'
            f'{percent_mal}% Malicious Clients, {poisoning_rate*100:.0f}% Poisoning',
            fontsize=13, fontweight='bold'
        )

        ax1.grid(True, alpha=0.3)

        if filename is None:
            filename = (
                f"{self.figures_dir}/{dataset.lower()}_"
                f"backdoor_{percent_mal}mal_30poison.png"
            )

        plt.tight_layout()
        plt.savefig(filename, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        return filename

    def plot_model_accuracy_and_detection_single_axis(
        self,
        model_acc: np.ndarray,
        detection_acc: np.ndarray,
        dataset: str,
        num_malicious: int,
        poisoning_rate: float,
        filename: Optional[str] = None,
    ) -> str:
        """
        Plot model accuracy and detection accuracy on same axis (if scales are compatible).

        Args:
            model_acc: Array of model accuracy per round
            detection_acc: Array of malicious detection accuracy per round
            dataset: Dataset name
            num_malicious: Number of malicious clients
            poisoning_rate: Poisoning rate
            filename: Custom filename

        Returns:
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        rounds = np.arange(len(model_acc))

        ax.plot(rounds, model_acc, marker='o', linewidth=2, label='Model Accuracy', markersize=4)
        ax.plot(rounds, detection_acc, marker='s', linewidth=2, label='Malicious Detection Accuracy', markersize=4)

        ax.set_xlabel('Global Round', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_ylim([0, 1.05])

        percent_mal = int(num_malicious / 50 * 100)
        ax.set_title(
            f'{dataset.upper()} Backdoor Attack\n'
            f'{percent_mal}% Malicious Clients',
            fontsize=13, fontweight='bold'
        )

        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)

        if filename is None:
            filename = (
                f"{self.figures_dir}/{dataset.lower()}_"
                f"backdoor_{percent_mal}mal_30poison.png"
            )

        plt.tight_layout()
        plt.savefig(filename, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        return filename

    def plot_final_metrics_table(
        self,
        scenarios: List[Dict],
        dataset: str,
        results_file: Optional[str] = None,
    ):
        """
        Create and save a summary table of final metrics across scenarios.

        Args:
            scenarios: List of scenario dictionaries with metrics
            dataset: Dataset name
            results_file: File to save results
        """
        import pandas as pd

        # Create DataFrame
        df_data = []
        for scenario in scenarios:
            row = {
                'Dataset': dataset,
                'Scenario': scenario.get('scenario_name', ''),
                'DACC': scenario.get('final_detection_acc', 0),
                'FPR': scenario.get('final_fpr', 0),
                'Precision': scenario.get('final_precision', 0),
                'Recall': scenario.get('final_recall', 0),
                'F1-score': scenario.get('final_f1', 0),
            }
            df_data.append(row)

        df = pd.DataFrame(df_data)

        if results_file is None:
            results_file = f"{self.figures_dir}/{dataset.lower()}_benchmark_table.csv"

        df.to_csv(results_file, index=False)
        print(f"\nResults table saved to {results_file}")
        print(df.to_string(index=False))

        return results_file

    def plot_comparison_across_settings(
        self,
        all_results: Dict[str, Dict],
        metric_name: str = "malicious_detection_acc",
        filename: Optional[str] = None,
    ) -> str:
        """
        Plot comparison of a metric across different experimental settings.

        Args:
            all_results: Dictionary mapping scenario names to results
            metric_name: Name of metric to plot
            filename: Custom filename

        Returns:
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        for scenario_name, results in all_results.items():
            if isinstance(results.get(metric_name), (list, np.ndarray)):
                ax.plot(results[metric_name], marker='o', label=scenario_name, linewidth=2)

        ax.set_xlabel('Global Round', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.set_title(f'Comparison: {metric_name.replace("_", " ").title()}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)

        if filename is None:
            filename = f"{self.figures_dir}/comparison_{metric_name}.png"

        plt.tight_layout()
        plt.savefig(filename, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        return filename
