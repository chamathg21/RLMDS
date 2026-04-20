"""
Metrics computation for malicious client detection evaluation.
"""

import numpy as np
from typing import Dict, Tuple


class MetricsCalculator:
    """
    Computes evaluation metrics for malicious client detection.
    """

    @staticmethod
    def compute_detection_metrics(
        predicted_labels: np.ndarray, true_labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute detection metrics.

        Args:
            predicted_labels: Predicted labels (1=malicious, 0=benign)
            true_labels: Ground truth labels (1=malicious, 0=benign)

        Returns:
            Dictionary with metrics
        """
        # True positives, false positives, false negatives, true negatives
        tp = np.sum((predicted_labels == 1) & (true_labels == 1))
        fp = np.sum((predicted_labels == 1) & (true_labels == 0))
        fn = np.sum((predicted_labels == 0) & (true_labels == 1))
        tn = np.sum((predicted_labels == 0) & (true_labels == 0))

        # Detection accuracies
        malicious_detection_acc = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        benign_detection_acc = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # Overall detection accuracy
        overall_acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

        # Precision and recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # F1-score
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        # False positive rate and false negative rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        return {
            "malicious_detection_acc": malicious_detection_acc,
            "benign_detection_acc": benign_detection_acc,
            "overall_detection_acc": overall_acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_positive_rate": fpr,
            "false_negative_rate": fnr,
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
        }

    @staticmethod
    def compute_backdoor_success_rate(
        predictions: np.ndarray,
        backdoor_indices: np.ndarray,
        target_class: int,
    ) -> float:
        """
        Compute backdoor attack success rate.

        Args:
            predictions: Model predictions on test set
            backdoor_indices: Indices of backdoored samples in test set
            target_class: Target class for backdoor attack

        Returns:
            Success rate (fraction of backdoored samples correctly predicted as target)
        """
        if len(backdoor_indices) == 0:
            return 0.0

        backdoor_preds = predictions[backdoor_indices]
        success_count = np.sum(backdoor_preds == target_class)

        return success_count / len(backdoor_indices)

    @staticmethod
    def aggregate_metrics_over_rounds(
        metrics_per_round: list,
    ) -> Dict[str, Tuple[np.ndarray, float, float]]:
        """
        Aggregate metrics across rounds.

        Args:
            metrics_per_round: List of metric dictionaries, one per round

        Returns:
            Dictionary mapping metric name to (array_over_rounds, mean, std)
        """
        aggregated = {}

        if not metrics_per_round:
            return aggregated

        # Get all metric keys from first round
        metric_keys = metrics_per_round[0].keys()

        for key in metric_keys:
            # Skip if value is not numeric
            values = []
            for round_metrics in metrics_per_round:
                val = round_metrics[key]
                if isinstance(val, (int, float, np.number)):
                    values.append(float(val))

            if values:
                values_array = np.array(values)
                aggregated[key] = (values_array, np.mean(values_array), np.std(values_array))

        return aggregated

    @staticmethod
    def print_metrics_summary(metrics: Dict[str, float], prefix: str = ""):
        """
        Print a summary of metrics.

        Args:
            metrics: Dictionary of metrics
            prefix: Prefix for print statements
        """
        print(f"\n{prefix}Metrics Summary:")
        print(f"  Malicious Detection Accuracy: {metrics.get('malicious_detection_acc', 0):.4f}")
        print(f"  Benign Detection Accuracy: {metrics.get('benign_detection_acc', 0):.4f}")
        print(f"  Overall Detection Accuracy: {metrics.get('overall_detection_acc', 0):.4f}")
        print(f"  Precision: {metrics.get('precision', 0):.4f}")
        print(f"  Recall: {metrics.get('recall', 0):.4f}")
        print(f"  F1-score: {metrics.get('f1', 0):.4f}")
        print(f"  False Positive Rate: {metrics.get('false_positive_rate', 0):.4f}")
        print(f"  False Negative Rate: {metrics.get('false_negative_rate', 0):.4f}")
