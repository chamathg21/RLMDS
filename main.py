"""
Federated Learning with RL-based Malicious Client Detection
Main entry point for running experiments.
"""

import sys
import os
import logging
import argparse
import json
import config
from src.experiments import ExperimentRunner
from src.plotting import PlotGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_single_experiment(
    dataset: str,
    attack_type: str,
    num_malicious: int,
    poisoning_rate: float,
):
    """
    Run a single experiment configuration.

    Args:
        dataset: Dataset name
        attack_type: Type of attack
        num_malicious: Number of malicious clients
        poisoning_rate: Poisoning rate
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Experiment: {dataset} - {attack_type}")
    logger.info(f"Malicious clients: {num_malicious}, Poisoning: {poisoning_rate*100:.1f}%")
    logger.info(f"{'='*60}\n")

    # Create experiment runner
    runner = ExperimentRunner(
        config=vars(config),
        dataset_name=dataset,
        attack_type=attack_type,
        num_malicious=num_malicious,
        poisoning_rate=poisoning_rate,
        logger=logger,
    )

    # Run experiment
    results = runner.run_experiment()

    # Save results
    runner.save_results(config.RESULTS_DIR)

    # Generate plots
    runner.generate_plots(config.FIGURES_DIR)

    return results


def run_label_flip_experiments():
    """Run all label flipping experiments."""
    logger.info("\n" + "="*60)
    logger.info("Running Label Flipping Experiments")
    logger.info("="*60)

    for preset in config.LABEL_FLIP_PRESETS:
        for dataset in ["mnist", "fashionmnist"]:
            run_single_experiment(
                dataset=dataset,
                attack_type="labelflip",
                num_malicious=preset["num_malicious"],
                poisoning_rate=preset["poisoning_rate"],
            )


def run_backdoor_experiments():
    """Run all backdoor experiments."""
    logger.info("\n" + "="*60)
    logger.info("Running Backdoor Experiments")
    logger.info("="*60)

    for preset in config.BACKDOOR_PRESETS:
        for dataset in ["mnist", "fashionmnist"]:
            run_single_experiment(
                dataset=dataset,
                attack_type="backdoor",
                num_malicious=preset["num_malicious"],
                poisoning_rate=preset["poisoning_rate"],
            )


def run_all_experiments():
    """Run all experiments (label flip and backdoor)."""
    run_label_flip_experiments()
    run_backdoor_experiments()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Federated Learning with RL-based Malicious Client Detection"
    )
    parser.add_argument(
        "mode",
        nargs='?',
        default="all",
        choices=["all", "labelflip", "backdoor", "single"],
        help="Experiment mode to run"
    )
    parser.add_argument(
        "--dataset",
        default="mnist",
        choices=["mnist", "fashionmnist", "cifar10"],
        help="Dataset to use (for single mode)"
    )
    parser.add_argument(
        "--attack",
        default="labelflip",
        choices=["labelflip", "backdoor"],
        help="Attack type (for single mode)"
    )
    parser.add_argument(
        "--malicious",
        type=int,
        default=10,
        help="Number of malicious clients (for single mode)"
    )
    parser.add_argument(
        "--poison-rate",
        type=float,
        default=0.1,
        help="Poisoning rate (for single mode)"
    )

    args = parser.parse_args()

    # Create output directories
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.FIGURES_DIR, exist_ok=True)

    logger.info(f"Configuration: {vars(config)}")

    if args.mode == "all":
        run_all_experiments()
    elif args.mode == "labelflip":
        run_label_flip_experiments()
    elif args.mode == "backdoor":
        run_backdoor_experiments()
    elif args.mode == "single":
        run_single_experiment(
            dataset=args.dataset,
            attack_type=args.attack,
            num_malicious=args.malicious,
            poisoning_rate=args.poison_rate,
        )

    logger.info("\n" + "="*60)
    logger.info("All experiments completed!")
    logger.info(f"Results saved to: {config.RESULTS_DIR}")
    logger.info(f"Figures saved to: {config.FIGURES_DIR}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
