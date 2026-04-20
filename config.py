"""
Configuration file for the Federative Learning with RL-based Malicious Client Detection system.
Modify these parameters to run different experiments.
"""

# ============================================================================
# FEDERATED LEARNING HYPERPARAMETERS
# ============================================================================
NUM_CLIENTS = 50              # Total number of clients in the federation
NUM_GLOBAL_ROUNDS = 50        # Number of global aggregation rounds
LOCAL_EPOCHS = 1              # Number of local training epochs per client per round
BATCH_SIZE = 32               # Batch size for local training
LEARNING_RATE = 0.01          # Learning rate for local training

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================
DATASET = "mnist"             # Options: "mnist", "fashionmnist", "cifar10"
NUM_CLASSES = 10              # Number of output classes (10 for MNIST/FashionMNIST)
IMG_HEIGHT = 28               # Image height (28 for MNIST/FashionMNIST, 32 for CIFAR10)
IMG_WIDTH = 28                # Image width (28 for MNIST/FashionMNIST, 32 for CIFAR10)
IMG_CHANNELS = 1              # Number of channels (1 for MNIST/FashionMNIST, 3 for CIFAR10)
DATA_DIR = "data"             # Directory to store/load datasets

# ============================================================================
# ATTACK CONFIGURATION
# ============================================================================
ATTACK_TYPE = "labelflip"     # Options: "labelflip", "backdoor", "none"
NUM_MALICIOUS_CLIENTS = 10    # Number of malicious clients
POISONING_RATE = 0.10         # Poisoning rate (fraction of local data to poison)
                               # For label flip: 0.10 = 10%, 0.15 = 15%
                               # For backdoor: 0.30 = 30%

# Backdoor attack specific parameters
BACKDOOR_TARGET_CLASS = 0     # Target class for backdoor attack
BACKDOOR_TRIGGER_SIZE = 4     # Size of trigger patch (4x4 pixels)

# ============================================================================
# DETECTION CONFIGURATION
# ============================================================================
DETECTION_METHOD = "rl"       # Options: "rl", "similarity_only"

# Clustering parameters for two-means
CLUSTERING_CONVERGENCE_STEPS = 10  # Max iterations for clustering

# RL-based detection parameters
NUM_RL_STATES = 2             # Number of RL states (suspicious, benign)
NUM_RL_ACTIONS = 2            # Number of RL actions (stay, transition)
RL_LEARNING_RATE = 0.1        # Q-learning rate (alpha)
RL_DISCOUNT_FACTOR = 0.9      # Discount factor (gamma)
RL_EXPLORATION_RATE = 0.1     # Epsilon-greedy exploration rate
Q_VALUE_THRESHOLD = 0.5       # Threshold for classifying as malicious based on Q-value

# Rewards for RL
RL_REWARD_CORRECT = 1.0       # Reward for correct detection
RL_REWARD_PENALTY = -0.5      # Penalty for false detection

# ============================================================================
# REPRODUCIBILITY
# ============================================================================
RANDOM_SEED = 42              # Fixed seed for reproducibility

# ============================================================================
# OUTPUT AND LOGGING
# ============================================================================
RESULTS_DIR = "results"       # Directory to save experiment results
FIGURES_DIR = "figures"       # Directory to save plots
LOG_INTERVAL = 5              # Print logs every N rounds
SAVE_INTERVAL = 1             # Save results every N rounds

# ============================================================================
# EXPERIMENT PRESETS
# ============================================================================
# These presets define common experimental configurations

LABEL_FLIP_PRESETS = [
    {"num_malicious": 10, "poisoning_rate": 0.10},
    {"num_malicious": 10, "poisoning_rate": 0.15},
    {"num_malicious": 20, "poisoning_rate": 0.10},
    {"num_malicious": 20, "poisoning_rate": 0.15},
    {"num_malicious": 30, "poisoning_rate": 0.10},
    {"num_malicious": 30, "poisoning_rate": 0.15},
]

BACKDOOR_PRESETS = [
    {"num_malicious": 10, "poisoning_rate": 0.30},  # 20% of 50 clients
    {"num_malicious": 20, "poisoning_rate": 0.30},  # 40% of 50 clients
    {"num_malicious": 30, "poisoning_rate": 0.30},  # 60% of 50 clients
]
