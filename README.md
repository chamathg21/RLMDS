# Federated Learning with RL-based Malicious Client Detection

A comprehensive Python framework for simulating federated learning environments with malicious client detection using reinforcement learning and clustering-based similarity analysis. This project reproduces experimental results for both label flipping and backdoor attacks on federated learning systems.

## Overview

This framework implements a federated learning system with an intelligent detection mechanism for identifying and excluding malicious clients. The detection method combines:

1. **Cosine Similarity Analysis**: Computes similarity between client gradients
2. **Two-Cluster Grouping**: Partitions clients into benign and suspicious clusters
3. **Q-Learning Based Detection**: Uses reinforcement learning to track client suspicion over time

Supported attack types:
- **Label Flipping Attacks**: Randomly flip labels of training samples
- **Backdoor Attacks**: Add trigger patterns and alter labels for backdoored samples

## Features

- ✅ Federated learning with FedAvg aggregation
- ✅ Multiple datasets: MNIST, FashionMNIST (CIFAR-10 support for future extension)
- ✅ Two attack modes: label flipping and backdoor
- ✅ RL-based malicious client detection with Q-learning
- ✅ Comprehensive metrics: detection accuracy, precision, recall, F1-score, FPR
- ✅ Publication-ready plots with matplotlib
- ✅ Reproducible with fixed random seeds
- ✅ Clean modular architecture
- ✅ JSON and CSV output formats

## Project Structure

```
project_root/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── config.py                # Configuration parameters
├── main.py                  # Main entry point
├── data/                    # Dataset storage (auto-created)
├── results/                 # Experiment results in JSON (auto-created)
├── figures/                 # Generated plots (auto-created)
└── src/
    ├── __init__.py
    ├── datasets.py          # Data distribution and loading
    ├── models.py            # Neural network models (SimpleCNN, CIFAR10CNN)
    ├── client.py            # Federated client implementation
    ├── server.py            # Aggregation and detection server
    ├── attacks.py           # Attack implementations
    ├── clustering.py        # Two-means clustering for grouping
    ├── rl_detector.py       # Q-learning based malicious detector
    ├── metrics.py           # Evaluation metrics computation
    ├── experiments.py       # Experiment orchestration
    └── plotting.py          # Visualization and plot generation
```

## Installation & Setup

### Prerequisites
- Python 3.8+
- pip or conda

### Step 1: Clone/Navigate to Project

```bash
cd RLMDS
```

### Step 2: Create Virtual Environment (Optional but Recommended)

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# OR using conda
conda create -n rlmds python=3.9
conda activate rlmds
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- **torch** & **torchvision**: Deep learning framework
- **numpy**: Numerical computing
- **matplotlib**: Plotting and visualization
- **scikit-learn**: Clustering and metrics
- **pandas**: Data manipulation
- **tqdm**: Progress bars

## Usage

### Running Experiments

#### Option 1: Run All Experiments (Label Flip + Backdoor)

```bash
python main.py all
```

This runs 12 label flipping experiments (2 datasets × 6 configurations each) and 
6 backdoor experiments (2 datasets × 3 configurations each).

#### Option 2: Run Only Label Flipping Experiments

```bash
python main.py labelflip
```

Runs 12 experiments with label flipping attacks on MNIST and FashionMNIST.

#### Option 3: Run Only Backdoor Experiments

```bash
python main.py backdoor
```

Runs 6 experiments with backdoor attacks on MNIST and FashionMNIST.

#### Option 4: Run Single Experiment

```bash bash
python main.py single --dataset mnist --attack labelflip --malicious 10 --poison-rate 0.1
```

Parameters for single experiments:
- `--dataset`: "mnist", "fashionmnist", "cifar10"
- `--attack`: "labelflip", "backdoor"
- `--malicious`: Number of malicious clients (1-50)
- `--poison-rate`: Poisoning rate (0.0-1.0)

### Output Files

#### Results Directory (`results/`)
Contains JSON files with detailed metrics for each experiment:

```
mnist_labelflip_10mal_10poison.json
mnist_labelflip_10mal_15poison.json
...
```

Each JSON file contains:
```json
{
  "config": {
    "dataset": "mnist",
    "attack_type": "labelflip",
    "num_malicious": 10,
    "poisoning_rate": 0.1,
    ...
  },
  "metrics_per_round": {
    "malicious_detection_acc": [0.5, 0.7, 0.85, ...],
    "benign_detection_acc": [0.9, 0.95, 0.98, ...],
    "precision": [...],
    ...
  },
  "final_metrics": {...}
}
```

#### Figures Directory (`figures/`)
Contains PNG plots with publication-ready quality:

**Label Flipping:**
```
mnist_labelflip_10mal_10poison.png
mnist_labelflip_10mal_15poison.png
mnist_labelflip_20mal_10poison.png
...
fashionmnist_labelflip_10mal_10poison.png
...
```

**Backdoor:**
```
mnist_backdoor_20mal_30poison.png   # 20% malicious
mnist_backdoor_40mal_30poison.png   # 40% malicious
mnist_backdoor_60mal_30poison.png   # 60% malicious
...
```

## Configuration

Edit `config.py` to modify parameters:

```python
# Federated Learning
NUM_CLIENTS = 50              # Total clients
NUM_GLOBAL_ROUNDS = 50        # Training rounds
LOCAL_EPOCHS = 1              # Local training epochs per round
BATCH_SIZE = 32               # Training batch size
LEARNING_RATE = 0.01          # Optimizer learning rate

# Attack Settings
ATTACK_TYPE = "labelflip"     # Or "backdoor", "none"
NUM_MALICIOUS_CLIENTS = 10    # Malicious clients
POISONING_RATE = 0.10         # Poison 10% of local data

# Detection (RL)
RL_LEARNING_RATE = 0.1        # Q-learning alpha
RL_DISCOUNT_FACTOR = 0.9      # Q-learning gamma
Q_VALUE_THRESHOLD = 0.5       # Detection threshold

# Reproducibility
RANDOM_SEED = 42              # Fixed seed
```

## Experimental Configurations

### Label Flipping
6 configurations per dataset:
1. 10 malicious clients, 10% poisoning → 20% attack
2. 10 malicious clients, 15% poisoning → 20% attack
3. 20 malicious clients, 10% poisoning → 40% attack
4. 20 malicious clients, 15% poisoning → 40% attack
5. 30 malicious clients, 10% poisoning → 60% attack
6. 30 malicious clients, 15% poisoning → 60% attack

### Backdoor Attacks
3 configurations per dataset:
1. 10 malicious clients (20% of 50), 30% local data poisoning
2. 20 malicious clients (40% of 50), 30% local data poisoning
3. 30 malicious clients (60% of 50), 30% local data poisoning

**Note:** Backdoor trigger is a 4×4 white square in bottom-right corner.

## Metrics Explained

### Detection Metrics
- **Malicious Detection Accuracy**: Fraction of actual malicious clients correctly identified
- **Benign Detection Accuracy**: Fraction of actual benign clients correctly kept as benign
- **Overall Detection Accuracy**: (TP + TN) / Total
- **Precision**: TP / (TP + FP) — Of flagged clients, how many are truly malicious
- **Recall**: TP / (TP + FN) — Of all malicious clients, how many were detected
- **F1-Score**: Harmonic mean of precision and recall
- **False Positive Rate (FPR)**: FP / (FP + TN) — Fraction of benign clients incorrectly flagged
- **False Negative Rate**: FN / (FN + TP) — Fraction of malicious clients missed

### Model Metrics (Backdoor)
- **Model Accuracy**: Global model accuracy on clean test set
- **Backdoor Success Rate**: Fraction of backdoored samples classified as target class

## Reproducibility

All experiments use fixed random seeds for reproducibility:

```python
RANDOM_SEED = 42  # In config.py
```

Seeds are applied to:
- ✓ NumPy random state
- ✓ PyTorch random state
- ✓ PyTorch CUDA (if available)
- ✓ Data shuffling
- ✓ Client maliciousness assignment
- ✓ Poisoning sample selection
- ✓ RL initialization

**To reproduce results:**
1. Keep `RANDOM_SEED = 42` in `config.py`
2. Run experiments with same PyTorch/NumPy versions (see requirements.txt)
3. Results should be identical across runs

## Expected Output

### Console Output Example
```
==============================================================
Running Label Flipping Experiments
==============================================================

--- Round 1/50 ---
Training on clients...
Average client loss: 2.1234
Running malicious client detection...
Test accuracy: 0.8945, Test loss: 0.3412

Metrics Summary:
  Malicious Detection Accuracy: 0.5000
  Benign Detection Accuracy: 0.9500
  Overall Detection Accuracy: 0.7600
  Precision: 0.8333
  Recall: 0.5000
  F1-score: 0.6250
  False Positive Rate: 0.0526
```

### Generated Files
- ✓ 18 JSON result files (9 per attack type for 2 datasets)
- ✓ 18 PNG plot files (same per dataset)
- ✓ All saved with descriptive names indicating configuration

## Modifying Parameters

### To Change Number of Clients

```python
# config.py
NUM_CLIENTS = 100  # Instead of 50

# Note: Adjust this accordingly
RLMDS_PRESETS = [
    {"num_malicious": 20, "poisoning_rate": 0.10},  # Now 20% of 100
    ...
]
```

### To Change Poisoning Rates

```python
# config.py
LABEL_FLIP_PRESETS = [
    {"num_malicious": 10, "poisoning_rate": 0.05},   # 5% instead of 10%
    {"num_malicious": 10, "poisoning_rate": 0.20},   # 20% instead of 15%
    ...
]
```

### To Adjust RL Parameters

```python
# config.py
RL_LEARNING_RATE = 0.05       # Lower learning rate
RL_DISCOUNT_FACTOR = 0.95     # Higher future reward emphasis
Q_VALUE_THRESHOLD = 0.3       # Lower threshold = more aggressive detection
```

### To Add New Datasets

1. Update `config.py`:
```python
# Add to LABEL_FLIP_PRESETS or BACKDOOR_PRESETS
# Update IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS if needed
```

2. The framework automatically handles MNIST, FashionMNIST, and CIFAR-10.

## Troubleshooting

### Out of Memory
```python
# config.py
BATCH_SIZE = 16  # Reduce from 32
NUM_CLIENTS = 30  # Reduce from 50
```

### Slow Training
```python
# config.py
LOCAL_EPOCHS = 0  # Train fewer local epochs
NUM_GLOBAL_ROUNDS = 30  # Reduce global rounds
```

### CUDA Not Available
```python
# Automatic fallback, but can force CPU
# Modify top of experiments.py if needed
```

### Results Not Saved
- Check that `results/` and `figures/` directories are writable
- Directories are auto-created if missing

## Implementation Details

### Detection Algorithm

1. **Gradient Extraction**: Each client computes last-layer gradients
2. **Similarity Matrix**: Cosine similarity computed between all client gradients
3. **Two-Means Clustering**: Clients partitioned into benign/suspicious clusters
4. **Cluster Interpretation**: Higher similarity cluster = benign
5. **Q-Learning Update**: Each client maintains Q(state, action)
   - States: Suspicious (1), Benign (0)
   - Actions: Stay (0), Transition (1)
   - Rewards: Based on cluster assignment
6. **Detection**: Client flagged if Q(suspicious) > Q(benign) + threshold

### Attack Implementations

**Label Flipping:**
- Select random subset of client's training data
- Flip labels to random incorrect class
- Image content unchanged

**Backdoor:**
- Select random subset of client's training data
- Add 4×4 white trigger in bottom-right corner
- Change label to target class (default: 0)
- Allows measuring model's backdoor success rate

## Performance Expectations

On CPU (Intel i5, 8GB RAM):
- 1 experiment: ~5-10 minutes
- All experiments (18 total): ~2-3 hours

On GPU (NVIDIA RTX 2080):
- 1 experiment: ~1-2 minutes
- All experiments: ~20-30 minutes

## References & Citation

## Contributing

For bug reports, feature requests, or contributions, please submit an issue or pull request.

## FAQ

**Q: Can I run this with my own dataset?**
A: Yes! Modify `datasets.py` to add support for your dataset. Follow the MNIST implementation as a template.

**Q: How do I interpret the plots?**
A: The x-axis shows training rounds. For label flipping, higher curves = better detection. For backdoor, you want high model accuracy + high detection accuracy.

**Q: Why does my detection accuracy start low?**
A: RL-based detection warm up over time. The Q-values need several rounds to stabilize. This is expected behavior.

**Q: Can I modify the reward function?**
A: Yes! Edit `config.py` `RL_REWARD_CORRECT` and `RL_REWARD_PENALTY`, or modify the update logic in `rl_detector.py`.

**Q: How reproducible is this?**
A: Fully reproducible with same random seed and same PyTorch version. Results are deterministic.

---



