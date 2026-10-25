"""All experiment settings in one place."""
import os
import random
from pathlib import Path

import numpy as np
import torch

DATA_DIR = Path("data/UCSD_Anomaly_Dataset.v1p2")
SITES = {"site1": DATA_DIR / "UCSDped1", "site2": DATA_DIR / "UCSDped2"}
RESULTS_DIR = Path("results")

IMAGE_SIZE = 64
ROUNDS = 10
LOCAL_EPOCHS = 3
BATCH_SIZE = 64
LR = 1e-3
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_CLIPS = None  # None = every clip

# Multi-seed runs in run_experiments.py (all with the same 30-epoch budget).
SEEDS = range(42, 47)  # Table 2
EXTRA_SEEDS = range(42, 43)  # Table 3, one run (FedProx, client count, same scene, frame stacks)
PROX_MUS = (0.001, 0.01)  # FedProx proximal weights; mu = 0.1 diverged on seed 42 (mean AUC 0.31)
CLIENTS_PER_SITE = (2, 4)  # each site's training clips split into this many clients
STACK = 8  # consecutive frames per input in the frame-stack study

# QUICK=1 runs a tiny version (4 clips per split, 1 round, 1 epoch, 2 seeds) to check shapes.
if os.environ.get("QUICK") == "1":
    MAX_CLIPS, ROUNDS, LOCAL_EPOCHS, SEEDS, EXTRA_SEEDS = 4, 1, 1, range(42, 44), range(42, 43)


def set_seed(seed=SEED):
    """Fix every random source so runs reproduce."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
