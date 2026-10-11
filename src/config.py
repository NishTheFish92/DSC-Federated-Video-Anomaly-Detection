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

# QUICK=1 runs a tiny version (2 clips per split, 1 round, 1 epoch) to check shapes.
if os.environ.get("QUICK") == "1":
    MAX_CLIPS, ROUNDS, LOCAL_EPOCHS = 2, 1, 1


def set_seed(seed=SEED):
    """Fix every random source so runs reproduce."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
