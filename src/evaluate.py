"""Score frames by reconstruction error and compute frame-level ROC-AUC."""
import json

import matplotlib
matplotlib.use("Agg")  # save plots to file, no display needed
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from src.config import BATCH_SIZE, DEVICE


@torch.no_grad()
def frame_errors(model, frames):
    """Mean squared reconstruction error of each frame."""
    model.eval()
    errors = []
    for batch in frames.split(BATCH_SIZE):
        batch = batch.to(DEVICE)
        errors.append(((model(batch) - batch) ** 2).mean(dim=(1, 2, 3)).cpu())
    return torch.cat(errors).numpy()


def site_auc(model, clips):
    """Frame-level AUC over all test clips of one site.

    Errors are min-max normalised within each clip (standard practice on UCSD) so that
    clips with brighter or busier scenes do not dominate the threshold.
    """
    scores, labels = [], []
    for frames, clip_labels in clips:
        e = frame_errors(model, frames)
        scores.append((e - e.min()) / (e.max() - e.min() + 1e-8))
        labels.append(clip_labels)
    return roc_auc_score(np.concatenate(labels), np.concatenate(scores))


def evaluate_sites(model, test_sets):
    """{site: AUC} plus their mean."""
    aucs = {site: site_auc(model, clips) for site, clips in test_sets.items()}
    aucs["mean"] = float(np.mean(list(aucs.values())))
    return aucs


def compare(results_dir):
    """Print federated vs centralised AUC and save a bar chart."""
    fed = json.loads((results_dir / "federated.json").read_text())["auc"]
    cen = json.loads((results_dir / "centralized.json").read_text())["auc"]
    print(f"\n{'':8}{'federated':>11}{'centralised':>13}{'gap':>8}")
    for key in fed:
        print(f"{key:8}{fed[key]:11.4f}{cen[key]:13.4f}{cen[key] - fed[key]:8.4f}")

    x = np.arange(len(fed))
    plt.figure(figsize=(6, 4))
    plt.bar(x - 0.2, list(fed.values()), 0.4, label="Federated (FedAvg)")
    plt.bar(x + 0.2, list(cen.values()), 0.4, label="Centralised")
    plt.xticks(x, list(fed.keys()))
    plt.ylim(0.5, 1.0)
    plt.ylabel("Frame-level ROC-AUC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "auc_comparison.png", dpi=150)
