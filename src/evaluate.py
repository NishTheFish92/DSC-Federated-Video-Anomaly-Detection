"""Score frames by reconstruction error and compute frame-level ROC-AUC."""
import json

import matplotlib
matplotlib.use("Agg")  # save plots to file, no display needed
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from src.config import BATCH_SIZE, DEVICE
from src.data import stack_frames


@torch.no_grad()
def frame_errors(model, frames):
    """Mean squared reconstruction error of each frame.

    A frame-stack model sees the model.channels frames ending at frame t, and frame t takes
    that stack's error. The first frames of a clip have no full stack, so they reuse the
    first stack's error.
    """
    model.eval()
    errors = []
    for batch in stack_frames(frames, model.channels).split(BATCH_SIZE):
        batch = batch.to(DEVICE)
        errors.append(((model(batch) - batch) ** 2).mean(dim=(1, 2, 3)).cpu())
    e = torch.cat(errors).numpy()
    return np.concatenate([np.repeat(e[:1], model.channels - 1), e])


def clip_scores(model, frames):
    """Anomaly score per frame: reconstruction error min-max normalised within the clip.

    Normalising per clip (standard practice on UCSD) stops clips with brighter or
    busier scenes from dominating the threshold.
    """
    e = frame_errors(model, frames)
    return (e - e.min()) / (e.max() - e.min() + 1e-8)


def site_scores(model, clips):
    """(labels, scores) for every test frame of one site, clips concatenated."""
    labels = np.concatenate([clip_labels for _, clip_labels in clips])
    scores = np.concatenate([clip_scores(model, frames) for frames, _ in clips])
    return labels, scores


def site_auc(model, clips):
    """Frame-level AUC over all test clips of one site."""
    return roc_auc_score(*site_scores(model, clips))


def site_extra_metrics(model, clips):
    """PR-AUC (average precision) and equal error rate (where false alarms = misses)."""
    labels, scores = site_scores(model, clips)
    fpr, tpr, _ = roc_curve(labels, scores)
    eer = fpr[np.argmin(np.abs((1 - tpr) - fpr))]
    return {"pr_auc": average_precision_score(labels, scores), "eer": float(eer)}


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
