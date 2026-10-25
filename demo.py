"""Demo: load the saved federated and centralised models and show their test results.

No training happens here. Run after run_federated.py and run_centralized.py:
    uv run demo.py             # every site, clip 1 shown in detail
    uv run demo.py --clip 4    # show test clip 4 of each site in detail
"""
import argparse

import matplotlib
matplotlib.use("Agg")  # save plots to file, no display needed
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve

from src.config import DEVICE, RESULTS_DIR, SITES
from src.data import test_clips
from src.evaluate import clip_scores, site_scores
from src.model import AutoEncoder

MODELS = {"Federated": "federated_model.pt", "Centralised": "centralized_model.pt"}


def load_model(filename):
    model = AutoEncoder().to(DEVICE)
    model.load_state_dict(torch.load(RESULTS_DIR / filename, map_location=DEVICE))
    return model.eval()


def print_auc_table(aucs):
    """aucs[model][site] -> AUC. Prints one row per site plus the mean and the gap."""
    sites = list(SITES) + ["mean"]
    print(f"\n{'site':8}" + "".join(f"{name:>13}" for name in aucs) + f"{'gap':>9}")
    for site in sites:
        row = [aucs[name][site] for name in aucs]
        print(f"{site:8}" + "".join(f"{v:13.4f}" for v in row) + f"{row[1] - row[0]:9.4f}")
    print("(gap = centralised - federated; positive means centralised is better)")


def plot_roc(scores, path):
    """One panel per site, one ROC curve per model."""
    fig, axes = plt.subplots(1, len(SITES), figsize=(5 * len(SITES), 4.5))
    for ax, site in zip(axes, SITES):
        for name, (labels, s) in scores[site].items():
            fpr, tpr, _ = roc_curve(labels, s)
            ax.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(labels, s):.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Chance")
        ax.set(title=f"{site} ({SITES[site].name})", xlabel="False positive rate",
               ylabel="True positive rate")
        ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def plot_clip(models, frames, labels, title, path):
    """Score timeline for one clip (anomalous frames shaded), plus the most anomalous frame."""
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(2, 1, 1)
    for name, model in models.items():
        ax.plot(clip_scores(model, frames), label=name)
    # Shade the ground-truth anomalous frames so the viewer can see the scores rise there.
    ax.fill_between(np.arange(len(labels)), 0, 1, where=labels == 1, color="red",
                    alpha=0.15, step="mid", label="Ground-truth anomaly")
    ax.set(title=title, xlabel="Frame", ylabel="Anomaly score", ylim=(0, 1.05))
    ax.legend(loc="upper left")

    # Bottom row: input, federated reconstruction, and its error map at the highest-scoring
    # anomalous frame (or the highest-scoring frame overall if the clip has no anomaly).
    fed = models["Federated"]
    s = clip_scores(fed, frames)
    peak = int(np.argmax(np.where(labels == 1, s, -1))) if labels.any() else int(np.argmax(s))
    with torch.no_grad():
        frame = frames[peak:peak + 1].to(DEVICE)
        recon = fed(frame)
    panels = [(frame, "gray", f"Input (frame {peak + 1})"),
              (recon, "gray", "Federated reconstruction"),
              ((frame - recon) ** 2, "hot", "Squared error")]
    for i, (img, cmap, name) in enumerate(panels):
        a = fig.add_subplot(2, 3, 4 + i)
        a.imshow(img[0, 0].cpu().numpy(), cmap=cmap)
        a.set_title(name)
        a.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--clip", type=int, default=1, help="test clip to show in detail (1-indexed)")
    args = parser.parse_args()

    models = {name: load_model(f) for name, f in MODELS.items()}
    print(f"loaded {', '.join(MODELS.values())} from {RESULTS_DIR}/ on {DEVICE}")

    aucs = {name: {} for name in models}
    scores = {}
    for site, path in SITES.items():
        clips = test_clips(path)
        n_frames = sum(len(labels) for _, labels in clips)
        n_anom = sum(labels.sum() for _, labels in clips)
        print(f"{site}: {len(clips)} test clips, {n_frames} frames, {n_anom} anomalous")

        scores[site] = {name: site_scores(m, clips) for name, m in models.items()}
        for name in models:
            aucs[name][site] = roc_auc_score(*scores[site][name])

        frames, labels = clips[min(args.clip, len(clips)) - 1]
        out = RESULTS_DIR / f"demo_{site}_clip{args.clip:02d}.png"
        plot_clip(models, frames, labels, f"{site} test clip {args.clip}: anomaly score per frame", out)
        print(f"  saved {out}")

    for name in models:
        aucs[name]["mean"] = float(np.mean([aucs[name][s] for s in SITES]))
    print_auc_table(aucs)

    plot_roc(scores, RESULTS_DIR / "demo_roc.png")
    print(f"\nsaved {RESULTS_DIR / 'demo_roc.png'}")


if __name__ == "__main__":
    main()
