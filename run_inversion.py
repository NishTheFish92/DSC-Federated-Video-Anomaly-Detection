"""Gradient-inversion check on Site 2 (Ped2), seed 42. Saves results/inversion.json and Figure 4.

The attack is tried on two things the server could see, at two points in training:
  - the gradient of a single batch of B real frames (the easiest case for the attacker)
  - one full round's weight update (3 local epochs, what FedAvg actually sends)
at round 1 (untrained global model) and after round 10 (results/federated_model.pt).
Run run_federated.py first.
    uv run run_inversion.py
"""
import copy
import json

import matplotlib
matplotlib.use("Agg")  # save plots to file, no display needed
import matplotlib.pyplot as plt
import torch

from src.config import DEVICE, LOCAL_EPOCHS, RESULTS_DIR, SEED, SITES, set_seed
from src.data import train_loader
from src.inversion import flat_grad, flat_update, invert, psnr_to_nearest
from src.model import AutoEncoder
from src.train_local import train


def attack(model, target, compare_to, n_frames):
    """Run the inversion; PSNR of each dummy against its closest frame in compare_to."""
    x, cos = invert(model, target, n_frames, SEED)
    psnr, idx = psnr_to_nearest(x.cpu(), compare_to)
    return x.cpu(), {"cosine": cos, "psnr": psnr, "nearest": idx}


def main():
    set_seed()
    loader = train_loader(SITES["site2"])
    frames = loader.dataset.tensors[0]
    mean_frame = frames.mean(0, keepdim=True)
    batch = next(iter(loader))[0][:8]  # one real shuffled batch of Site 2 frames
    trained = AutoEncoder().to(DEVICE)
    trained.load_state_dict(torch.load(RESULTS_DIR / "federated_model.pt", map_location=DEVICE))
    states = {"round1": AutoEncoder().to(DEVICE), "round11": trained}

    results = {"mean_frame_psnr_to_batch": psnr_to_nearest(batch, mean_frame)[0],
               "mean_frame_psnr_to_nearest": psnr_to_nearest(mean_frame, frames)[0][0]}
    images = {"batch": batch, "mean_frame": mean_frame}
    for state, model in states.items():
        local = copy.deepcopy(model)
        train(local, loader, LOCAL_EPOCHS)  # one round of Site 2's local training
        update = -flat_update(model, local).detach()
        for n in (1, 8):
            grad = flat_grad(model, batch[:n].to(DEVICE)).detach()
            key = f"{state}_batch{n}"
            images[key], results[key] = attack(model, grad, batch[:n], n)
            key = f"{state}_update{n}"
            images[key], results[key] = attack(model, update, frames, n)
            print(key, {k: results[key][k] for k in ("cosine", "psnr")})

    (RESULTS_DIR / "inversion.json").write_text(json.dumps(results, indent=1))
    plot(images, results, frames, RESULTS_DIR / "figure4_inversion.png")


def plot(images, results, frames, path):
    """Row per attack: the target frame(s) the server would like to see, then the reconstructions."""
    rows = [("Real frame", images["batch"][0], None),
            ("Mean training frame", images["mean_frame"][0], results["mean_frame_psnr_to_batch"][0]),
            ("From one batch's gradient\n(B = 1, round 1)", images["round1_batch1"][0],
             results["round1_batch1"]["psnr"][0]),
            ("From one round's update\n(B = 1, round 1)", images["round1_update1"][0],
             results["round1_update1"]["psnr"][0]),
            ("Closest training frame\nto the image on the left", frames[results["round1_update1"]["nearest"][0]],
             None)]
    plt.rcParams["font.family"] = "serif"
    fig, axes = plt.subplots(1, len(rows), figsize=(2.6 * len(rows), 3.2))
    for ax, (title, img, psnr) in zip(axes, rows):
        ax.imshow(img[0].numpy(), cmap="gray", vmin=0, vmax=1)
        ax.set_title(title + (f"\nPSNR {psnr:.1f} dB" if psnr is not None else "\n "), fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
