"""Federated run: FedAvg over two camera sites, then evaluate the global model."""
import json

import torch

from src.config import RESULTS_DIR, SITES, set_seed
from src.data import test_clips, train_loader
from src.evaluate import evaluate_sites
from src.federated import communication_bytes, run_fedavg

set_seed()
loaders = {site: train_loader(path) for site, path in SITES.items()}
for site, loader in loaders.items():
    print(f"{site}: {len(loader.dataset)} training frames")

model, history = run_fedavg(loaders)
aucs = evaluate_sites(model, {site: test_clips(path) for site, path in SITES.items()})
comm = communication_bytes(model, len(SITES))
print("AUC:", {k: round(v, 4) for k, v in aucs.items()})
print(f"communication cost: {comm / 1e6:.1f} MB")

RESULTS_DIR.mkdir(exist_ok=True)
torch.save(model.state_dict(), RESULTS_DIR / "federated_model.pt")
(RESULTS_DIR / "federated.json").write_text(json.dumps(
    {"auc": aucs, "communication_bytes": comm, "round_losses": history}, indent=2))
