"""Centralised baseline on pooled normal frames, then compare with the federated run."""
import json

import torch

from src.centralized import train_centralized
from src.config import RESULTS_DIR, SITES, set_seed
from src.data import test_clips, train_loader
from src.evaluate import compare, evaluate_sites

set_seed()
# Pooling every site's data is exactly what federated learning avoids; this is only a reference.
loaders = {site: train_loader(path) for site, path in SITES.items()}
model = train_centralized(loaders)
aucs = evaluate_sites(model, {site: test_clips(path) for site, path in SITES.items()})
print("AUC:", {k: round(v, 4) for k, v in aucs.items()})

RESULTS_DIR.mkdir(exist_ok=True)
torch.save(model.state_dict(), RESULTS_DIR / "centralized_model.pt")
(RESULTS_DIR / "centralized.json").write_text(json.dumps({"auc": aucs}, indent=2))
if (RESULTS_DIR / "federated.json").exists():
    compare(RESULTS_DIR)
