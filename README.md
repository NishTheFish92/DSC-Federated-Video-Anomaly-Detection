# Federated video anomaly detection (UCSD Ped1 + Ped2)

Two simulated camera sites (Site 1 = Ped1, Site 2 = Ped2) each train a small conv
autoencoder on normal frames; FedAvg averages their weights. A frame's reconstruction
error is its anomaly score. Compared against a centralised baseline by frame-level AUC.

## Run
```bash
# 1. Data (~700 MB) into data/
mkdir -p data && curl -L http://www.svcl.ucsd.edu/projects/anomaly/UCSD_Anomaly_Dataset.tar.gz | tar -xz -C data

# 2. Quick shape check on 2 clips per split (seconds)
QUICK=1 uv run run_federated.py && QUICK=1 uv run run_centralized.py

# 3. Full experiment (~1 min each on a GTX 1650 Ti)
uv run run_federated.py
uv run run_centralized.py   # also prints the comparison table and saves the plot
```
On Colab: install uv with `!curl -LsSf https://astral.sh/uv/install.sh | sh`, then run the same commands from the project folder.

## Results (seed 42, 10 rounds x 3 local epochs, 64x64 frames)
| Frame-level AUC | Federated | Centralised | Gap |
|---|---|---|---|
| Site 1 (Ped1) | 0.709 | 0.689 | -0.020 |
| Site 2 (Ped2) | 0.721 | 0.771 | +0.050 |
| Mean | 0.715 | 0.730 | +0.015 |

Communication cost: 329,025 params x 4 B x 2 x 2 sites x 10 rounds = 52.6 MB.
Outputs: `results/federated.json`, `results/centralized.json`, `results/auc_comparison.png`.

## Notes
- Anomaly scores are min-max normalised within each test clip before computing AUC (standard on UCSD).
- `Ped1/Test/Test017/142.tif` is corrupt in the official archive; it is replaced by the previous frame.
