# CLAUDE.md

## Project
Federated video anomaly detection for a course project. Two simulated camera sites
(UCSD Ped1 and Ped2) each train a small convolutional autoencoder on normal footage.
Weights are averaged with FedAvg. We compare the federated model against a centralised
baseline using frame-level AUC. Goal: a working, easy-to-explain result, not state of the art.

## Principles
- Keep code simple and readable. Short functions, clear names, comments that explain why.
- Plain PyTorch with a hand-written FedAvg loop. No Flower, no PySyft.
- One responsibility per module. Avoid classes unless needed (the model is the only one).
- No GANs, no future-frame prediction, no memory modules, no extra datasets.
- When the choice is between clever and obvious, pick obvious.

## Data
- Site 1 = Ped1, Site 2 = Ped2. A site only ever reads its own folder.
- Train on normal training clips only. Labels are used only for evaluation.
- Preprocessing: grayscale, resize to 64x64, scale to [0, 1].
- Datasets live in `data/` and are gitignored.

## Structure
```
src/
  config.py       # all settings in one place (image size, rounds, epochs, lr, seed)
  data.py         # load clips, preprocess frames, build per-site DataLoaders
  model.py        # small conv autoencoder (encoder + decoder)
  train_local.py  # train one site's model for a few epochs
  federated.py    # FedAvg: average weights, run the round loop
  centralized.py  # baseline: same model trained on pooled normal frames
  evaluate.py     # reconstruction error per frame, frame-level AUC
run_federated.py  # entry point for the federated run
run_centralized.py# entry point for the baseline
results/          # saved metrics and plots
```

## Pipeline
1. Load and preprocess each site's data locally.
2. Each round: send global weights to both sites, each trains locally, sends weights back.
3. Server averages the weights, weighted by number of training frames.
4. Repeat for 10 rounds, 3 local epochs per round.
5. Evaluate the final global model on each site's test clips (frame-level AUC).
6. Train the centralised baseline with the same architecture and same total epochs.
7. Report AUC for both, the gap, and the communication cost.

## Metrics
- Frame-level ROC-AUC: federated vs centralised, per site and averaged.
- Communication cost = parameters x 4 bytes x 2 (up and down) x sites x rounds.

## Rules for changes
- Fix the random seed in config.py so results reproduce.
- Use `uv` for environment and packages: `uv add <pkg>` to install, `uv run <script>` to run. No pip or manual venvs.
- Do not add dependencies without asking.
- Keep each file under about 150 lines. If one grows past that, ask before splitting.
- Test on a small subset first to check tensor shapes before full training.
- Tools: Python, PyTorch, scikit-learn (AUC), matplotlib. Must also run on free Colab.

## Out of scope
Live deployment, anomaly type classification, UCF-Crime, Avenue, ShanghaiTech,
differential privacy, real edge devices.
