"""FedAvg: sites train locally, the server averages their weights."""
import copy

from src.config import DEVICE, LOCAL_EPOCHS, ROUNDS
from src.model import AutoEncoder
from src.train_local import train


def average_weights(states, sizes):
    """Weighted mean of state dicts; a site with more frames gets more say."""
    total = sum(sizes)
    return {k: sum(s[k] * (n / total) for s, n in zip(states, sizes)) for k in states[0]}


def run_fedavg(loaders, channels=1, mu=0.0, after_round=None):
    """loaders: {client_name: DataLoader}. Returns the final global model and per-round losses.

    mu > 0 runs FedProx instead of plain FedAvg. after_round(global_model), if given, is
    called at the end of every round (used to record AUC round by round).
    """
    global_model = AutoEncoder(channels).to(DEVICE)
    history = []
    for r in range(1, ROUNDS + 1):
        states, sizes, losses = [], [], {}
        for site, loader in loaders.items():
            local = copy.deepcopy(global_model)  # "download" the global weights
            losses[site] = train(local, loader, LOCAL_EPOCHS, mu)
            states.append(local.state_dict())  # "upload" the local weights
            sizes.append(len(loader.dataset))
        global_model.load_state_dict(average_weights(states, sizes))
        history.append(losses)
        if after_round:
            after_round(global_model)
        print(f"round {r}/{ROUNDS} " + " ".join(f"{s} loss={l:.5f}" for s, l in losses.items()))
    return global_model, history


def communication_bytes(model, n_sites):
    """Parameters x 4 bytes (float32) x 2 (up and down) x sites x rounds."""
    n_params = sum(p.numel() for p in model.parameters())
    return n_params * 4 * 2 * n_sites * ROUNDS
