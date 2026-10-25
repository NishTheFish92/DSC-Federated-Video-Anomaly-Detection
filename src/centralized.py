"""Baseline: one model trained on every site's normal frames pooled together."""
from torch.utils.data import ConcatDataset, DataLoader

from src.config import BATCH_SIZE, DEVICE, LOCAL_EPOCHS, ROUNDS
from src.model import AutoEncoder
from src.train_local import train


def train_centralized(loaders, channels=1):
    """Same architecture and same total epochs (ROUNDS x LOCAL_EPOCHS) as the federated run."""
    pooled = ConcatDataset([loader.dataset for loader in loaders.values()])
    loader = DataLoader(pooled, batch_size=BATCH_SIZE, shuffle=True)
    model = AutoEncoder(channels).to(DEVICE)
    loss = train(model, loader, ROUNDS * LOCAL_EPOCHS)
    print(f"centralised final loss={loss:.5f}")
    return model
