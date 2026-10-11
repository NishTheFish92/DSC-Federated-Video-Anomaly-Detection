"""Train one model on one site's data for a few epochs."""
import torch
from torch import nn

from src.config import DEVICE, LR


def train(model, loader, epochs):
    """Minimise reconstruction error on normal frames. Returns the last epoch's mean loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        total = 0.0
        for (x,) in loader:
            x = x.to(DEVICE)
            loss = loss_fn(model(x), x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(x)
    return total / len(loader.dataset)
