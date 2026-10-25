"""Train one model on one site's data for a few epochs."""
import torch
from torch import nn

from src.config import DEVICE, LR


def train(model, loader, epochs, mu=0.0):
    """Minimise reconstruction error on normal frames. Returns the last epoch's mean loss.

    mu > 0 turns local training into FedProx: the term (mu / 2) * ||w - w_start||^2 keeps
    the weights near the global weights the site received at the start of the round.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    start = [p.detach().clone() for p in model.parameters()]
    model.train()
    for _ in range(epochs):
        total = 0.0
        for (x,) in loader:
            x = x.to(DEVICE)
            recon_loss = loss_fn(model(x), x)
            loss = recon_loss
            if mu > 0:
                loss = loss + mu / 2 * sum(((p - s) ** 2).sum() for p, s in zip(model.parameters(), start))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += recon_loss.item() * len(x)
    return total / len(loader.dataset)
