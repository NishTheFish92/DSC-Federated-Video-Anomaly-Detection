"""Gradient-inversion check: can the server rebuild a site's frames from what the site sends?

Deep leakage from gradients (Zhu et al.) with the cosine loss of Geiping et al.: start from
random dummy frames and adjust them until the gradient they produce points the same way as
the gradient (or weight update) the server observed. If the dummies end up looking like real
training frames, the shared update leaks those frames.
"""
import torch
import torch.nn.functional as F

from src.config import DEVICE, IMAGE_SIZE

STEPS = 1000
TV_WEIGHT = 1e-2  # total-variation prior: nudges dummies towards smooth, image-like frames


def flat_grad(model, x, create_graph=False):
    """Gradient of the reconstruction loss on frames x, flattened into one vector."""
    loss = F.mse_loss(model(x), x)
    grads = torch.autograd.grad(loss, list(model.parameters()), create_graph=create_graph)
    return torch.cat([g.flatten() for g in grads])


def flat_update(before, after):
    """after - before, flattened. Minus this points along the gradient the site followed."""
    return torch.cat([(a - b).flatten() for b, a in zip(before.parameters(), after.parameters())])


def total_variation(x):
    return (x[..., 1:, :] - x[..., :-1, :]).abs().mean() + (x[..., 1:] - x[..., :-1]).abs().mean()


def invert(model, target, n_frames, seed):
    """n_frames dummy frames whose gradient best matches target (by cosine similarity)."""
    model.eval()
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    z = torch.randn(n_frames, 1, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE, generator=gen)
    z.requires_grad_(True)
    optimizer = torch.optim.Adam([z], lr=0.1)
    for _ in range(STEPS):
        x = torch.sigmoid(z)  # keeps every pixel in [0, 1], like the real frames
        g = flat_grad(model, x, create_graph=True)
        loss = 1 - F.cosine_similarity(g, target, dim=0) + TV_WEIGHT * total_variation(x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    x = torch.sigmoid(z).detach()
    return x, F.cosine_similarity(flat_grad(model, x), target, dim=0).item()


def psnr_to_nearest(images, frames):
    """For each image, its PSNR in dB against the closest frame in frames (peak value 1)."""
    mse = ((images.flatten(1)[:, None] - frames.flatten(1)[None]) ** 2).mean(-1)
    best, idx = mse.min(dim=1)
    return (-10 * torch.log10(best)).tolist(), idx.tolist()
