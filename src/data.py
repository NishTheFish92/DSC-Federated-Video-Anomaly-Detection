"""Load UCSD clips, preprocess frames, build per-site DataLoaders.

Each function takes one site's folder, so a site only ever reads its own data.
"""
import re

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from src.config import BATCH_SIZE, IMAGE_SIZE, MAX_CLIPS


def load_frame(path):
    """Grayscale, resize to IMAGE_SIZE x IMAGE_SIZE, scale to [0, 1]."""
    img = Image.open(path).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def clip_dirs(split_dir):
    """Clip folders like Train001 / Test001, skipping pixel-mask folders like Test001_gt."""
    dirs = sorted(d for d in split_dir.iterdir() if re.fullmatch(r"(Train|Test)\d+", d.name))
    return dirs[:MAX_CLIPS]


def load_clip(clip_dir):
    """All frames of one clip as a tensor of shape (frames, 1, H, W)."""
    # "[0-9]*" skips the macOS "._" junk files shipped inside the archive.
    frames = []
    for path in sorted(clip_dir.glob("[0-9]*.tif")):
        try:
            frames.append(load_frame(path))
        except OSError:
            # One frame in the official archive (Ped1 Test017/142.tif) is corrupt.
            # Repeat the previous frame so frame indices still line up with the labels.
            print(f"warning: unreadable frame {path}, using previous frame")
            frames.append(frames[-1])
    return torch.from_numpy(np.stack(frames)).unsqueeze(1)


def train_loader(site_dir):
    """Shuffled DataLoader over every frame of the site's normal training clips."""
    frames = torch.cat([load_clip(d) for d in clip_dirs(site_dir / "Train")])
    return DataLoader(TensorDataset(frames), batch_size=BATCH_SIZE, shuffle=True)


def anomaly_ranges(site_dir):
    """Parse the dataset's .m file, e.g. 'gt_frame = [61:180, 190:200]', into lists of ranges."""
    text = (site_dir / "Test" / f"{site_dir.name}.m").read_text()  # e.g. UCSDped1.m
    clips = re.findall(r"gt_frame\s*=\s*\[([^\]]*)\]", text)
    return [[tuple(map(int, part.split(":"))) for part in clip.split(",")] for clip in clips]


def test_clips(site_dir):
    """List of (frames, labels) per test clip; label 1 = anomalous frame (1-indexed in the .m file)."""
    clips = []
    for clip_dir, ranges in zip(clip_dirs(site_dir / "Test"), anomaly_ranges(site_dir)):
        frames = load_clip(clip_dir)
        labels = np.zeros(len(frames), dtype=int)
        for start, end in ranges:
            labels[start - 1:end] = 1
        clips.append((frames, labels))
    return clips
