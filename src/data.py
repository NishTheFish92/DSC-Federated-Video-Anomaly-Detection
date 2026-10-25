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


def stack_frames(frames, length):
    """(N, 1, H, W) clip -> (N - length + 1, length, H, W): every run of `length` consecutive frames."""
    if length == 1:
        return frames
    return frames[:, 0].unfold(0, length, 1).permute(0, 3, 1, 2)


def clips_loader(dirs, stack=1):
    """Shuffled DataLoader over the given training clips (stacks never cross clip boundaries)."""
    frames = torch.cat([stack_frames(load_clip(d), stack) for d in dirs])
    return DataLoader(TensorDataset(frames), batch_size=BATCH_SIZE, shuffle=True)


def train_loader(site_dir, stack=1):
    """Shuffled DataLoader over every frame of the site's normal training clips."""
    return clips_loader(clip_dirs(site_dir / "Train"), stack)


def client_loaders(site_dir, n_clients):
    """Split one site's training clips into n_clients contiguous groups, one DataLoader each."""
    dirs = clip_dirs(site_dir / "Train")
    return [clips_loader(list(group)) for group in np.array_split(dirs, n_clients)]


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
