# training/augmentation.py
"""
Train on level-0 frames but augment with random simulated distortions so
the detector generalises to levels 1–5 without re-collecting labelled data.
Theory: data augmentation as a regulariser (Goodfellow et al. Ch. 7).

All functions operate on a CHW float32 tensor in [0, 1].
Implementations mirror the transforms in visual_utils.py (which works on
WHC uint8 numpy arrays) but are re-expressed for the PyTorch tensor format
used by the training pipeline.
"""
import math
import random

import torch
import torch.nn.functional as F


def add_stars(img: torch.Tensor) -> torch.Tensor:
    """Scatter bright sparkle points across the frame."""
    _, H, W = img.shape
    n = random.randint(50, 200)
    out = img.clone()

    xs = torch.randint(0, W, (n,))
    ys = torch.randint(0, H, (n,))
    b = torch.empty(n).uniform_(0.8, 1.0)

    # Centre pixel
    out[:, ys, xs] = b.unsqueeze(0).expand(3, -1)
    # Cross arms (radius 1–2 px) — matches visual_utils add_random_stars_transform
    for d in (1, 2):
        out[:, (ys - d).clamp(0, H - 1), xs] = b.unsqueeze(0).expand(3, -1)
        out[:, (ys + d).clamp(0, H - 1), xs] = b.unsqueeze(0).expand(3, -1)
        out[:, ys, (xs - d).clamp(0, W - 1)] = b.unsqueeze(0).expand(3, -1)
        out[:, ys, (xs + d).clamp(0, W - 1)] = b.unsqueeze(0).expand(3, -1)

    return out


def add_clouds(img: torch.Tensor) -> torch.Tensor:
    """Overlay soft semi-transparent cloud patches."""
    _, H, W = img.shape

    # Generate low-res noise then blur + upsample (mirrors add_clouds_transform)
    scale = 20
    h0, w0 = max(2, H // scale), max(2, W // scale)
    noise = torch.rand(1, 1, h0, w0)
    for _ in range(3):
        noise = F.avg_pool2d(F.pad(noise, (1, 1, 1, 1), mode="reflect"), 3, stride=1)

    field = F.interpolate(noise, size=(H, W), mode="bilinear", align_corners=False).squeeze()
    field = (field - field.min()) / (field.max() - field.min() + 1e-6)

    coverage = random.uniform(0.20, 0.45)
    thresh = torch.quantile(field, 1.0 - coverage)
    mask = ((field - thresh) / (field.max() - thresh + 1e-6)).clamp(0, 1)
    mask = mask * mask  # softer edges, matches visual_utils

    intensity = random.uniform(0.20, 0.50)
    alpha = (mask * intensity).unsqueeze(0)  # 1×H×W — broadcast over channels
    return (img * (1.0 - alpha) + alpha).clamp(0, 1)


def colour_shift(img: torch.Tensor) -> torch.Tensor:
    """Random brightness + contrast shift (mirrors photometric_jitter_transform)."""
    brightness = random.uniform(0.75, 1.30)
    contrast   = random.uniform(0.75, 1.25)
    out = img * brightness
    out = (out - 0.5) * contrast + 0.5
    return out.clamp(0, 1)


def wave_distort(img: torch.Tensor) -> torch.Tensor:
    """Sinusoidal spatial warp (mirrors heat_haze_warp_transform in visual_utils)."""
    _, H, W = img.shape
    amplitude  = random.uniform(3.0, 10.0)   # pixels
    wavelength = random.uniform(60.0, 130.0)  # pixels
    phase_x    = random.uniform(0, 2 * math.pi)
    phase_y    = random.uniform(0, 2 * math.pi)

    # Pixel-space coordinate grids
    ys = torch.arange(H, dtype=torch.float32)  # (H,)
    xs = torch.arange(W, dtype=torch.float32)  # (W,)

    # dx varies with y; dy varies with x — same formula as visual_utils
    dx = amplitude * torch.sin(ys / wavelength * 2 * math.pi + phase_x)  # (H,)
    dy = amplitude * torch.sin(xs / wavelength * 2 * math.pi + phase_y)  # (W,)

    grid_x = xs.unsqueeze(0).expand(H, W)  # (H, W)
    grid_y = ys.unsqueeze(1).expand(H, W)  # (H, W)

    src_x = grid_x + dx.unsqueeze(1)  # (H, W)
    src_y = grid_y + dy.unsqueeze(0)  # (H, W)

    # Normalise to [-1, 1] for grid_sample
    src_x_n = (src_x / (W - 1)) * 2 - 1
    src_y_n = (src_y / (H - 1)) * 2 - 1

    grid = torch.stack([src_x_n, src_y_n], dim=-1).unsqueeze(0)  # (1, H, W, 2)
    return F.grid_sample(
        img.unsqueeze(0), grid,
        mode="bilinear", padding_mode="border", align_corners=True
    ).squeeze(0)


def pixelate_zombies(img: torch.Tensor) -> torch.Tensor:
    """Pixelate random rectangular patches (simulates distortion levels 3–4)."""
    _, H, W = img.shape
    out = img.clone()
    for _ in range(random.randint(2, 6)):
        ph = random.randint(20, 60)
        pw = random.randint(20, 60)
        y0 = random.randint(0, max(0, H - ph))
        x0 = random.randint(0, max(0, W - pw))
        bs = random.randint(4, 10)  # block size
        patch = out[:, y0:y0 + ph, x0:x0 + pw].unsqueeze(0)
        small = F.avg_pool2d(patch, bs, stride=bs, ceil_mode=True)
        out[:, y0:y0 + ph, x0:x0 + pw] = F.interpolate(small, size=(ph, pw), mode="nearest").squeeze(0)
    return out


def random_distortion(img: torch.Tensor) -> torch.Tensor:
    """Apply a random subset of distortions. img: 3×H×W float32 [0,1]."""
    if random.random() < 0.4:
        img = add_stars(img)
    if random.random() < 0.4:
        img = add_clouds(img)
    if random.random() < 0.4:
        img = colour_shift(img)
    if random.random() < 0.3:
        img = wave_distort(img)
    if random.random() < 0.3:
        img = pixelate_zombies(img)
    return img.clamp(0, 1)
