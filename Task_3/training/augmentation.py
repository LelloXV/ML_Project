# training/augmentations.py
"""
Train on level-0 frames but augment with random simulated distortions
so the detector generalises to levels 1–5 without ever seeing them
labelled. Theory: data augmentation as a regularizer (Goodfellow et al.
Deep Learning, Ch. 7).
"""
import torch, random
import torch.nn.functional as F


def random_distortion(img: torch.Tensor) -> torch.Tensor:
    """img: 3×H×W in [0,1]. Returns distorted image."""
    if random.random() < 0.4:
        img = add_stars(img)
    if random.random() < 0.4:
        img = add_clouds(img)
    if random.random() < 0.4:
        img = colour_shift(img)
    if random.random() < 0.3:
        img = wave_distort(img)
    if random.random() < 0.3:
        img = pixelate_zombies(img)   # needs the boxes to localise
    return img.clamp_(0, 1)