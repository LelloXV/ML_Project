# Task_3/models/cnn_encoder.py
"""
DQN-style CNN encoder for KAZ pixel observations.

Architecture (Mnih et al. 2015 "Methods" §"Network architecture"):
    Input:  (B, K, H, W)              K = frame stack channels
    Conv2d(K, 32, kernel=8, stride=4) + ReLU
    Conv2d(32, 64, kernel=4, stride=2) + ReLU
    Conv2d(64, 64, kernel=3, stride=1) + ReLU
    Flatten
    Linear(?, 512) + ReLU
    Output: (B, 512)                  shared feature vector

The 512-dim feature vector is consumed by separate policy and value
heads inside the PPO RLModule (Step 3). Sharing the trunk between actor
and critic is a standard design (Sutton & Barto Ch. 13.5).

Note on input size: the original DQN paper used 84x84 -> 7x7x64 = 3136
features after the conv stack. We use 64x64 -> 4x4x64 = 1024 to save
compute on Apple Silicon. The architecture is otherwise identical.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class CNNEncoder(nn.Module):

    def __init__(self,
                 in_channels: int = 4,
                 out_dim: int = 512,
                 input_hw: tuple[int, int] = (64, 64)):
        super().__init__()
        self.in_channels = in_channels
        self.out_dim = out_dim

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )

        # Compute the flattened feature size dynamically so this class
        # auto-adapts if input_hw changes (e.g. switching to 84x84).
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, *input_hw)
            n_flat = self.conv(dummy).shape[1]
        self.flat_dim = n_flat

        self.fc = nn.Sequential(
            nn.Linear(n_flat, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) tensor of stacked frames, values in [0, 1].
        Returns:
            (B, out_dim) feature vector.
        """
        return self.fc(self.conv(x))

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)