# models/cnn_encoder.py
"""
DQN-style CNN encoder (Mnih et al. 2015 architecture).
Theory anchor: LeCun et al. Nature 2015 — local connections, shared
weights, pooling, depth. We use stride-2 convs instead of pooling
because that's what works best for RL pixel inputs (Mnih et al.).
"""
import torch.nn as nn


class CNNEncoder(nn.Module):
    def __init__(self, in_channels=4, out_dim=512):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        # 84×84×4 input → 7×7×64 = 3136 features
        self.fc = nn.Sequential(nn.Linear(3136, out_dim), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.fc(self.convs(x))