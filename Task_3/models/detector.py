# models/detector.py
import torch
import torch.nn as nn


class TinyZombieDetector(nn.Module):
    """
    Single-stage anchor-free detector.
    Backbone: 4 conv blocks. Head: per-cell (objectness, dx, dy, dw, dh).
    Loss: BCE on objectness + Smooth-L1 on box regression.
    Theory: this is the YOLO family idea (Redmon et al. 2016) reduced
    to its essentials. The CNN architecture justification comes from
    LeCun et al. Nature 2015 §"Convolutional neural networks".
    """
    def __init__(self, in_channels=3, grid_size=13):
        super().__init__()
        def block(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1),
                nn.BatchNorm2d(o),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
        self.backbone = nn.Sequential(
            block(in_channels, 32),
            block(32, 64),
            block(64, 128),
            block(128, 256),
        )
        # Output per cell: [objectness, x, y, w, h]
        self.head = nn.Conv2d(256, 5, kernel_size=1)
        self.grid_size = grid_size

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat)  # B × 5 × G × G