"""
CNN — Convolutional Neural Network for CIFAR-10.

Architecture:
    Input  (B, 3, 32, 32)
    ConvBlock(3 → 32)   MaxPool/2  → (B,  32, 16, 16)
    ConvBlock(32 → 64)  MaxPool/2  → (B,  64,  8,  8)
    ConvBlock(64 → 128) MaxPool/2  → (B, 128,  4,  4)
    AdaptiveAvgPool → (B, 128, 1, 1) → flatten (B, 128)
    Linear(128 → num_classes)

ConvBlock = Conv2d + BatchNorm2d + ReLU + MaxPool2d

Teaches:
  • Conv2d   — local receptive field, weight sharing across spatial positions
  • BatchNorm2d — normalise over (N,H,W) per channel; has running stats (buffers)
  • MaxPool2d   — halves spatial size, keeps strongest activation in each window
  • AdaptiveAvgPool2d(1) — global average pooling; collapses H,W to 1,1
    regardless of input size → model is input-size agnostic

Why global avg pool instead of flatten → linear?
  • Much fewer parameters (128 vs 128*4*4=2048 → head input)
  • Forces the network to learn spatially meaningful features
  • Less prone to overfitting on small datasets like CIFAR-10
"""

import torch.nn as nn
from .base import BaseModel


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class CNN(BaseModel):
    def __init__(self, in_channels: int = 3, num_classes: int = 10):
        super().__init__()

        self.features = nn.Sequential(
            _conv_block(in_channels,  32),   # → (B,  32, H/2, W/2)
            _conv_block(32,           64),   # → (B,  64, H/4, W/4)
            _conv_block(64,          128),   # → (B, 128, H/8, W/8)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)  # → (B, 128, 1, 1)
        self.classifier = nn.Linear(128, num_classes)

        self.init_weights()

    def forward(self, x):
        x = self.features(x)       # spatial feature extraction
        x = self.pool(x)           # global average pool
        x = x.flatten(1)           # (B, 128)
        return self.classifier(x)  # (B, num_classes) — raw logits
