"""
MiniResNet — ResNet-18 style network for Tiny ImageNet (64×64, 200 classes).

Architecture mirrors torchvision's ResNet-18 but uses a 3×3 stem instead of
the 7×7 stride-2 conv that is designed for 224×224 images.

Stages:
    stem   Conv(3→64, 3×3, s=1) + BN + ReLU          (B, 64, 64, 64)
    layer1 2 × BasicBlock(64→64,   stride=1)          (B,  64, 64, 64)
    layer2 2 × BasicBlock(64→128,  stride=2)          (B, 128, 32, 32)
    layer3 2 × BasicBlock(128→256, stride=2)          (B, 256, 16, 16)
    layer4 2 × BasicBlock(256→512, stride=2)          (B, 512,  8,  8)
    AdaptiveAvgPool(1)  → (B, 512)
    Linear(512 → num_classes)

Teaches:
  • Residual connections — why skip connections solve vanishing gradients
  • Downsampling with stride — vs MaxPool for spatial reduction
  • Projection shortcuts — 1×1 conv to match channel dims when they change
  • BatchNorm placement — after conv, before activation (He et al. convention)

The skip connection (residual):
    out = F(x, {W_i}) + x     ← if dimensions match (same channels, same stride)
    out = F(x, {W_i}) + proj(x)  ← if dimensions differ (downsample block)

This means gradients can flow directly from loss to early layers via the
identity shortcut, bypassing the conv stack entirely. Very deep networks
(50, 100, 1000 layers) become trainable this way.
"""

import torch
import torch.nn as nn
from .base import BaseModel


class BasicBlock(nn.Module):
    """Two 3×3 convolutions with a residual shortcut."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

        # Projection shortcut: needed when spatial size or channels change
        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)          # might be projected
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))      # no activation before residual add
        out = self.relu(out + identity)      # add skip, then activate
        return out


def _make_layer(in_ch: int, out_ch: int, num_blocks: int, stride: int) -> nn.Sequential:
    """Build one ResNet stage: first block may downsample, rest are identity."""
    blocks = [BasicBlock(in_ch, out_ch, stride=stride)]
    for _ in range(1, num_blocks):
        blocks.append(BasicBlock(out_ch, out_ch, stride=1))
    return nn.Sequential(*blocks)


class MiniResNet(BaseModel):
    def __init__(self, num_classes: int = 200, in_channels: int = 3):
        super().__init__()

        # 3×3 stem: better suited to 64×64 input than the original 7×7 s=2
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.layer1 = _make_layer(64,  64,  num_blocks=2, stride=1)
        self.layer2 = _make_layer(64,  128, num_blocks=2, stride=2)
        self.layer3 = _make_layer(128, 256, num_blocks=2, stride=2)
        self.layer4 = _make_layer(256, 512, num_blocks=2, stride=2)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Linear(512, num_classes)

        self.init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)
