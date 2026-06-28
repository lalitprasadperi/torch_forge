"""
MLP — Multi-Layer Perceptron for MNIST.

Architecture:
    Input  (B, 1, 28, 28)
    Flatten → (B, 784)
    Linear(784 → H0) + BatchNorm1d + ReLU + Dropout
    Linear(H0  → H1) + BatchNorm1d + ReLU + Dropout
    ...
    Linear(Hn → num_classes)

Teaches:
  • nn.Flatten  — turns (B,1,28,28) into (B,784) in one layer
  • nn.Sequential — composing layers without writing a custom forward()
  • BatchNorm1d  — normalises across the batch dimension for 1-D features
  • Dropout      — randomly zeros activations during training (not eval)
"""

import torch.nn as nn
from .base import BaseModel


class MLP(BaseModel):
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: list[int] = (256, 128),
        num_classes: int = 10,
        dropout: float = 0.2,
    ):
        super().__init__()

        layers = [nn.Flatten()]  # (B,1,28,28) → (B,784)
        in_dim = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))

        self.net = nn.Sequential(*layers)
        self.init_weights()

    def forward(self, x):
        return self.net(x)
