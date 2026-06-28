"""
Residual Connections — Why Transformers Can Be Deep

WHY RESIDUALS?
──────────────
Without residual connections, a 96-layer transformer would suffer from
vanishing gradients — the gradient signal decays exponentially as it
propagates back through 96 matrix multiplications.

The residual (skip) connection provides a gradient highway:
  output = x + sublayer(x)
  ∂L/∂x = ∂L/∂output × (1 + ∂sublayer/∂x)

The "1 +" means gradients flow directly from the loss to early layers
without passing through the sublayer's Jacobian. Early layers can always
learn, regardless of how deep the network is.

ORIGINAL (POST-NORM):
  x = LayerNorm(x + sublayer(x))
  As used in the original "Attention is All You Need" paper.

MODERN (PRE-NORM):
  x = x + sublayer(LayerNorm(x))
  As used in GPT-2, GPT-3, LLaMA. More stable — LayerNorm is applied
  BEFORE the sublayer, so the sublayer always receives normalised input.
  Pre-norm allows training without learning rate warmup in many cases.

PreNormResidual below implements the pre-norm variant.
"""

import torch
import torch.nn as nn


class PreNormResidual(nn.Module):
    """
    Pre-norm residual wrapper:
      x = x + sublayer(norm(x))

    Usage:
      self.attn_layer = PreNormResidual(d_model, MultiHeadAttention(...))
      self.ff_layer   = PreNormResidual(d_model, FeedForward(...))

    The norm is a separate parameter object per layer — each layer learns
    its own scale (gamma) and shift (beta).
    """

    def __init__(self, d_model: int, sublayer: nn.Module):
        super().__init__()
        self.norm     = nn.LayerNorm(d_model)
        self.sublayer = sublayer

    def forward(self, x: torch.Tensor, **kwargs) -> tuple[torch.Tensor, ...]:
        # Pre-norm: normalise BEFORE passing to sublayer
        out = self.sublayer(self.norm(x), **kwargs)

        # sublayer might return (output, weights) tuple (attention does)
        if isinstance(out, tuple):
            return x + out[0], *out[1:]
        return x + out
