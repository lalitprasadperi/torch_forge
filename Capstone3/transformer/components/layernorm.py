"""
LayerNorm — Normalise activations across the feature dimension.

WHY NORMALISE?
──────────────
Without normalisation, activations in deep networks grow or shrink
exponentially with depth (covariate shift). This makes training unstable:
gradients explode or vanish, learning rates become very sensitive.

LayerNorm computes mean and variance ACROSS the feature dimension for each
token independently. Unlike BatchNorm (which normalises across the batch),
LayerNorm works on a single sample — it's stable for variable-length
sequences and small batch sizes.

FORMULA:
  y = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta

  gamma: learnable scale  (initialised to 1)
  beta:  learnable shift  (initialised to 0)
  eps:   small constant to prevent division by zero (typically 1e-5)

WHERE IT'S APPLIED (Pre-norm vs Post-norm):
  Original transformer (post-norm):  x = LayerNorm(x + sublayer(x))
  Modern transformers (pre-norm):    x = x + sublayer(LayerNorm(x))

  Pre-norm is more stable and is used in GPT-2, LLaMA, etc.
  Our PreNormResidual in residual.py uses pre-norm.

WHY gamma AND beta?
  After normalisation, the distribution is forced to N(0,1).
  But what if the optimal representation is NOT zero-centred?
  gamma/beta allow the model to "undo" normalisation if needed —
  they let LayerNorm be a no-op when gamma=1, beta=0.
"""

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    Manual LayerNorm implementation — shows the math explicitly.
    Drop-in equivalent to nn.LayerNorm(d_model, eps=eps).
    """

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps   = eps
        # gamma and beta are registered as Parameters so the optimizer updates them
        self.gamma = nn.Parameter(torch.ones(d_model))   # scale
        self.beta  = nn.Parameter(torch.zeros(d_model))  # shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d_model) — last dim is the feature/model dimension
        mean = x.mean(dim=-1, keepdim=True)              # E[x]
        var  = x.var(dim=-1, keepdim=True, unbiased=False)  # Var[x]
        x_norm = (x - mean) / (var + self.eps).sqrt()   # normalise
        return self.gamma * x_norm + self.beta           # scale + shift

    def extra_repr(self):
        return f"d_model={self.gamma.shape[0]}, eps={self.eps}"
