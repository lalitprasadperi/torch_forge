"""
RMSNorm — Root Mean Square Layer Normalisation

Paper: "Root Mean Square Layer Normalization" (Zhang & Sennrich, 2019)
Used in: LLaMA, Mistral, Falcon, Gemma, PaLM 2

WHAT'S DIFFERENT FROM LAYERNORM?
──────────────────────────────────
LayerNorm:   normalise by (x - mean) / sqrt(var + eps)   — centres AND scales
RMSNorm:     normalise by x / RMS(x)                     — scales only

RMS(x) = sqrt( mean(x²) ) = sqrt( (1/d) Σ xᵢ² )

WHY REMOVE THE MEAN?
  • 10-30% faster: no mean subtraction, no mean gradient
  • Hypothesis: re-centring is not necessary; re-scaling is what matters
  • Empirically matches or beats LayerNorm on language tasks
  • LLaMA-1/2/3, Mistral, and Qwen all use RMSNorm exclusively

FORMULA:
  RMSNorm(x) = x / RMS(x) × γ
  where γ is a learnable scale vector (no β shift term)

IMPLEMENTATION NOTE:
  We upcast to float32 before computing RMS for numerical stability,
  then cast back to the input dtype. This is critical for BF16 training
  where RMS computed in BF16 can be significantly inaccurate.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps   = eps
        self.gamma = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()                                       # upcast for stability
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x * rms).to(dtype) * self.gamma

    def extra_repr(self):
        return f"d_model={self.gamma.shape[0]}, eps={self.eps}"
