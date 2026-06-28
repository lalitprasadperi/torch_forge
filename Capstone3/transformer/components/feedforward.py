"""
Feed-Forward Network (FFN) — The "Memory" of the Transformer

WHY IS THERE AN FFN?
─────────────────────
Attention mixes information across positions: it decides WHICH tokens
to combine. But it doesn't transform the token representations themselves.
The FFN is applied to each position independently, transforming what
information is stored at each token.

Researchers have found that FFN layers act like associative memories:
they store (key, value) pairs in their weight matrices and can retrieve
facts based on the input activation. This is where the model "remembers"
things like "Paris is the capital of France".

ORIGINAL FFN (Vaswani 2017):
  FFN(x) = W₂ · max(0, W₁x + b₁) + b₂
           Linear → ReLU → Linear
  d_ff = 4 × d_model  (typical expansion ratio)

GELU VARIANT (GPT-2, BERT):
  FFN(x) = W₂ · GELU(W₁x + b₁) + b₂
  GELU is smoother than ReLU: GELU(x) ≈ x · σ(1.702x)
  Empirically better than ReLU for language models.

SWIGLU VARIANT (LLaMA, PaLM):
  SwiGLU(x) = (W₁x ⊙ SiLU(W₃x)) W₂
  Three weight matrices. SiLU(x) = x · σ(x) (same as Swish).
  The gate (W₃x) learns when to pass information from W₁x.
  Outperforms GELU on many tasks; d_ff = 2/3 × 4 × d_model for iso-param.
  See SwiGLUFeedForward below and transformer/modern/swiglu.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """
    Standard transformer FFN with GELU activation.
    Applied independently to each token position.
    """

    def __init__(
        self,
        d_model:  int,
        d_ff:     int   = None,    # default: 4 × d_model
        dropout:  float = 0.1,
        bias:     bool  = False,
        activation: str = "gelu",  # "relu" | "gelu"
    ):
        super().__init__()
        d_ff = d_ff or 4 * d_model

        self.w1     = nn.Linear(d_model, d_ff, bias=bias)
        self.w2     = nn.Linear(d_ff, d_model, bias=bias)
        self.drop   = nn.Dropout(dropout)

        act_map = {"relu": F.relu, "gelu": F.gelu}
        if activation not in act_map:
            raise ValueError(f"Unknown activation: {activation!r}")
        self.act = act_map[activation]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        x = self.w1(x)      # (B, T, d_ff)
        x = self.act(x)
        x = self.drop(x)
        x = self.w2(x)      # (B, T, d_model)
        return x


class SwiGLUFeedForward(nn.Module):
    """
    SwiGLU FFN — used in LLaMA, Mistral, PaLM.

    SwiGLU(x) = (xW₁ ⊙ SiLU(xW₃)) W₂

    ⊙ = element-wise multiplication (the "gate")
    SiLU(x) = x · σ(x)  (also called Swish)

    Compared to standard FFN:
      • One extra weight matrix (W₃ for the gate)
      • 2/3 × standard d_ff to keep parameter count equal
      • Better perplexity on language modelling tasks
    """

    def __init__(
        self,
        d_model:  int,
        d_ff:     int   = None,   # default: 8/3 × d_model (≈ 2/3 × 4× d_model)
        dropout:  float = 0.0,
        bias:     bool  = False,
    ):
        super().__init__()
        # 8/3 × d_model keeps param count ≈ same as standard 4× d_model FFN
        # (two matrices at d_model×d_ff vs three at d_model×2/3×d_ff)
        d_ff = d_ff or int(8 * d_model / 3)
        # Round to nearest multiple of 256 for hardware efficiency
        d_ff = (d_ff + 255) // 256 * 256

        self.w1   = nn.Linear(d_model, d_ff, bias=bias)   # value path
        self.w3   = nn.Linear(d_model, d_ff, bias=bias)   # gate path
        self.w2   = nn.Linear(d_ff, d_model, bias=bias)   # output projection
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gated linear unit: multiply activated gate by value
        gate  = F.silu(self.w3(x))   # SiLU gate: (B, T, d_ff)
        value = self.w1(x)            # value:      (B, T, d_ff)
        x = self.drop(gate * value)   # element-wise gate
        return self.w2(x)             # project back: (B, T, d_model)
