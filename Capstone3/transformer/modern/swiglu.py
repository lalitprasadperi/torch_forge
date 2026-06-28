"""
SwiGLU — Swish-Gated Linear Unit

Paper: "GLU Variants Improve Transformer" (Noam Shazeer, 2020)
Used in: LLaMA 1/2/3, PaLM, Mistral, Gemma, Qwen

BACKGROUND: Gated Linear Units (GLU)
──────────────────────────────────────
Original GLU (Dauphin et al. 2017):
  GLU(x) = (xW + b) ⊗ σ(xV + c)
  where σ is sigmoid. The second half acts as a gate [0,1].

SwiGLU uses Swish instead of sigmoid:
  Swish(x) = x · σ(x) = SiLU(x)     (in PyTorch: F.silu)

  SwiGLU(x, W, V, W₂) = (xW ⊗ SiLU(xV)) W₂

WHY DOES IT WORK BETTER?
  • Gating: the model learns WHICH information to pass (gate) and WHERE (value)
  • SiLU is smooth and non-monotonic: small negative values are preserved
    (unlike ReLU which zeros them). This gives richer gradient signal.
  • Empirically: SwiGLU outperforms GELU and ReLU on perplexity at matched
    parameter counts (Shazeer 2020 showed consistent improvements)

PARAMETER COUNT:
  Standard FFN:  2 × d_model × d_ff  (two matrices)
  SwiGLU FFN:    3 × d_model × d_ff  (three matrices: W, V, W₂)
  To keep params equal:  d_ff_swiglu = 2/3 × d_ff_standard
  LLaMA rounds this to nearest multiple of 256 for hardware efficiency.

NOTE: SwiGLUFeedForward in feedforward.py is the full module.
This file contains the isolated activation function for demonstration.
"""

import torch
import torch.nn.functional as F


def swiglu(gate_input: torch.Tensor, value_input: torch.Tensor) -> torch.Tensor:
    """
    Core SwiGLU computation.

    gate_input:  xW₃  (the gate path)
    value_input: xW₁  (the value path)
    returns:     SiLU(gate_input) ⊗ value_input
    """
    return F.silu(gate_input) * value_input


class SwiGLU(torch.nn.Module):
    """
    Standalone SwiGLU activation for educational use.
    Matches SwiGLUFeedForward but exposes intermediate tensors.
    """

    def __init__(self, d_model: int, d_ff: int = None):
        super().__init__()
        import torch.nn as nn
        d_ff = d_ff or int(8 * d_model / 3)
        d_ff = (d_ff + 255) // 256 * 256

        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

        self.last_gate:  torch.Tensor = None
        self.last_value: torch.Tensor = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.last_gate  = F.silu(self.w3(x))    # gate: learned gating signal
        self.last_value = self.w1(x)             # value: information to gate
        hidden = self.last_gate * self.last_value
        return self.w2(hidden)
