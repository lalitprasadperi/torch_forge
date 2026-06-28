"""
Rotary Position Embedding (RoPE)

Paper: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
       (Su et al., 2021)
Used in: LLaMA 1/2/3, Mistral, Falcon, GPT-NeoX, PaLM, Gemma

WHY RoPE?
──────────
Both sinusoidal PE and learned PE encode ABSOLUTE positions. But for language
understanding, RELATIVE positions matter more:

  "The cat that sat on the mat" — what matters is:
    - "that" is 1 position after "cat"  (relative)
    - not that "cat" is at absolute position 3

RoPE encodes position directly into Q and K, so that:
  dot(Q_m, K_n) depends only on the content AND on (m - n), the relative distance.

HOW IT WORKS:
  Instead of adding a position vector to the token, we ROTATE Q and K
  by an angle proportional to their position.

  For 2D vectors: rotate by angle θ_pos:
    [q₀  q₁] → [q₀cos(θ) - q₁sin(θ)]
                [q₀sin(θ) + q₁cos(θ)]

  For d_k dimensional Q and K, treat them as d_k/2 pairs and rotate each pair
  by a different frequency:
    θᵢ = pos / 10000^(2i/d_k)   (same base frequencies as sinusoidal PE)

  KEY PROPERTY: rotation is an isometry (preserves dot product norms), so:
    Q_m · K_n = f(content, m-n)
  The attention score encodes relative position naturally.

ADVANTAGES OVER LEARNED PE:
  • Generalises to sequences longer than training length (with some tricks)
  • No extra parameters (frequencies are computed, not learned)
  • Relative position awareness is baked into attention, not added as input
  • Works better for very long contexts

IMPLEMENTATION:
  We precompute cos and sin tables for positions 0..max_len.
  Then apply rotation to Q and K inside each attention layer.
"""

import torch
import torch.nn as nn
import math


class RotaryEmbedding(nn.Module):
    """
    Precomputes RoPE cos/sin tables. Called once per forward pass.
    """

    def __init__(self, d_head: int, base: int = 10000, max_len: int = 4096):
        super().__init__()
        self.d_head  = d_head
        self.base    = base
        self.max_len = max_len

        # Compute inverse frequencies: θᵢ = 1 / base^(2i/d_head)
        inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute tables for speed
        self._build_cache(max_len, device=torch.device("cpu"))

    def _build_cache(self, max_len: int, device: torch.device):
        t = torch.arange(max_len, device=device).float()
        freqs = torch.outer(t, self.inv_freq.to(device))  # (max_len, d_head/2)
        emb = torch.cat([freqs, freqs], dim=-1)            # (max_len, d_head)
        self.register_buffer("cos_cache", emb.cos(), persistent=False)
        self.register_buffer("sin_cache", emb.sin(), persistent=False)

    def forward(
        self,
        seq_len: int,
        device:  torch.device,
        offset:  int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (cos, sin) slices for positions [offset, offset+seq_len)."""
        if seq_len + offset > self.cos_cache.shape[0]:
            self._build_cache(seq_len + offset, device)

        cos = self.cos_cache[offset : offset + seq_len].to(device)  # (T, d_head)
        sin = self.sin_cache[offset : offset + seq_len].to(device)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate the second half of the last dimension:
    [x₀, x₁, x₂, x₃] → [-x₂, -x₃, x₀, x₁]

    This is the rotation operation in complex number form:
    (a + ib) × (cos θ + i sin θ) = (a cos θ - b sin θ) + i(a sin θ + b cos θ)
    where we split x into real (first half) and imaginary (second half).
    """
    x1 = x[..., : x.shape[-1] // 2]   # first half
    x2 = x[..., x.shape[-1] // 2 :]   # second half
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    x:   torch.Tensor,   # (B, n_heads, T, d_head)
    cos: torch.Tensor,   # (T, d_head)
    sin: torch.Tensor,   # (T, d_head)
) -> torch.Tensor:
    """
    Apply rotary position embedding to query or key tensor.

    x_rotated = x * cos + rotate_half(x) * sin

    This implements: each (x₀, x₁) pair gets rotated by the position angle:
      x₀' = x₀ cos θ - x₁ sin θ
      x₁' = x₀ sin θ + x₁ cos θ
    """
    # Reshape cos/sin for broadcasting: (T, d_head) → (1, 1, T, d_head)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return x * cos + rotate_half(x) * sin
