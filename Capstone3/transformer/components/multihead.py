"""
Multi-Head Attention (MHA)

WHY MULTIPLE HEADS?
────────────────────
A single attention head computes one weighted average of the values.
But a sentence has many different types of relationships simultaneously:
  - syntactic dependency (subject → verb)
  - coreference      (pronoun → noun it refers to)
  - semantic context (word → surrounding topic words)

Multiple heads allow the model to attend to different aspects in parallel.
Each head learns its own Q/K/V projection → its own attention pattern.

HOW IT WORKS:
  d_model = 512,  n_heads = 8  →  d_k = d_v = 512/8 = 64

  Input x: (B, T, d_model)
    ↓
  W_Q, W_K, W_V: (d_model, d_model) — project to Q, K, V
    ↓
  Split Q, K, V into n_heads chunks of d_k each
    ↓
  Run scaled_dot_product_attention on each head in parallel
    ↓
  Concatenate head outputs: (B, T, n_heads × d_v) = (B, T, d_model)
    ↓
  W_O: (d_model, d_model) — output projection

EFFICIENT IMPLEMENTATION:
  Instead of n_heads separate projections, do ONE big projection
  then reshape:
    Q = xW_Q  → shape (B, T, d_model) → reshape to (B, n_heads, T, d_k)
  This is how it's done in practice (single large matmul > many small ones).

KV CACHE (for inference):
  During autoregressive generation we generate one token at a time.
  For each new token, we'd recompute K and V for all previous tokens — wasteful.
  Instead, cache K and V for each layer and append the new token's K,V:
    step t=0: K = [K_0],        V = [V_0]
    step t=1: K = [K_0, K_1],   V = [V_0, V_1]
    step t=2: K = [K_0, K_1, K_2], ...
  The new query attends to ALL cached keys. O(1) extra compute per step.
"""

import torch
import torch.nn as nn
from .attention import scaled_dot_product_attention, flash_attention


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model:    int,
        n_heads:    int,
        dropout:    float = 0.1,
        use_flash:  bool  = True,   # use torch SDPA (FlashAttn backend) when possible
        bias:       bool  = False,  # GPT-style: no bias in Q,K,V,O projections
    ):
        super().__init__()
        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.d_model   = d_model
        self.n_heads   = n_heads
        self.d_k       = d_model // n_heads   # dim per head
        self.dropout   = dropout
        self.use_flash = use_flash

        # Single fused projection: computes Q, K, V in one matmul
        # Output dim = 3 × d_model: first d_model = Q, next = K, last = V
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        # Initialise output projection with smaller std (GPT-2 trick):
        # with n residual layers, each adds variance → scale by 1/sqrt(2*n_layers)
        # Done externally by the model after building.

    def forward(
        self,
        x:          torch.Tensor,        # (B, T, d_model)
        mask:       torch.Tensor = None,
        kv_cache:   dict = None,         # for inference: {'k': tensor, 'v': tensor}
        layer_idx:  int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T, _ = x.shape

        # ── Project to Q, K, V ────────────────────────────────────────────────
        qkv = self.qkv_proj(x)                     # (B, T, 3 × d_model)
        Q, K, V = qkv.split(self.d_model, dim=-1)  # each: (B, T, d_model)

        # ── Reshape into heads ────────────────────────────────────────────────
        # (B, T, d_model) → (B, T, n_heads, d_k) → (B, n_heads, T, d_k)
        def split_heads(t):
            return t.view(B, -1, self.n_heads, self.d_k).transpose(1, 2)

        Q = split_heads(Q)   # (B, n_heads, T, d_k)
        K = split_heads(K)
        V = split_heads(V)

        # ── KV Cache (inference only) ─────────────────────────────────────────
        weights = None
        if kv_cache is not None:
            key = f"layer_{layer_idx}"
            if key in kv_cache:
                K = torch.cat([kv_cache[key]["k"], K], dim=2)
                V = torch.cat([kv_cache[key]["v"], V], dim=2)
            kv_cache[key] = {"k": K.detach(), "v": V.detach()}

        # ── Attention ─────────────────────────────────────────────────────────
        if self.use_flash and kv_cache is None:
            # torch SDPA dispatches to FlashAttention when on CUDA
            # Does NOT return weights (incompatible with tiling approach)
            out = flash_attention(Q, K, V, causal=True,
                                  dropout=self.dropout, training=self.training)
        else:
            # Our manual implementation: returns weights for visualisation
            out, weights = scaled_dot_product_attention(
                Q, K, V, mask=mask,
                dropout=self.dropout, training=self.training,
            )

        # ── Merge heads ───────────────────────────────────────────────────────
        # (B, n_heads, T, d_k) → (B, T, n_heads, d_k) → (B, T, d_model)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.d_model)

        # ── Output projection ─────────────────────────────────────────────────
        out = self.out_proj(out)   # (B, T, d_model)

        return out, weights
