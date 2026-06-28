"""
Decoder Block — One Layer of a GPT-Style Transformer

A single "decoder block" stacks:
  1. Multi-head causal self-attention  (with pre-norm + residual)
  2. Position-wise feed-forward network (with pre-norm + residual)

"Decoder" here means decoder-only (no cross-attention to an encoder).
GPT, LLaMA, Mistral, Falcon are all decoder-only transformers.

ASCII diagram of one block:

    x (B, T, d_model)
    │
    ├──→ LayerNorm → MultiHeadAttention → (+) ──→ x'
    │                                     ↑
    └─────────────────────────────────────┘  (residual)
    │
    ├──→ LayerNorm → FeedForward ──────→ (+) ──→ output
    │                                    ↑
    └────────────────────────────────────┘  (residual)

The block can use either:
  - Standard FFN + standard LayerNorm     (GPT-2 style)
  - SwiGLU FFN  + RMSNorm                 (LLaMA style)
Controlled by the GPTConfig.
"""

import torch
import torch.nn as nn
from ..components.multihead import MultiHeadAttention
from ..components.feedforward import FeedForward, SwiGLUFeedForward
from ..modern.rmsnorm import RMSNorm
from ..modern.rope import apply_rope


class DecoderBlock(nn.Module):
    def __init__(
        self,
        d_model:    int,
        n_heads:    int,
        d_ff:       int   = None,
        dropout:    float = 0.1,
        use_flash:  bool  = True,
        use_swiglu: bool  = False,  # SwiGLU FFN (LLaMA style)
        use_rmsnorm:bool  = False,  # RMSNorm instead of LayerNorm
        use_rope:   bool  = False,  # Rotary PE (applied inside attention)
        bias:       bool  = False,
    ):
        super().__init__()
        self.use_rope = use_rope

        # ── Normalisation ──────────────────────────────────────────────────────
        norm_cls = RMSNorm if use_rmsnorm else nn.LayerNorm
        self.norm1 = norm_cls(d_model)
        self.norm2 = norm_cls(d_model)

        # ── Self-attention ─────────────────────────────────────────────────────
        self.attn = MultiHeadAttention(
            d_model   = d_model,
            n_heads   = n_heads,
            dropout   = dropout,
            use_flash = use_flash,
            bias      = bias,
        )

        # ── Feed-forward ───────────────────────────────────────────────────────
        ff_cls = SwiGLUFeedForward if use_swiglu else FeedForward
        self.ff = ff_cls(d_model=d_model, d_ff=d_ff, dropout=dropout, bias=bias)

        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x:         torch.Tensor,       # (B, T, d_model)
        mask:      torch.Tensor = None,
        kv_cache:  dict = None,
        layer_idx: int  = 0,
        freqs:     tuple = None,       # (cos, sin) for RoPE
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # ── Attention sublayer (pre-norm) ──────────────────────────────────────
        h = self.norm1(x)

        if self.use_rope and freqs is not None:
            # RoPE is applied to Q and K inside attention
            # We pass freqs through to the attention layer
            h, weights = self._attn_with_rope(h, mask, kv_cache, layer_idx, freqs)
        else:
            h, weights = self.attn(h, mask=mask, kv_cache=kv_cache, layer_idx=layer_idx)

        x = x + self.drop(h)

        # ── FFN sublayer (pre-norm) ────────────────────────────────────────────
        x = x + self.drop(self.ff(self.norm2(x)))

        return x, weights

    def _attn_with_rope(self, h, mask, kv_cache, layer_idx, freqs):
        """Apply RoPE to Q and K before attention."""
        B, T, _ = h.shape

        # Project to Q, K, V
        qkv = self.attn.qkv_proj(h)
        Q, K, V = qkv.split(self.attn.d_model, dim=-1)

        # Reshape to heads
        def split_heads(t):
            return t.view(B, -1, self.attn.n_heads, self.attn.d_k).transpose(1, 2)

        Q = split_heads(Q)
        K = split_heads(K)
        V = split_heads(V)

        # Apply RoPE to Q and K
        cos, sin = freqs
        Q = apply_rope(Q, cos, sin)
        K = apply_rope(K, cos, sin)

        # KV cache
        if kv_cache is not None:
            key = f"layer_{layer_idx}"
            if key in kv_cache:
                K = torch.cat([kv_cache[key]["k"], K], dim=2)
                V = torch.cat([kv_cache[key]["v"], V], dim=2)
            kv_cache[key] = {"k": K.detach(), "v": V.detach()}

        from ..components.attention import scaled_dot_product_attention, flash_attention
        if self.attn.use_flash and kv_cache is None:
            out = flash_attention(Q, K, V, causal=True,
                                  dropout=self.attn.dropout, training=self.training)
            weights = None
        else:
            out, weights = scaled_dot_product_attention(
                Q, K, V, mask=mask,
                dropout=self.attn.dropout, training=self.training,
            )

        out = out.transpose(1, 2).contiguous().view(B, -1, self.attn.d_model)
        out = self.attn.out_proj(out)
        return out, weights
