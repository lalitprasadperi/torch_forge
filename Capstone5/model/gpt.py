"""
PagedGPT — Transformer with Paged KV Cache Support

Standard GPT architecture modified to:
  1. Accept pre-allocated KV cache tensors (one per layer)
  2. Write K,V to the cache during prefill and decode
  3. Read K,V from the cache using block tables during decode

PREFILL MODE:
  Input:  token_ids (B, T)           — the prompt
  KV ops: standard causal attention, write all T tokens to cache blocks
  Output: logits (B, T, vocab)       — we only use logits[-1] for sampling

DECODE MODE:
  Input:  token_ids (B, 1)           — one new token per sequence
  KV ops: paged gather + attend over all cached positions
  Output: logits (B, 1, vocab)       — one prediction per sequence

The model does NOT track which mode it's in — the caller (ModelRunner)
sets `is_prefill` and passes the appropriate block_tables + seq_lens.
"""

import math
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import ModelConfig
from model.paged_attention import (
    paged_attention_decode,
    paged_attention_prefill,
    write_to_cache,
)


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps    = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        normed = x32 * x32.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (normed * self.weight.float()).to(x.dtype)


class PagedAttention(nn.Module):
    """
    Multi-head attention with paged KV cache read/write.

    During PREFILL: computes full causal attention, writes K,V to blocks.
    During DECODE: reads K,V from blocks, computes single-token attention.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head  = config.d_head
        self.scale   = 1.0 / math.sqrt(self.d_head)

        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj  = nn.Linear(config.d_model, config.d_model,     bias=False)

    def forward(
        self,
        x:            torch.Tensor,          # (B, T, d_model)
        kv_cache:     torch.Tensor,          # (num_blocks, 2, block_size, n_heads, d_head)
        block_tables: torch.Tensor,          # (B, max_blocks) int32
        seq_lens:     torch.Tensor,          # (B,) int32
        is_prefill:   bool,
        cache_offset: torch.Tensor,          # (B,) — how many tokens are already in cache
    ) -> torch.Tensor:
        B, T, C = x.shape
        block_size = kv_cache.shape[2]

        # ── Project to Q, K, V ────────────────────────────────────────────────
        qkv = self.qkv_proj(x)                           # (B, T, 3*d_model)
        q, k, v = qkv.split(self.n_heads * self.d_head, dim=-1)

        # Reshape to (B, T, n_heads, d_head)
        def _reshape(t):
            return t.view(B, T, self.n_heads, self.d_head)

        q, k, v = _reshape(q), _reshape(k), _reshape(v)

        # ── Write K,V to cache ────────────────────────────────────────────────
        write_to_cache(kv_cache, k, v, block_tables, cache_offset, block_size)

        # ── Attention ─────────────────────────────────────────────────────────
        if is_prefill:
            # Full causal attention over the T input tokens
            attn_out = paged_attention_prefill(q, k, v)  # (B, T, H, D)
        else:
            # Single-token attention using cached KV
            assert T == 1, "Decode mode expects one token per sequence"
            q_dec = q[:, 0, :, :]                        # (B, H, D)
            attn_out = paged_attention_decode(
                q_dec, kv_cache, block_tables, seq_lens, self.scale
            )                                             # (B, H, D)
            attn_out = attn_out.unsqueeze(1)              # (B, 1, H, D)

        # ── Merge heads → project ─────────────────────────────────────────────
        attn_out = attn_out.reshape(B, T, self.n_heads * self.d_head)
        return self.out_proj(attn_out)


class FFN(nn.Module):
    """SwiGLU feed-forward network (used in Llama-family models)."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up   = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down = nn.Linear(config.d_ff,    config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.attn  = PagedAttention(config)
        self.norm2 = RMSNorm(config.d_model)
        self.ffn   = FFN(config)

    def forward(
        self,
        x:            torch.Tensor,
        kv_cache:     torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens:     torch.Tensor,
        is_prefill:   bool,
        cache_offset: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), kv_cache, block_tables, seq_lens, is_prefill, cache_offset)
        x = x + self.ffn(self.norm2(x))
        return x


class PagedGPT(nn.Module):
    """
    Full transformer decoder with paged KV cache.

    KV caches are passed in from ModelRunner (pre-allocated once at startup).
    Each layer gets its own cache tensor.

    kv_caches: List[Tensor], one per layer
               each: (num_blocks, 2, block_size, n_heads, d_head)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config   = config
        self.embed    = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Embedding(config.context_len, config.d_model)
        self.blocks   = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm     = RMSNorm(config.d_model)
        self.lm_head  = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: embedding and lm_head share the same matrix
        self.lm_head.weight = self.embed.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        token_ids:    torch.Tensor,          # (B, T)
        kv_caches:    List[torch.Tensor],    # one per layer
        block_tables: torch.Tensor,          # (B, max_blocks) int32
        seq_lens:     torch.Tensor,          # (B,) int32 — full length after this step
        is_prefill:   bool,
        cache_offset: Optional[torch.Tensor] = None,  # (B,) — tokens already in cache
    ) -> torch.Tensor:
        """Returns logits (B, T, vocab_size)."""
        B, T = token_ids.shape
        device = token_ids.device

        if cache_offset is None:
            # Prefill: nothing was in cache yet
            cache_offset = torch.zeros(B, dtype=torch.int32, device=device)

        # Position IDs: for decode, position = seq_len - 1 (the current token)
        if is_prefill:
            positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)  # (B, T)
        else:
            # cache_offset = seq_len - 1 (number of tokens before this one)
            positions = cache_offset.unsqueeze(1)  # (B, 1)

        x = self.embed(token_ids) + self.pos_embed(positions)

        for i, block in enumerate(self.blocks):
            x = block(x, kv_caches[i], block_tables, seq_lens, is_prefill, cache_offset)

        x = self.norm(x)
        return self.lm_head(x)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @classmethod
    def from_config(cls, config: ModelConfig) -> "PagedGPT":
        return cls(config)

    def allocate_kv_caches(
        self,
        num_blocks:  int,
        block_size:  int,
        device:      torch.device,
        dtype:       torch.dtype = torch.float16,
    ) -> List[torch.Tensor]:
        """
        Allocate the KV cache tensors for all layers.
        Called once at server startup.

        Shape per layer: (num_blocks, 2, block_size, n_heads, d_head)
        """
        return [
            torch.zeros(
                num_blocks, 2, block_size, self.config.n_heads, self.config.d_head,
                dtype=dtype, device=device
            )
            for _ in range(self.config.n_layers)
        ]

    def kv_cache_bytes(self, num_blocks: int, block_size: int, dtype: torch.dtype = torch.float16) -> int:
        """Total bytes for all KV cache tensors."""
        bytes_per_element = torch.finfo(dtype).bits // 8
        per_layer = num_blocks * 2 * block_size * self.config.n_heads * self.config.d_head
        return per_layer * self.config.n_layers * bytes_per_element


# ── Quick smoke test ───────────────────────────────────────────────────────────

def _smoke_test():
    import sys
    sys.path.insert(0, "/home/jmd/Desktop/TrainHard/LearnTorch/Capstone5")

    config     = ModelConfig.nano()
    model      = PagedGPT(config).cuda().half()
    num_blocks = 64
    block_size = 16
    B, T       = 2, 32

    kv_caches    = model.allocate_kv_caches(num_blocks, block_size, device=torch.device("cuda"))
    block_tables = torch.zeros(B, num_blocks, dtype=torch.int32, device="cuda")
    # assign blocks: seq 0 gets blocks 0..1, seq 1 gets blocks 2..3
    block_tables[0, :2] = torch.tensor([0, 1])
    block_tables[1, :2] = torch.tensor([2, 3])

    token_ids = torch.randint(0, config.vocab_size, (B, T), device="cuda")
    seq_lens  = torch.tensor([T, T], dtype=torch.int32, device="cuda")

    # Prefill
    with torch.no_grad():
        logits = model(token_ids, kv_caches, block_tables, seq_lens, is_prefill=True)
    print(f"  Prefill logits: {logits.shape}")   # (B, T, vocab)

    # Decode (one step)
    next_tok    = logits[:, -1:, :].argmax(dim=-1)         # (B, 1)
    cache_off   = seq_lens.clone()                          # (B,) — already T tokens in cache
    seq_lens_d  = seq_lens + 1

    # Need one more block per seq if block boundary crossed
    for i in range(B):
        if T % block_size == 0:
            block_tables[i, T // block_size] = 4 + i  # assign next block

    with torch.no_grad():
        logits_d = model(next_tok, kv_caches, block_tables, seq_lens_d,
                         is_prefill=False, cache_offset=cache_off)
    print(f"  Decode  logits: {logits_d.shape}")  # (B, 1, vocab)
    print(f"  Params: {model.num_parameters():,}")
    print("  ✓ Smoke test passed")


if __name__ == "__main__":
    _smoke_test()
