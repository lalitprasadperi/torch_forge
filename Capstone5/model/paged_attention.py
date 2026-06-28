"""
Paged Attention — KV cache write + paged decode attention.

Two modes:
  PREFILL: standard causal attention over N tokens, writes K,V to cache blocks.
  DECODE:  single-token attention over all cached K,V, gathered from block table.

The attention KERNEL for decode is the performance-critical part.
This file provides a correct PyTorch reference implementation.
See kernels/paged_attn_triton.py for the optimised Triton version.

BLOCK TABLE LAYOUT:
  block_table: (batch, max_blocks_per_seq) — int32 tensor
  kv_cache:    (num_blocks, 2, block_size, n_heads, d_head) — fp16 tensor
                dim 1: 0=K  1=V

  To get K for sequence i:
    blocks = block_table[i, :ceil(seq_len/block_size)]
    k_flat = kv_cache[blocks, 0].reshape(-1, n_heads, d_head)  # (n_blocks*block_size, H, D)
    k      = k_flat[:seq_len]                                   # (seq_len, H, D)
"""

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def write_to_cache(
    kv_cache:    torch.Tensor,   # (num_blocks, 2, block_size, n_heads, d_head)
    key:         torch.Tensor,   # (batch, seq_len, n_heads, d_head) or (batch, n_heads, d_head) for decode
    value:       torch.Tensor,   # same shape
    block_tables: torch.Tensor,  # (batch, max_blocks) int32
    seq_lens:    torch.Tensor,   # (batch,) int32 — number of tokens currently in cache per seq
    block_size:  int,
) -> None:
    """
    Write K,V tensors into the allocated cache blocks.

    For prefill: key has shape (batch, seq_len, n_heads, d_head).
                 We write all seq_len tokens starting at position seq_lens[i]
                 (which is 0 on first prefill, >0 if chunked prefill).

    For decode: key has shape (batch, n_heads, d_head) — one token per sequence.
                We write to the single next slot.

    This is the SCATTER step of paged attention.
    """
    if key.ndim == 3:
        # Decode: (batch, n_heads, d_head) → add seq_len dim
        key   = key.unsqueeze(1)    # (batch, 1, n_heads, d_head)
        value = value.unsqueeze(1)

    B, T, H, D = key.shape

    for i in range(B):
        offset = seq_lens[i].item()   # where in this sequence we start writing
        for t in range(T):
            pos      = offset + t
            block_id = block_tables[i, pos // block_size].item()
            slot     = pos % block_size
            kv_cache[block_id, 0, slot] = key[i, t]    # K
            kv_cache[block_id, 1, slot] = value[i, t]  # V


def paged_attention_prefill(
    query: torch.Tensor,   # (batch, T, n_heads, d_head)
    key:   torch.Tensor,   # (batch, T, n_heads, d_head)
    value: torch.Tensor,   # (batch, T, n_heads, d_head)
) -> torch.Tensor:
    """
    Standard causal self-attention for prefill.
    Inputs are the CURRENT batch — no page table needed here.
    We use PyTorch's SDPA (FlashAttention when available).
    """
    # Rearrange to (batch, n_heads, T, d_head) for SDPA
    q = query.permute(0, 2, 1, 3)   # (B, H, T, D)
    k = key.permute(0, 2, 1, 3)
    v = value.permute(0, 2, 1, 3)

    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return out.permute(0, 2, 1, 3)   # (B, T, H, D)


def paged_attention_decode(
    query:        torch.Tensor,   # (batch, n_heads, d_head)
    kv_cache:     torch.Tensor,   # (num_blocks, 2, block_size, n_heads, d_head)
    block_tables: torch.Tensor,   # (batch, max_blocks) int32
    seq_lens:     torch.Tensor,   # (batch,) int32 — total length INCLUDING new token
    scale:        float,
) -> torch.Tensor:                # (batch, n_heads, d_head)
    """
    Paged attention for decode phase.

    For each sequence i:
      1. Look up block_table[i] to find which physical blocks hold its KV
      2. Gather K and V from those blocks
      3. Compute standard dot-product attention: q[i] attends over all cached K,V[i]

    This is the GATHER + ATTENTION step of paged attention.

    In production (vLLM), this is replaced by a fused Triton kernel that
    avoids materialising the gathered K,V tensors.
    See: kernels/paged_attn_triton.py
    """
    B, H, D = query.shape
    block_size = kv_cache.shape[2]
    output = torch.zeros_like(query)

    for i in range(B):
        seq_len = seq_lens[i].item()
        n_blocks = math.ceil(seq_len / block_size)

        # ── Gather K and V for sequence i ──────────────────────────────────────
        blocks = block_tables[i, :n_blocks]              # (n_blocks,)
        kv     = kv_cache[blocks]                        # (n_blocks, 2, block_size, H, D)
        k_flat = kv[:, 0].reshape(-1, H, D)             # (n_blocks*block_size, H, D)
        v_flat = kv[:, 1].reshape(-1, H, D)

        k = k_flat[:seq_len].to(query.dtype)             # (seq_len, H, D)
        v = v_flat[:seq_len].to(query.dtype)

        # ── Attention: q[i] attends over all seq_len positions ─────────────────
        # q: (H, D)  k,v: (seq_len, H, D)
        k_h = k.permute(1, 0, 2)                        # (H, seq_len, D)
        v_h = v.permute(1, 0, 2)                        # (H, seq_len, D)
        q_h = query[i].unsqueeze(1)                     # (H, 1, D)

        scores  = torch.bmm(q_h, k_h.transpose(1, 2)) * scale   # (H, 1, seq_len)
        weights = torch.softmax(scores, dim=-1)
        attn    = torch.bmm(weights, v_h).squeeze(1)            # (H, D)

        output[i] = attn

    return output
