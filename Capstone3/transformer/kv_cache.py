"""
KV Cache — Efficient Autoregressive Inference

WHY KV CACHE?
──────────────
During training, we process all T tokens in parallel (teacher forcing):
  • One forward pass → logits for all positions simultaneously

During inference (text generation), we generate token-by-token:
  • Step 0: input = [BOS]      → predict token_1
  • Step 1: input = [BOS, t1]  → predict token_2
  • Step 2: input = [BOS, t1, t2] → predict token_3
  ...

Without KV cache, at step n we recompute K and V for ALL n past tokens.
This is O(n) extra compute per step → O(n²) total for n tokens.

WITH KV CACHE:
  At step n, only the NEW token needs to generate its K and V.
  Append to cached K, V → attention over all past keys as before.
  O(1) extra compute per step → O(n) total.

MEMORY TRADE-OFF:
  KV cache stores K and V for every layer, every head, every past token:
  memory = 2 × n_layers × n_heads × d_head × max_new_tokens × 2 bytes (FP16)

  For LLaMA-7B (32 layers, 32 heads, 128 d_head, 2048 new tokens):
    = 2 × 32 × 32 × 128 × 2048 × 2 = 1 GB

  This is why large-context inference is memory-hungry.
  Techniques like grouped-query attention (GQA, MQA) reduce n_heads for K/V
  to decrease KV cache size.

IMPLEMENTATION:
  We use a plain dict: { "layer_i": {"k": tensor, "v": tensor} }
  DecoderBlock.forward() reads and writes this dict.
  KVCache below is a helper that manages creation and clearing.
"""

import torch
from dataclasses import dataclass, field


class KVCache:
    """
    Simple KV cache manager for autoregressive inference.

    Usage:
        cache = KVCache()
        # First token
        logits, _ = model(tokens[:, :1], kv_cache=cache.cache)
        # Each subsequent token
        for i in range(max_new_tokens):
            next_tok = logits[:, -1:].argmax(-1)
            logits, _ = model(next_tok, kv_cache=cache.cache)
        cache.clear()
    """

    def __init__(self):
        self.cache: dict = {}

    def clear(self):
        self.cache.clear()

    def num_tokens_cached(self) -> int:
        """How many past tokens are in the cache."""
        if not self.cache:
            return 0
        k = next(iter(self.cache.values()))["k"]
        return k.shape[2]  # (B, H, T_past, d_k) → T_past

    def memory_bytes(self) -> int:
        """Total bytes used by this cache."""
        total = 0
        for v in self.cache.values():
            total += v["k"].nbytes + v["v"].nbytes
        return total

    def __repr__(self):
        n = self.num_tokens_cached()
        mb = self.memory_bytes() / 1024**2
        return f"KVCache(tokens={n}, layers={len(self.cache)}, {mb:.1f} MB)"
