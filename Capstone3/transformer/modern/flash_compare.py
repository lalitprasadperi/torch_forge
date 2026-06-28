"""
FlashAttention vs Naive Attention — Memory and Speed Comparison

WHAT IS FLASHATTENTION?
────────────────────────
Paper: "FlashAttention: Fast and Memory-Efficient Exact Attention
        with IO-Awareness" (Dao et al., 2022)
Used in: every serious production LLM since 2022

Standard attention materialises the full (T, T) attention matrix in GPU HBM
(high-bandwidth memory). For T=8192, that's 8192² × 2 bytes = 128 MB just
for the attention scores — per layer, per head.

FlashAttention avoids this by computing attention in TILES that fit in SRAM
(the fast on-chip cache), never writing the full matrix to HBM:

MEMORY COMPARISON:
  Naive:  O(T²)   — stores full attention matrix in HBM
  Flash:  O(T)    — only stores output + softmax normaliser in HBM

For T=32768 (32k context):
  Naive:  32768² × 2B × 32 heads = 64 GB  per layer (!)
  Flash:  ~4 MB per layer

SPEED:
  Naive is "compute-bound" on paper (O(T² × d_k) FLOPs) but actually
  "memory-bound" in practice — the bottleneck is reading/writing T² values.
  FlashAttention eliminates most HBM traffic → 2-4x wall-clock speedup.

TILING ALGORITHM (simplified):
  Divide Q into blocks of size B_r, K/V into blocks of size B_c.
  For each Q block:
    For each K/V block:
      Compute local attention scores S_ij = Q_i K_j^T / sqrt(d_k)
      Update running max (for numerically stable softmax)
      Accumulate output O_i += softmax_piece × V_j
  Only O (the output) is written to HBM. S never materialises fully.

THIS FILE:
  Compares our manual implementation vs torch.nn.functional.scaled_dot_product_attention
  which dispatches to FlashAttention on CUDA (PyTorch >= 2.0).
"""

import math
import time
import torch
import torch.nn.functional as F


def naive_attention(Q, K, V, causal=True):
    """Our manual O(T²) memory attention."""
    d_k    = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if causal:
        T = Q.size(-2)
        mask = torch.tril(torch.ones(T, T, device=Q.device, dtype=torch.bool))
        scores = scores.masked_fill(~mask, -1e9)
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, V)


def flash_attention_torch(Q, K, V, causal=True):
    """PyTorch built-in SDPA → dispatches to FlashAttention on CUDA."""
    return F.scaled_dot_product_attention(Q, K, V, is_causal=causal)


def compare_attention_implementations(
    seq_len:  int = 1024,
    d_model:  int = 512,
    n_heads:  int = 8,
    n_repeat: int = 10,
    device:   str = None,
) -> dict:
    """
    Benchmark naive vs Flash attention. Returns timing and memory stats.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    d_k = d_model // n_heads
    B   = 1

    Q = torch.randn(B, n_heads, seq_len, d_k, device=device, dtype=torch.float16
                    if device == "cuda" else torch.float32)
    K = torch.randn_like(Q)
    V = torch.randn_like(Q)

    results = {}

    # ── Naive attention ────────────────────────────────────────────────────────
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_repeat):
        with torch.no_grad():
            out_naive = naive_attention(Q.float() if device == "cuda" else Q, K.float() if device == "cuda" else K, V.float() if device == "cuda" else V)
    if device == "cuda":
        torch.cuda.synchronize()
    t_naive = (time.perf_counter() - t0) / n_repeat * 1000

    naive_mem = 0
    if device == "cuda":
        naive_mem = torch.cuda.max_memory_allocated() / 1024**2

    results["naive"] = {"time_ms": t_naive, "peak_mem_mb": naive_mem}

    # ── Flash attention ────────────────────────────────────────────────────────
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_repeat):
        with torch.no_grad():
            out_flash = flash_attention_torch(Q, K, V)
    if device == "cuda":
        torch.cuda.synchronize()
    t_flash = (time.perf_counter() - t0) / n_repeat * 1000

    flash_mem = 0
    if device == "cuda":
        flash_mem = torch.cuda.max_memory_allocated() / 1024**2

    results["flash"] = {"time_ms": t_flash, "peak_mem_mb": flash_mem}

    # ── Theoretical T² memory ─────────────────────────────────────────────────
    bytes_per_el   = 4  # float32
    attn_mat_mb    = (B * n_heads * seq_len * seq_len * bytes_per_el) / 1024**2
    results["theoretical_attn_matrix_mb"] = attn_mat_mb
    results["seq_len"]   = seq_len
    results["device"]    = device

    return results
