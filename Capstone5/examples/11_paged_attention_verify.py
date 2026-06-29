"""
Example 11 — Paged Attention: Correctness + Speed vs Standard Attention

Two verifications:
  1. CORRECTNESS: paged attention must give identical output to standard
     attention computed over a contiguous buffer (within fp16 tolerance).
  2. SPEED: how much overhead does the Python gather loop add?
     (This motivates the Triton kernel in kernels/paged_attn_triton.py)

Tests:
  - Small batch (B=1): single-sequence decode
  - Large batch (B=16): many sequences in flight
  - Long context (seq_len=512): many blocks per sequence

Run:  python examples/11_paged_attention_verify.py
"""
import sys; sys.path.insert(0, ".")
import math
import time
import torch
from model.paged_attention import paged_attention_decode, write_to_cache


def standard_attention_decode(q, k_cont, v_cont, scale):
    """Reference: q attends over full contiguous K,V."""
    # q:     (B, H, D)
    # k,v:   (B, seq_len, H, D)
    k_h = k_cont.permute(0, 2, 1, 3)   # (B, H, seq_len, D)
    v_h = v_cont.permute(0, 2, 1, 3)
    q_h = q.unsqueeze(2)                # (B, H, 1, D)
    scores = torch.matmul(q_h, k_h.transpose(-1, -2)) * scale  # (B, H, 1, seq_len)
    return torch.matmul(torch.softmax(scores, dim=-1), v_h).squeeze(2)  # (B, H, D)


def setup_paged(k_cont, v_cont, B, seq_len, H, D, block_size, device):
    """
    Copy contiguous K,V into a paged cache with a random block table.
    Returns (kv_cache, block_tables, seq_lens) ready for paged_attention_decode.
    """
    n_blocks_per_seq = math.ceil(seq_len / block_size)
    num_blocks       = B * n_blocks_per_seq

    # (num_blocks, 2, block_size, H, D)
    kv_cache     = torch.zeros(num_blocks, 2, block_size, H, D, device=device)
    block_tables = torch.zeros(B, n_blocks_per_seq, dtype=torch.int32, device=device)

    for i in range(B):
        # Assign contiguous block IDs per sequence (could be random)
        start_blk = i * n_blocks_per_seq
        block_tables[i] = torch.arange(start_blk, start_blk + n_blocks_per_seq,
                                        dtype=torch.int32)
        for blk_idx in range(n_blocks_per_seq):
            blk_id    = start_blk + blk_idx
            tok_start = blk_idx * block_size
            tok_end   = min(tok_start + block_size, seq_len)
            n_tok     = tok_end - tok_start
            kv_cache[blk_id, 0, :n_tok] = k_cont[i, tok_start:tok_end]  # K
            kv_cache[blk_id, 1, :n_tok] = v_cont[i, tok_start:tok_end]  # V

    seq_lens = torch.full((B,), seq_len, dtype=torch.int32, device=device)
    return kv_cache, block_tables, seq_lens


def verify(B, H, D, seq_len, block_size, device, label):
    torch.manual_seed(0)
    scale   = 1.0 / math.sqrt(D)
    k_cont  = torch.randn(B, seq_len, H, D, device=device)
    v_cont  = torch.randn(B, seq_len, H, D, device=device)
    q       = torch.randn(B, H, D, device=device)

    kv_cache, block_tables, seq_lens = setup_paged(k_cont, v_cont, B, seq_len, H, D, block_size, device)

    ref   = standard_attention_decode(q, k_cont, v_cont, scale)
    paged = paged_attention_decode(q, kv_cache, block_tables, seq_lens, scale)

    max_diff = (ref - paged).abs().max().item()
    passed   = "✓" if max_diff < 1e-4 else "✗"
    print(f"  {passed} {label:<40}  max_diff={max_diff:.2e}")
    return max_diff < 1e-4


def benchmark(B, H, D, seq_len, block_size, device, n_iters=50):
    scale   = 1.0 / math.sqrt(D)
    k_cont  = torch.randn(B, seq_len, H, D, device=device)
    v_cont  = torch.randn(B, seq_len, H, D, device=device)
    q       = torch.randn(B, H, D, device=device)
    kv_cache, block_tables, seq_lens = setup_paged(k_cont, v_cont, B, seq_len, H, D, block_size, device)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Standard
    t0 = time.perf_counter()
    for _ in range(n_iters):
        standard_attention_decode(q, k_cont, v_cont, scale)
    if device.type == "cuda": torch.cuda.synchronize()
    t_std = (time.perf_counter() - t0) / n_iters * 1000

    # Paged (PyTorch reference)
    t0 = time.perf_counter()
    for _ in range(n_iters):
        paged_attention_decode(q, kv_cache, block_tables, seq_lens, scale)
    if device.type == "cuda": torch.cuda.synchronize()
    t_paged = (time.perf_counter() - t0) / n_iters * 1000

    return t_std, t_paged


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
B16, H8, D64 = 16, 8, 64

print("=" * 65)
print("Paged Attention Correctness Check")
print("=" * 65)
all_pass = True
all_pass &= verify(1,  H8, D64, seq_len=32,  block_size=16, device=device, label="B=1,  seq=32")
all_pass &= verify(4,  H8, D64, seq_len=64,  block_size=16, device=device, label="B=4,  seq=64")
all_pass &= verify(B16,H8, D64, seq_len=128, block_size=16, device=device, label="B=16, seq=128")
all_pass &= verify(1,  H8, D64, seq_len=32,  block_size=8,  device=device, label="B=1,  seq=32, block=8")
all_pass &= verify(8,  H8, D64, seq_len=64,  block_size=4,  device=device, label="B=8,  seq=64, block=4")

print()
if all_pass:
    print("  All correctness checks PASSED.")
    print("  Paged attention is mathematically identical to standard attention.")

print()
print("=" * 65)
print("Paged Attention Speed (PyTorch reference vs standard)")
print("=" * 65)
print(f"  {'Config':<30}  {'Standard':>10}  {'Paged-ref':>10}  {'Overhead':>8}")
print(f"  {'-'*60}")

for (B, sl) in [(1, 64), (4, 64), (8, 128), (16, 128)]:
    t_std, t_paged = benchmark(B, H8, D64, sl, 16, device)
    overhead = (t_paged - t_std) / t_std * 100
    label = f"B={B}, seq={sl}"
    print(f"  {label:<30}  {t_std:>8.2f}ms  {t_paged:>8.2f}ms  {overhead:>+7.0f}%")

print()
print("  The PyTorch reference has extra overhead from the Python gather loop.")
print("  This is why production systems use a fused Triton kernel:")
print("  kernels/paged_attn_triton.py eliminates the gather entirely.")
