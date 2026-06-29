"""
Example 12 — Triton Paged Attention Kernel

Benchmarks the Triton kernel from kernels/paged_attn_triton.py
against the PyTorch reference and standard (contiguous) attention.

The Triton kernel fuses the gather + attention into a single pass:
  - No intermediate gathered K,V tensors allocated
  - Each (batch, head) program reads KV blocks directly from the pool
  - Online softmax: numerically stable without materialising all scores

Run:  python examples/12_triton_paged_attention.py  (requires CUDA)
"""
import sys; sys.path.insert(0, ".")
import math
import time
import torch

if not torch.cuda.is_available():
    print("CUDA required for Triton kernel. Exiting.")
    sys.exit(0)

from model.paged_attention import paged_attention_decode
from kernels.paged_attn_triton import triton_paged_attention_decode


def make_paged_cache(B, H, D, seq_len, block_size, device):
    n_blocks     = math.ceil(seq_len / block_size)
    num_blocks   = B * n_blocks
    k_cache      = torch.randn(num_blocks, block_size, H, D, device=device, dtype=torch.float16)
    v_cache      = torch.randn(num_blocks, block_size, H, D, device=device, dtype=torch.float16)
    kv_cache_ref = torch.stack([k_cache, v_cache], dim=1)  # (num_blocks, 2, block_size, H, D)

    block_tables = torch.zeros(B, n_blocks, dtype=torch.int32, device=device)
    for i in range(B):
        block_tables[i] = torch.arange(i * n_blocks, (i + 1) * n_blocks, dtype=torch.int32)

    seq_lens = torch.full((B,), seq_len, dtype=torch.int32, device=device)
    return k_cache, v_cache, kv_cache_ref, block_tables, seq_lens


def timed(fn, n=100):
    for _ in range(5): fn()  # warmup
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000


# ── Correctness check ─────────────────────────────────────────────────────────
print("=" * 65)
print("Correctness: Triton vs PyTorch Reference")
print("=" * 65)
torch.manual_seed(42)

configs = [(1, 8, 64, 64), (4, 8, 64, 128), (8, 12, 64, 256)]
for B, H, D, seq_len in configs:
    block_size = 16
    scale      = 1.0 / math.sqrt(D)
    k_cache, v_cache, kv_cache_ref, block_tables, seq_lens = \
        make_paged_cache(B, H, D, seq_len, block_size, "cuda")
    q = torch.randn(B, H, D, device="cuda", dtype=torch.float16)

    ref    = paged_attention_decode(q, kv_cache_ref, block_tables, seq_lens, scale)
    triton = triton_paged_attention_decode(q, k_cache, v_cache, block_tables, seq_lens, scale)

    diff  = (ref.float() - triton.float()).abs().max().item()
    check = "✓" if diff < 0.05 else "✗"
    print(f"  {check} B={B}, H={H}, D={D}, seq={seq_len}  max_diff={diff:.3e}")


# ── Speed benchmark ───────────────────────────────────────────────────────────
print()
print("=" * 65)
print("Speed: Standard vs PyTorch-ref vs Triton  (fp16, CUDA)")
print("=" * 65)
print(f"  {'Config':<24}  {'Std-attn':>10}  {'PyRef':>10}  {'Triton':>10}  {'Speedup':>8}")
print(f"  {'-'*68}")

for (B, H, D, seq_len) in [(1, 8, 64, 128), (4, 8, 64, 256), (16, 12, 64, 256)]:
    block_size = 16
    scale      = 1.0 / math.sqrt(D)
    k_cache, v_cache, kv_cache_ref, block_tables, seq_lens = \
        make_paged_cache(B, H, D, seq_len, block_size, "cuda")
    q = torch.randn(B, H, D, device="cuda", dtype=torch.float16)

    # Standard: q @ K^T over full contiguous K
    k_cont = torch.randn(B, seq_len, H, D, device="cuda", dtype=torch.float16)
    v_cont = torch.randn(B, seq_len, H, D, device="cuda", dtype=torch.float16)
    k_h = k_cont.permute(0, 2, 1, 3); v_h = v_cont.permute(0, 2, 1, 3)
    q_h = q.unsqueeze(2)
    def std_fn():
        scores = torch.matmul(q_h, k_h.transpose(-1, -2)) * scale
        return torch.matmul(torch.softmax(scores, dim=-1), v_h).squeeze(2)

    def pyref_fn():
        return paged_attention_decode(q, kv_cache_ref, block_tables, seq_lens, scale)

    def triton_fn():
        return triton_paged_attention_decode(q, k_cache, v_cache, block_tables, seq_lens, scale)

    t_std    = timed(std_fn)
    t_pyref  = timed(pyref_fn)
    t_triton = timed(triton_fn)
    speedup  = t_pyref / t_triton

    label = f"B={B}, H={H}, seq={seq_len}"
    print(f"  {label:<24}  {t_std:>8.3f}ms  {t_pyref:>8.3f}ms  {t_triton:>8.3f}ms  {speedup:>7.1f}×")

print()
print("  Triton kernel eliminates the Python gather loop and avoids")
print("  materialising gathered K,V tensors — fewer HBM round-trips.")
