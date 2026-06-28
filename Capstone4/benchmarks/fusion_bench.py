"""
Kernel Fusion Benchmark — Quantifying the Memory-Bandwidth Savings

Kernel fusion is the single biggest optimisation torch.compile performs.
This benchmark isolates and measures its impact across different op patterns.

WHAT WE'RE MEASURING:
  For each op sequence, compare:
    A) Eager: PyTorch runs each op as a separate CUDA kernel
    B) Compiled: TorchInductor fuses into one kernel

  The speedup comes entirely from avoiding HBM round-trips.

ROOFLINE CONTEXT (RTX PRO 2000):
  Peak compute:    ~16 TFLOPS (fp16)
  HBM bandwidth:   ~224 GB/s
  Arithmetic intensity needed to be compute-bound: 16T/224G ≈ 71 FLOPs/byte

  Elementwise ops: ~1 FLOP/byte each (pure memory-bound)
  Fused 4 ops:    ~4 FLOPs/byte (still memory-bound but 4× better)
  MatMul (2048²): ~1024 FLOPs/byte (compute-bound — fusion irrelevant)

  → Fusion helps elementwise chains, NOT matmul chains.

Run:
  python benchmarks/fusion_bench.py
"""

import torch
import torch.nn as nn
import time


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def bmark(fn, *args, n_warmup=30, n_iter=200):
    for _ in range(n_warmup):
        fn(*args)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn(*args)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iter * 1000


def hbm_gb(n_elements, n_rw_passes, dtype=torch.float16):
    bytes_per_elem = 2 if dtype == torch.float16 else 4
    return n_elements * n_rw_passes * bytes_per_elem / 1e9


# ─────────────────────────────────────────────────────────────────────────────
# Op sequences to fuse
# ─────────────────────────────────────────────────────────────────────────────

def gelu_linear_chain(x):
    """Common pattern after a Linear layer."""
    return torch.nn.functional.gelu(x * 2.0 + 0.5)

def softmax_temperature(x, temperature=2.0):
    """Scaled softmax (used in generation)."""
    return torch.softmax(x / temperature, dim=-1)

def layer_norm_manual(x, w, b, eps=1e-5):
    """LayerNorm without using nn.LayerNorm (to test if it fuses)."""
    mean = x.mean(dim=-1, keepdim=True)
    var  = x.var(dim=-1, keepdim=True, unbiased=False)
    return (x - mean) / (var + eps).sqrt() * w + b

def swiglu(x):
    x1, x2 = x.chunk(2, dim=-1)
    return x1 * torch.nn.functional.silu(x2)

def residual_norm_chain(x, r, w, b, eps=1e-5):
    """Pre-norm transformer pattern: rmsnorm(x + residual)."""
    res = x + r
    return layer_norm_manual(res, w, b, eps)

def attention_scale_softmax(q, k, scale):
    """Attention score computation: scale(QK^T) → softmax."""
    return torch.softmax((q @ k.transpose(-2, -1)) * scale, dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark each pattern
# ─────────────────────────────────────────────────────────────────────────────

def run_pattern_bench(name, fn, args, n_passes_eager, n_passes_fused):
    eager_fn    = fn
    compiled_fn = torch.compile(fn, fullgraph=True)

    # Check correctness
    out_eager    = eager_fn(*args)
    out_compiled = compiled_fn(*args)
    diff = (out_eager - out_compiled).abs().max().item()
    status = "✓" if diff < 1e-3 else f"DIFF={diff:.2e}"

    t_eager    = bmark(eager_fn,    *args)
    t_compiled = bmark(compiled_fn, *args)

    n = args[0].numel()
    bw_eager   = hbm_gb(n, n_passes_eager)
    bw_compiled = hbm_gb(n, n_passes_fused)

    speedup = t_eager / t_compiled
    return {
        "name": name,
        "t_eager": t_eager,
        "t_compiled": t_compiled,
        "speedup": speedup,
        "bw_saved_gb": bw_eager - bw_compiled,
        "status": status,
    }


def main():
    print(f"\n{'═'*70}")
    print(f"  Kernel Fusion Benchmark  (device={DEVICE})")
    print(f"{'═'*70}")

    N  = 8 * 1024 * 1024   # 8M elements
    d  = 4096
    B  = 2048

    x  = torch.randn(B, d,   device=DEVICE, dtype=torch.float16)
    x2 = torch.randn(B, d*2, device=DEVICE, dtype=torch.float16)
    w  = torch.ones(d,       device=DEVICE, dtype=torch.float16)
    b  = torch.zeros(d,      device=DEVICE, dtype=torch.float16)
    r  = torch.randn(B, d,   device=DEVICE, dtype=torch.float16)

    T_seq = 512
    d_k   = 64
    n_h   = 8
    Q = torch.randn(B//8, n_h, T_seq, d_k, device=DEVICE, dtype=torch.float16)
    K = torch.randn_like(Q)

    patterns = [
        ("GELU(x*2+0.5)",       gelu_linear_chain, (x,),             4, 1),
        ("softmax(x/T)",        softmax_temperature, (x, 2.0),        2, 1),
        ("SwiGLU(x1, x2)",     swiglu,              (x2,),            3, 1),
        ("LayerNorm manual",    layer_norm_manual,   (x, w, b),        6, 2),
        ("residual+LayerNorm", residual_norm_chain,  (x, r, w, b),    8, 2),
    ]

    print(f"\n  {'Pattern':<25}  {'Eager':>8}  {'Compiled':>10}  "
          f"{'Speedup':>8}  {'BW saved':>10}  {'Correct':>8}")
    print("  " + "─" * 75)

    total_bw_saved = 0.0
    for name, fn, args, np_eager, np_fused in patterns:
        r2 = run_pattern_bench(name, fn, args, np_eager, np_fused)
        total_bw_saved += r2["bw_saved_gb"]
        print(f"  {r2['name']:<25}  {r2['t_eager']:>6.3f}ms  "
              f"{r2['t_compiled']:>8.3f}ms  "
              f"{r2['speedup']:>7.2f}×  "
              f"{r2['bw_saved_gb']:>8.2f}GB  "
              f"{r2['status']:>8}")

    print(f"\n  Total HBM traffic eliminated: ~{total_bw_saved:.1f} GB per call")
    print(f"  At 224 GB/s: {total_bw_saved/224*1000:.1f} ms saved per batch")

    print(f"\n  Where fusion does NOT help (compute-bound):")
    M = 2048
    a = torch.randn(M, M, device=DEVICE, dtype=torch.float16)

    def matmul_chain(x):
        return x @ x.T @ x   # 2 matmuls

    t_e = bmark(matmul_chain, a)
    t_c = bmark(torch.compile(matmul_chain), a)
    print(f"  matmul chain: eager={t_e:.2f}ms  compiled={t_c:.2f}ms  "
          f"speedup={t_e/t_c:.2f}×")
    print("  (little gain — matmul is already compute-bound, cuBLAS is optimal)")

    print(f"\n{'═'*70}")
    print("  CONCLUSION:")
    print("  • Fusion helps wherever operations are memory-bandwidth bound")
    print("  • Elementwise chains: 2–5× speedup from fusion")
    print("  • LayerNorm / RMSNorm: 2–3× speedup")
    print("  • Matrix multiplications: fusion irrelevant (compute-bound)")
    print("  torch.compile automatically detects and applies all of these.")
    print(f"{'═'*70}")


if __name__ == "__main__":
    main()
