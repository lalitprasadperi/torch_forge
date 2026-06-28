"""
Tour: Kernel Fusion — The Most Important GPU Optimisation

Everything torch.compile does ultimately reduces to one principle:
fewer trips to GPU memory. This tour explains WHY, shows HOW,
and lets you MEASURE the difference.

Run:
  python tours/fusion_tour.py
"""

import torch
import torch.nn as nn
import time

COLS = 68
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def box(title):
    line = "─" * (COLS - 2)
    print(f"\n┌{line}┐")
    pad = (COLS - 2 - len(title)) // 2
    print(f"│{' '*pad}{title}{' '*(COLS-2-pad-len(title))}│")
    print(f"└{line}┘")

def lesson(n, title):
    print(f"\n{'═'*COLS}")
    print(f"  Lesson {n}: {title}")
    print(f"{'═'*COLS}")

def explain(text):
    for line in text.strip().split("\n"):
        print(f"  {line}")
    print()

def show(label):
    print(f"  ► {label}")

def bmark(fn, *args, n=200):
    for _ in range(30):
        fn(*args)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn(*args)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000


# ─────────────────────────────────────────────────────────────────────────────

lesson(1, "The Memory Wall")

explain("""
A GPU computes at 16 TFLOPS. Its memory transfers at 224 GB/s.
That sounds fast. Let's check if it's fast ENOUGH.

Take a simple op: relu(x) on a 2048×4096 tensor of fp16.
  Size:         2048 * 4096 * 2 bytes = 16 MB
  Read time:    16 MB / 224 GB/s = 0.071 ms
  Write time:   0.071 ms
  Total:        0.14 ms just for memory traffic

  Actual FLOP count: 2048 * 4096 * 1 = 8.4M FLOPs
  At 16 TFLOPS: 8.4M / 16T = 0.00053 ms to COMPUTE

The GPU spends 0.14ms moving data and 0.0005ms computing.
That's 99.6% of time in memory transfers. PURE memory bottleneck.

Now consider a transformer block on the SAME tensor:
  ops: add_residual, rmsnorm, linear1, gelu, linear2, add_residual, rmsnorm
  That's 5+ elementwise ops, each reading+writing 16 MB.
  Just the elementwise memory traffic: 5 * 2 * 16 MB = 160 MB
  At 224 GB/s: 0.71 ms — before any linear layers!

FUSION ELIMINATES THIS: read once, apply ALL ops, write once.
  With fusion: 2 * 16 MB = 32 MB = 0.14 ms
  Without fusion: 160 MB = 0.71 ms
  Speedup: 5×. For FREE. From torch.compile.
""")

# ─────────────────────────────────────────────────────────────────────────────

lesson(2, "What Fusion Looks Like")

explain("""
Without fusion (eager PyTorch):
  tmp1 = x + residual    → 1 kernel: reads x (16MB), r (16MB), writes tmp1 (16MB)
  tmp2 = tmp1 ** 2       → 1 kernel: reads tmp1, writes tmp2
  mean = tmp2.mean(-1)   → 1 kernel: reads tmp2, writes mean
  tmp3 = x / (mean+eps)  → 1 kernel: reads x, mean, writes tmp3
  out  = tmp3 * weight   → 1 kernel: reads tmp3, w, writes out

  Total HBM traffic: ~9 reads + 5 writes = 14 * 16 MB = 224 MB

With fusion (torch.compile / Triton):
  out = fused_residual_rmsnorm(x, r, w)
  → 1 kernel: reads x (16MB), r (16MB), w (tiny), writes out (16MB)

  Total HBM traffic: 3 * 16 MB = 48 MB
  Speedup: 224/48 = 4.7×
""")

show("Key fusion rule:")
print("""
  TWO ops can be fused if they have the SAME iteration space
  (i.e., each output element of op2 depends on exactly one output of op1).

  FUSEABLE:
    • relu(x + y)         — both pointwise, same shape
    • sigmoid(x * scale)  — both pointwise
    • norm(x + residual)  — both over same rows

  NOT FUSEABLE (different iteration spaces):
    • matmul then relu    — matmul produces new shape, then relu
                            (well, they CAN be fused but it's complex)
    • two matmuls         — output of first is input of second,
                            but blocking structure doesn't match
""")

# ─────────────────────────────────────────────────────────────────────────────

lesson(3, "Inductor's Fusion Strategy")

explain("""
TorchInductor applies fusion in a specific order:

  1. POINTWISE FUSION:
     All elementwise ops that touch the same tensor are fused.
     relu(x+y)*2 → one kernel reading x,y and writing output.

  2. REDUCTION FUSION:
     A pointwise op feeding into a reduction can be fused.
     mean(relu(x)) → compute relu inline while accumulating sum.
     (this avoids writing relu(x) to memory at all)

  3. PERSISTENT REDUCTION:
     For reductions over small dimensions (e.g. LayerNorm over d=256):
     Load the entire row into registers, do all computation, write once.

  4. TEMPLATE-BASED:
     Special kernels for matmul (Cutlass templates), softmax, attention.
     These aren't expressed as loops — they're pre-written expert kernels.

Priority: matmul NEVER gets merged into pointwise (it's a different algorithm).
          The split is: matmul kernels + elementwise kernels.

In practice for a transformer block:
  • Q,K,V projections:  3 separate cuBLAS GEMMs (can't fuse matmuls)
  • Attention scores:   Flash Attention kernel (tiled, fused internally)
  • FFN:                2 GEMMs + 1 fused (bias+GELU) kernel
  • Layer norms:        1 fused kernel per norm
  • Total kernels per block:  ~10 instead of ~40 eager kernels
""")

# ─────────────────────────────────────────────────────────────────────────────

lesson(4, "Live Fusion Demo: Measure the Savings")

if DEVICE == "cuda":
    B, d = 2048, 4096

    print(f"  Config: B={B}, d={d}, dtype=fp16")
    print(f"  Tensor size per activation: {B*d*2/1e6:.1f} MB\n")

    x = torch.randn(B, d, device=DEVICE, dtype=torch.float16)
    w = torch.ones(d,    device=DEVICE, dtype=torch.float16)
    r = torch.randn(B, d, device=DEVICE, dtype=torch.float16)

    # Pattern 1: GELU chain
    def gelu_chain(x):
        a = x * 2.0
        b = a + 0.5
        c = torch.nn.functional.gelu(b)
        return c

    gelu_compiled = torch.compile(gelu_chain)

    t_eager   = bmark(gelu_chain, x)
    t_compiled = bmark(gelu_compiled, x)
    show(f"GELU chain (x*2+0.5 → GELU):")
    print(f"      Eager:    {t_eager:.3f} ms  (3 kernels, 6 HBM passes)")
    print(f"      Compiled: {t_compiled:.3f} ms  (1 kernel, 2 HBM passes)")
    print(f"      Speedup:  {t_eager/t_compiled:.2f}×")

    # Pattern 2: RMSNorm manual
    def rmsnorm(x, w, eps=1e-6):
        x32 = x.float()
        rms = x32.pow(2).mean(-1, keepdim=True).add(eps).rsqrt()
        return (x32 * rms * w.float()).to(x.dtype)

    rmsnorm_compiled = torch.compile(rmsnorm)
    t_e2 = bmark(rmsnorm, x, w)
    t_c2 = bmark(rmsnorm_compiled, x, w)
    print()
    show(f"RMSNorm manual:")
    print(f"      Eager:    {t_e2:.3f} ms  (~5 intermediate tensors)")
    print(f"      Compiled: {t_c2:.3f} ms  (1 persistent reduction kernel)")
    print(f"      Speedup:  {t_e2/t_c2:.2f}×")

    # Pattern 3: Residual + RMSNorm
    def res_rmsnorm(x, r, w):
        res = x + r
        return rmsnorm(res, w)

    res_compiled = torch.compile(res_rmsnorm)
    t_e3 = bmark(res_rmsnorm, x, r, w)
    t_c3 = bmark(res_compiled, x, r, w)
    print()
    show(f"Residual + RMSNorm (Llama pre-norm):")
    print(f"      Eager:    {t_e3:.3f} ms")
    print(f"      Compiled: {t_c3:.3f} ms")
    print(f"      Speedup:  {t_e3/t_c3:.2f}×")
    total_saved = t_e2 + t_e3 - t_c2 - t_c3
    print(f"      Time saved per block: {total_saved:.3f} ms")
    print(f"      Over 32 layers: {total_saved*32:.1f} ms per forward pass")

else:
    explain("CUDA not available — skipping live benchmarks.")

# ─────────────────────────────────────────────────────────────────────────────

lesson(5, "Fusion Patterns in Transformers")

explain("""
Where does fusion actually help in a transformer?

  ┌─────────────────────────────────────────────────────┐
  │ Transformer Block (one layer)                       │
  │                                                     │
  │  x ──────────────────────────────────────┐          │
  │  │                                       │          │
  │  ▼                                       │          │
  │  [RMSNorm] ◄── FUSED: residual+norm      │          │
  │  │                                       │          │
  │  ▼                                       │          │
  │  [QKV Linear] = GEMM (separate kernel)   │          │
  │  │                                       │          │
  │  ▼                                       │          │
  │  [FlashAttention] = fused internally     │          │
  │  │                                       │          │
  │  ▼                                       │          │
  │  [Out Proj] = GEMM                       │          │
  │  │                                       ▼          │
  │  [+ residual] ◄─────────────────── FUSED: add+norm │
  │  │                                                  │
  │  ▼                                                  │
  │  [RMSNorm]                                          │
  │  │                                                  │
  │  ▼                                                  │
  │  [Gate Linear] [Up Linear] = 2 GEMMs                │
  │  │                                                  │
  │  ▼                                                  │
  │  [SwiGLU] ◄── FUSED: gate*silu(up)                  │
  │  │                                                  │
  │  ▼                                                  │
  │  [Down Linear] = GEMM                               │
  │  │                                                  │
  │  [+ residual]                                       │
  └─────────────────────────────────────────────────────┘

  Fused operations per block: ~6
  Unfused ops that stay separate: ~5 GEMMs + FlashAttention

  Without compile: ~40+ kernel launches per block
  With compile:    ~12 kernel launches per block
  Speedup from kernel count reduction alone: ~3×
""")

# ─────────────────────────────────────────────────────────────────────────────

lesson(6, "Advanced: Horizontal Fusion (Batching Small Ops)")

explain("""
Sometimes you have MANY small independent kernels.
Example: batch normalization in ResNet runs one kernel per channel group.
Each kernel is tiny — launch overhead dominates.

HORIZONTAL FUSION: run multiple independent kernels in ONE launch.

  Before: kernel_A(x1), kernel_A(x2), kernel_A(x3)  ← 3 launches
  After:  kernel_A_batched([x1, x2, x3])              ← 1 launch

PyTorch has this for grouped operations:
  • F.group_norm     → fuses N separate layer norms
  • grouped GEMMs    → one GEMM call for N independent linear layers
  • torch.vmap       → vectorise a function over a batch dimension

When does it matter?
  ResNet-50 with many small batch norms: 2–3× from horizontal fusion alone.
  Transformer with large matmuls: minimal impact.
""")

# ─────────────────────────────────────────────────────────────────────────────

lesson(7, "When NOT to Fuse")

explain("""
Fusion is not always beneficial.

  1. Very large ops that are already compute-bound:
     Matmul at 2048×2048 is compute-bound.
     Fusing a relu after it doesn't help — the relu is free.

  2. Ops with incompatible memory layouts:
     NHWC conv + NCHW norm — need transpose in between, can't fuse.

  3. Ops that benefit from separate autotuning:
     A fused kernel has one tile size for all ops.
     Separate kernels can be individually tuned.

  4. Reductions that change the tensor shape:
     sum(x, dim=0): shape goes from (B, D) → (D,).
     What follows can't be fused with what came before (different shape).
     (Actually Inductor CAN fuse in some cases — pointwise → reduction is OK)

The right mental model:
  Fuse whenever it reduces HBM traffic.
  Don't fuse if it prevents better individual kernel tuning.
  torch.compile makes most of these decisions automatically.
""")

# ─────────────────────────────────────────────────────────────────────────────

box("Fusion Tour Complete")
print("""
  Key Takeaways:
    1. Most transformer ops are memory-bound — they spend 99%+ of time in HBM.
    2. Fusion reduces HBM traffic by eliminating intermediate tensor writes.
    3. torch.compile fuses automatically — you get this for free.
    4. Writing Triton kernels lets you fuse ops that torch.compile can't.
    5. GEMMs (matmuls) are compute-bound — fusion irrelevant for them.

  The formula:
    Speedup from fusion ≈ (N ops) / (HBM passes with fusion)
    For pure elementwise chains: close to N× speedup.
    For mixed (matmul + elementwise): speedup on the elementwise portion.

  Files to run next:
    benchmarks/compile_speedup.py  → full model speedup measurement
    benchmarks/fusion_bench.py     → fusion speedup by op pattern
    benchmarks/cuda_graph_bench.py → CUDA Graph overhead measurements
""")
