"""
TorchInductor — The Default Backend: FX Graph → Triton Kernels

WHAT INDUCTOR DOES
──────────────────
Inductor takes an FX graph and:
  1. Converts FX nodes to Inductor IR (a lower-level graph)
  2. Applies optimisations: loop fusion, memory planning, layout optimisation
  3. Generates Triton kernels (GPU) or C++ with OpenMP (CPU)
  4. Caches the generated kernels (in ~/.cache/torch_extensions/)

INDUCTOR IR
───────────
Between FX and Triton, Inductor has its own IR based on "loops":
  Each operation is expressed as a pointwise or reduction loop.
  Inductor can then FUSE loops that have compatible iteration spaces.

  Example: relu(x + y)
    FX nodes: add, relu
    Inductor IR:
      for i in range(N):
          tmp = x[i] + y[i]      ← add loop
          out[i] = max(tmp, 0)   ← relu loop (same iteration space)
    After fusion:
      for i in range(N):
          out[i] = max(x[i] + y[i], 0)   ← ONE loop, ONE kernel

  The single fused kernel reads x,y once and writes out once.
  Unfused: reads x,y, writes tmp, reads tmp, writes out (2× memory traffic).

HOW TO SEE WHAT INDUCTOR GENERATES
────────────────────────────────────
  Method 1: TORCH_LOGS="output_code" python my_script.py
  Method 2: torch._inductor.config.debug = True
  Method 3: torch._inductor.codecache.CachingAutotuner (inspect .kernel attr)

The generated Triton code is saved to:
  ~/.cache/torch_extensions/inductor_*/

Run this file:
  TORCH_LOGS="output_code" python compiler/inductor/ir_inspect.py
"""

import torch
import torch._inductor.config as inductor_config


def demo_see_generated_code():
    """
    Enable Inductor debug output to see the generated Triton kernels.
    Output goes to stderr when TORCH_LOGS="output_code" is set.
    """
    print("\n── Inductor kernel codegen ──────────────────────────────────────")
    print("  Run with: TORCH_LOGS='output_code' python compiler/inductor/ir_inspect.py")
    print("  to see the generated Triton kernels printed to stdout.\n")

    def fused_fn(x, y):
        return torch.relu(x + y)

    compiled = torch.compile(fused_fn)
    x = torch.randn(1024, device="cuda")
    y = torch.randn(1024, device="cuda")

    # Warmup triggers compilation and codegen
    for _ in range(3):
        out = compiled(x, y)

    print("  Compilation done. The Triton kernel for relu(x+y) fuses:")
    print("    - elementwise add")
    print("    - elementwise relu")
    print("  into a SINGLE GPU kernel (one HBM read + one HBM write).")


def demo_inductor_options():
    """
    Inductor has many configuration options that control behaviour.
    These are set via torch._inductor.config.
    """
    print("\n── Key Inductor configuration options ───────────────────────────")

    options = {
        "triton.cudagraphs":
            "Wrap compiled kernels in CUDA Graphs (eliminates launch overhead)",
        "max_fusion_size":
            "Max number of ops to fuse into one kernel (default 64)",
        "unroll_reductions_threshold":
            "Unroll small reductions (sum over dim=1 of small tensors)",
        "coordinate_descent_tuning":
            "Auto-tune tile sizes via coordinate descent (slower compile, faster runtime)",
        "benchmark_kernel":
            "Benchmark each generated kernel to pick fastest tile config",
        "conv_1x1_as_mm":
            "Fuse 1×1 convolutions into matmul (faster on Tensor Cores)",
        "shape_padding":
            "Pad tensor dimensions for better memory alignment",
    }

    for key, desc in options.items():
        print(f"  inductor_config.{key}")
        print(f"    → {desc}")
        print()


def demo_kernel_cache():
    """
    Inductor caches compiled kernels to disk. On re-run, it skips
    recompilation if inputs have the same shape/dtype (guards pass).
    """
    print("\n── Inductor kernel cache ────────────────────────────────────────")
    import os
    cache_dir = os.path.expanduser("~/.cache/torch_extensions")
    print(f"  Kernel cache location: {cache_dir}")

    def fn(x):
        return torch.relu(x) * 2 + x

    compiled = torch.compile(fn)
    x = torch.randn(256, device="cuda")

    import time
    t0 = time.perf_counter()
    compiled(x)   # first call: compile + run
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    compiled(x)   # second call: cache hit, just run
    torch.cuda.synchronize()
    t2 = time.perf_counter()

    print(f"  First call  (compile+run): {(t1-t0)*1000:.0f} ms")
    print(f"  Second call (cache hit):   {(t2-t1)*1000:.1f} ms")
    print(f"  Speedup from cache: ~{(t1-t0)/(t2-t1):.0f}×")


def demo_fusion_comparison():
    """
    Measure the speedup from Inductor's op fusion on elementwise chains.
    """
    print("\n── Fusion speedup: unfused vs Inductor ──────────────────────────")
    import time

    N = 16 * 1024 * 1024   # 16M elements

    def unfused(x):
        a = x * 2
        b = torch.relu(a)
        c = b + 1
        d = torch.sigmoid(c)
        return d

    compiled_fn = torch.compile(unfused)

    x = torch.randn(N, device="cuda")

    # Warmup
    for _ in range(5):
        unfused(x)
        compiled_fn(x)
    torch.cuda.synchronize()

    # Benchmark
    N_ITER = 100
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        unfused(x)
    torch.cuda.synchronize()
    t_eager = (time.perf_counter() - t0) / N_ITER * 1000

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        compiled_fn(x)
    torch.cuda.synchronize()
    t_compiled = (time.perf_counter() - t0) / N_ITER * 1000

    print(f"  Eager   (4 separate kernels): {t_eager:.2f} ms")
    print(f"  Compiled (1 fused kernel):    {t_compiled:.2f} ms")
    print(f"  Speedup: {t_eager/t_compiled:.2f}×")
    print()
    print("  WHY: eager makes 4 HBM passes (read+write each op)")
    print("       compiled makes 1 HBM pass (read x once, write d once)")
    bw_saved = 3 * N * 4 * 2 / 1e9   # 3 intermediate tensors, float32, read+write
    print(f"  Estimated HBM traffic saved: {bw_saved:.1f} GB per call")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA required for Inductor GPU kernels")
    else:
        demo_see_generated_code()
        demo_inductor_options()
        demo_kernel_cache()
        demo_fusion_comparison()
        print("\nNext: compiler/cuda_graphs/basics.py")
