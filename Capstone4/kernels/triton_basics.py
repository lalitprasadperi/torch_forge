"""
Triton Kernels — Writing GPU Kernels in Python

WHAT IS TRITON?
───────────────
Triton is a Python-based DSL (domain-specific language) for writing GPU kernels.
It was created by OpenAI and is now the primary codegen target for TorchInductor.

TRITON vs CUDA:
  CUDA:   You write threads. You manage shared memory. You handle warp divergence.
          Total control, lots of complexity.

  Triton: You write BLOCKS. Triton handles threading, memory coalescing,
          shared memory, and warp synchronisation automatically.
          Less control, much less complexity, 80% of the performance.

THE PROGRAMMING MODEL:
  A Triton kernel launches a GRID of PROGRAMS (blocks).
  Each program processes a BLOCK of data.

  CUDA:       thread → warp → block → grid
  Triton:     program (block) → grid
              [Triton handles threading inside the block]

  Key primitives:
    tl.program_id(axis)     — which program (block) am I?
    tl.arange(0, BLOCK)     — vector of indices [0, 1, ..., BLOCK-1]
    tl.load(ptr, mask)      — load from HBM, masked
    tl.store(ptr, val, mask)— write to HBM, masked
    tl.dot(a, b)            — blocked matrix multiply (uses Tensor Cores)

AUTOTUNING:
  @triton.autotune decorates a kernel with a list of configs.
  Triton runs each config once and picks the fastest.
  Configs vary: BLOCK_SIZE, num_warps, num_stages (software pipeline depth).

Install:
  pip install triton   (bundled with PyTorch 2.x)

Run this file:
  python kernels/triton_basics.py
"""

import torch
import triton
import triton.language as tl
import time


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 1: Vector Addition
# The "Hello World" of GPU programming.
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def vector_add_kernel(
    x_ptr,          # pointer to input vector x
    y_ptr,          # pointer to input vector y
    out_ptr,        # pointer to output vector
    n_elements,     # total number of elements
    BLOCK_SIZE: tl.constexpr,   # tile size (must be a compile-time constant)
):
    """
    Each program processes BLOCK_SIZE elements.
    Program i handles elements [i*BLOCK_SIZE, (i+1)*BLOCK_SIZE).
    """
    pid = tl.program_id(axis=0)   # which block am I?

    # Compute the starting element index for this block
    block_start = pid * BLOCK_SIZE
    offsets     = block_start + tl.arange(0, BLOCK_SIZE)  # [start, start+1, ...]

    # Mask: don't load/store past the end of the vector
    mask = offsets < n_elements

    # Load from HBM (high bandwidth memory — the GPU's VRAM)
    x   = tl.load(x_ptr   + offsets, mask=mask, other=0.0)
    y   = tl.load(y_ptr   + offsets, mask=mask, other=0.0)

    # Compute
    out = x + y

    # Store result back to HBM
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Python wrapper around the Triton kernel."""
    assert x.is_cuda and y.is_cuda
    assert x.shape == y.shape

    out         = torch.empty_like(x)
    n_elements  = x.numel()
    BLOCK_SIZE  = 1024   # process 1024 elements per program

    # Number of programs to launch = ceil(n_elements / BLOCK_SIZE)
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    vector_add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 2: Softmax (row-wise)
# Classic reduction kernel. Each program handles one row.
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def softmax_kernel(
    x_ptr,
    out_ptr,
    n_cols,         # number of columns (vocab size, etc.)
    BLOCK: tl.constexpr,
):
    """
    One program per row. BLOCK >= n_cols (process whole row at once).
    This is a 'single-pass online softmax':
      1. Find max (numerical stability)
      2. Subtract max, compute exp
      3. Sum
      4. Divide by sum
    """
    row = tl.program_id(0)
    row_start = row * n_cols
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    # Load the row
    x = tl.load(x_ptr + row_start + cols, mask=mask, other=-float("inf"))

    # Numerically stable softmax
    x_max = tl.max(x, axis=0)
    x     = x - x_max           # subtract max
    num   = tl.exp(x)
    denom = tl.sum(num, axis=0)
    out   = num / denom

    tl.store(out_ptr + row_start + cols, out, mask=mask)


def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """Row-wise softmax using Triton."""
    assert x.is_cuda and x.ndim == 2
    rows, cols = x.shape

    # BLOCK must be a power of 2 and >= cols
    BLOCK = triton.next_power_of_2(cols)

    out  = torch.empty_like(x)
    grid = (rows,)   # one program per row

    softmax_kernel[grid](x, out, cols, BLOCK=BLOCK)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 3: Fused Bias + ReLU (elementwise)
# Demonstrates: writing fused kernels that torch.compile also generates.
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128},  num_warps=4),
        triton.Config({"BLOCK_SIZE": 256},  num_warps=4),
        triton.Config({"BLOCK_SIZE": 512},  num_warps=8),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
    ],
    key=["n_elements"],   # retune when n_elements changes
)
@triton.jit
def fused_bias_relu_kernel(
    x_ptr,
    bias_ptr,
    out_ptr,
    n_elements,
    n_cols,         # bias has shape (n_cols,), x has shape (n_rows, n_cols)
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel: out = relu(x + bias)
    One kernel, one HBM read, one HBM write.
    Unfused would be: write (x+bias) to HBM, read it back for relu.
    """
    pid     = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask    = offsets < n_elements

    x    = tl.load(x_ptr    + offsets,           mask=mask, other=0.0)
    bias = tl.load(bias_ptr + (offsets % n_cols), mask=mask, other=0.0)

    out  = tl.maximum(x + bias, 0.0)   # fused add + relu

    tl.store(out_ptr + offsets, out, mask=mask)


def triton_fused_bias_relu(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and bias.is_cuda
    assert bias.shape[0] == x.shape[-1]

    out        = torch.empty_like(x)
    n_elements = x.numel()
    n_cols     = x.shape[-1]
    grid       = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    fused_bias_relu_kernel[grid](x, bias, out, n_elements, n_cols)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Demonstration
# ─────────────────────────────────────────────────────────────────────────────

def benchmark(fn, *args, n_warmup=20, n_iter=200):
    for _ in range(n_warmup):
        fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iter * 1000


def demo():
    print("\n── Kernel 1: Vector Addition ─────────────────────────────────────")
    N = 1024 * 1024   # 1M elements
    x = torch.randn(N, device="cuda")
    y = torch.randn(N, device="cuda")

    triton_out = triton_vector_add(x, y)
    torch_out  = x + y
    print(f"  Max diff (Triton vs torch): {(triton_out - torch_out).abs().max():.2e}")

    t_triton = benchmark(triton_vector_add, x, y)
    t_torch  = benchmark(torch.add, x, y)
    print(f"  Triton: {t_triton:.3f} ms  |  torch: {t_torch:.3f} ms")
    print(f"  (Triton is close — both are memory-bandwidth-bound at this size)")

    print("\n── Kernel 2: Row-wise Softmax ────────────────────────────────────")
    x2 = torch.randn(512, 32768, device="cuda")   # 512 rows, 32K cols (vocab)

    triton_out2 = triton_softmax(x2)
    torch_out2  = torch.softmax(x2, dim=-1)
    print(f"  Max diff: {(triton_out2 - torch_out2).abs().max():.2e}")

    t_triton2 = benchmark(triton_softmax, x2)
    t_torch2  = benchmark(lambda x: torch.softmax(x, dim=-1), x2)
    print(f"  Triton: {t_triton2:.3f} ms  |  torch (cuDNN): {t_torch2:.3f} ms")

    print("\n── Kernel 3: Fused Bias + ReLU ───────────────────────────────────")
    x3   = torch.randn(2048, 4096, device="cuda")
    bias = torch.randn(4096, device="cuda")

    triton_out3 = triton_fused_bias_relu(x3, bias)
    torch_out3  = torch.relu(x3 + bias)
    print(f"  Max diff: {(triton_out3 - torch_out3).abs().max():.2e}")

    t_triton3 = benchmark(triton_fused_bias_relu, x3, bias)
    t_torch3  = benchmark(lambda a, b: torch.relu(a + b), x3, bias)
    print(f"  Triton (fused): {t_triton3:.3f} ms  |  torch (2 kernels): {t_torch3:.3f} ms")
    print(f"  Fusion speedup: {t_torch3/t_triton3:.2f}×")

    print()
    print("  Note: torch.compile would generate the same fused kernel automatically.")
    print("  Writing Triton manually lets you control tiling and pipeline depth.")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA required for Triton")
    else:
        demo()
        print("\nNext: kernels/matmul.py")
