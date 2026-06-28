"""
Tour: Triton — Writing GPU Kernels in Python

From zero to a working matrix multiply in Triton.

Run:
  python tours/triton_tour.py
"""

import torch
import triton
import triton.language as tl
import math

COLS = 68

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

def show(label, value=""):
    print(f"  ► {label}")
    if value:
        for line in str(value).split("\n"):
            print(f"      {line}")


# ─────────────────────────────────────────────────────────────────────────────

lesson(1, "GPU Architecture: Why GPU Kernels Exist")

explain("""
A CPU has 4–16 cores, each fast (GHz, out-of-order, big caches).
A GPU has thousands of cores, each slower, but massively parallel.

RTX PRO 2000 (your GPU):
  2560 CUDA cores (little processors)
  20 Streaming Multiprocessors (SMs)
  128 CUDA cores per SM
  16 GB VRAM (HBM — High Bandwidth Memory)
  HBM bandwidth: ~224 GB/s
  Peak compute:  ~16 TFLOPS (fp16)

The key insight: GPUs are fast at doing the SAME thing to MANY elements.
  CPU: x[0] += 1; x[1] += 1; ... x[N] += 1  ← sequential
  GPU: all x[i] += 1 simultaneously           ← parallel

A CUDA KERNEL is a function that runs on EVERY element in parallel.
You write it for ONE element; the GPU runs it for ALL elements at once.

CUDA THREAD HIERARCHY:
  Thread      → one "worker" (one element)
  Warp        → 32 threads that execute in lockstep (SIMD)
  Block       → group of warps sharing fast "shared memory" (SRAM)
  Grid        → all blocks executing the kernel

TRITON PROGRAMMING MODEL:
  You write for BLOCKS, not threads.
  Each Triton program = one block.
  Triton handles the threading inside the block.
  You describe WHAT data to process; Triton handles HOW.
""")

# ─────────────────────────────────────────────────────────────────────────────

lesson(2, "Memory Hierarchy: Why Bandwidth Matters")

explain("""
GPU memory has a strict hierarchy (fast/small → slow/large):

  Registers    ~8,192 per thread  ~2 TB/s  (invisible — compiler managed)
  L1/SRAM      ~128 KB per SM     ~19 TB/s (shared memory in CUDA)
  L2 Cache     ~64 MB             ~7 TB/s  (automatically managed)
  HBM (VRAM)   ~16 GB             ~224 GB/s (where your tensors live)

The bottleneck for most ops is HBM bandwidth.

Example: relu(x) where x has 1B float32 elements = 4 GB
  Reading x: 4 GB / 224 GB/s = 17.8 ms
  Writing:   4 GB / 224 GB/s = 17.8 ms
  Total:     ~35 ms just to read+write (computation is negligible!)

ARITHMETIC INTENSITY = FLOPs / bytes
  relu:       1 FLOP / 4 bytes   = 0.25 FLOPs/byte (memory-bound)
  add(x,y):  1 FLOP / 8 bytes   = 0.125 FLOPs/byte (memory-bound)
  matmul(M²): 2M³ / 4*(2M²) ≈ M/4 FLOPs/byte (compute-bound for large M)

  RTX ridge point: 16 TFLOPS / 224 GB/s = 71 FLOPs/byte
  Below 71: memory-bound (fusion helps)
  Above 71: compute-bound (tiling/Tensor Cores help)
""")

show("Your RTX PRO 2000 Roofline:")
print("""
  Throughput
  (TFLOPS)
    16 ┤                              ●  matmul (compute-bound)
       │                         ●●●
    8  ┤                    ●●●●
       │               ●●●●
    4  ┤          ●●●●
       │     ●●●●  ← memory-bound region
    2  ┤●●●●
       │
    1  ┤
       └──────────────────────────────────── Intensity (FLOPs/byte)
              1      10      71     1000
              relu  softmax matmul

  Operations in memory-bound region: benefit from FUSION (reduce HBM passes)
  Operations in compute-bound region: benefit from TILING (Tensor Cores)
""")

# ─────────────────────────────────────────────────────────────────────────────

lesson(3, "Triton Building Blocks")

explain("""
Triton programs are written as Python functions decorated with @triton.jit.
They compile to PTX (GPU assembly) and run natively on the GPU.

KEY PRIMITIVES:
  tl.program_id(axis)       → which block am I? (like blockIdx in CUDA)
  tl.arange(0, N)           → [0, 1, 2, ..., N-1] (vector of indices)
  tl.load(ptr + offs, mask) → load from GPU memory (HBM → registers)
  tl.store(ptr + offs, v)   → store to GPU memory (registers → HBM)
  tl.dot(a, b)              → blocked matrix multiply (uses Tensor Cores)
  tl.sum(x, axis=0)         → reduce over a vector
  tl.max(x, axis=0)         → max over a vector
  tl.exp(x)                 → elementwise exp (vectorised)
  tl.zeros((M, N), dtype)   → zero-filled tensor in registers

TYPES:
  tl.float16 / tl.float32 / tl.int32 / tl.int64
  tl.constexpr  → compile-time constant (e.g. BLOCK_SIZE)
                  Must be power of 2. Used for static array sizes.

MASKING:
  Triton processes data in fixed-size blocks.
  If N isn't divisible by BLOCK_SIZE, the last block needs masking:
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x    = tl.load(ptr + offs, mask=mask, other=0.0)  # safe OOB load
""")

# ─────────────────────────────────────────────────────────────────────────────

lesson(4, "Your First Triton Kernel: Vector Scale")

explain("""
Let's write: out = x * scale
This is the simplest possible kernel.
""")

show("The kernel:")
print("""
  @triton.jit
  def scale_kernel(
      x_ptr,              # pointer to input
      out_ptr,            # pointer to output
      n_elements,         # total number of elements
      scale,              # scalar multiplier
      BLOCK: tl.constexpr,  # tile size (compile-time constant)
  ):
      # Step 1: Which block am I?
      pid     = tl.program_id(0)

      # Step 2: Which elements do I own?
      offsets = pid * BLOCK + tl.arange(0, BLOCK)
      mask    = offsets < n_elements   # boundary guard

      # Step 3: Load my elements from HBM
      x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

      # Step 4: Compute
      out = x * scale

      # Step 5: Write result back to HBM
      tl.store(out_ptr + offsets, out, mask=mask)

  # Launch:
  grid = lambda meta: (triton.cdiv(N, meta['BLOCK']),)
  scale_kernel[grid](x, out, N, scale, BLOCK=1024)
""")

explain("""
The grid lambda computes HOW MANY programs to launch.
  N = 1,000,000 elements, BLOCK = 1024
  Programs = ceil(1M / 1024) = 977 programs

Each program runs the kernel body independently on different elements.
All 977 programs run in PARALLEL on the GPU.
""")

# ─────────────────────────────────────────────────────────────────────────────

lesson(5, "Autotuning: Finding the Best Tile Size")

explain("""
Different hardware and problem sizes perform best with different tile sizes.
Triton has built-in autotuning: run the kernel with multiple configs, keep the fastest.

@triton.autotune decorates the kernel with a list of (config → result) pairs.
The first time the kernel runs with a new set of key args, it benchmarks all configs.

Config parameters:
  BLOCK_SIZE    → tile size (larger = more parallelism, but more shared memory)
  num_warps     → warps per block (usually 4 or 8)
  num_stages    → software pipeline depth (overlap memory load with compute)
""")

show("Example autotune config:")
print("""
  @triton.autotune(
      configs=[
          triton.Config({'BLOCK': 128},  num_warps=4),
          triton.Config({'BLOCK': 256},  num_warps=4),
          triton.Config({'BLOCK': 512},  num_warps=8),
          triton.Config({'BLOCK': 1024}, num_warps=8),
      ],
      key=['n_elements'],   # retune if n_elements changes
  )
  @triton.jit
  def my_kernel(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
      ...
""")

explain("""
Autotuning adds overhead the FIRST time (benchmarking each config).
Results are cached — subsequent calls use the best config immediately.
This is what 'max-autotune' mode in torch.compile does for every kernel.
""")

# ─────────────────────────────────────────────────────────────────────────────

lesson(6, "Tiled Matrix Multiplication (How Tensor Cores Work)")

explain("""
Matrix multiply: C[i,j] = sum_k A[i,k] * B[k,j]
Naive: read A[i,:] and B[:,j] for each output C[i,j] → O(M*N*K) HBM reads.

TILED MATMUL:
  Divide the matrices into tiles of size (BM, BK) and (BK, BN).
  Each Triton program computes ONE output tile C[i:i+BM, j:j+BN].
  It accumulates over K in BK-sized chunks.

  For each K chunk:
    Load A tile (BM × BK) from HBM → registers (or shared memory)
    Load B tile (BK × BN) from HBM → registers
    Accumulate: acc += A_tile @ B_tile  (using tl.dot → Tensor Cores)

  Total HBM reads for the whole matmul:
    A: each row of A is read N/BN times (once per output column tile)
    B: each col of B is read M/BM times (once per output row tile)
    Total: A*(N/BN) + B*(M/BM) instead of A*N + B*M for naive
    Reuse factor: BM = BN = 128 → 128× fewer reads!

TENSOR CORES:
  tl.dot(a, b) in Triton automatically uses NVIDIA Tensor Core units.
  Tensor Cores do D = A @ B + C in one hardware instruction for small tiles.
  For RTX PRO 2000: supports fp16 matmuls at 16 TFLOPS (2× the CUDA core peak).
  The 16 TFLOPS stat IS the tensor core peak.
""")

show("How tl.dot maps to Tensor Cores:")
print("""
  tl.dot(A_tile, B_tile)
  ↓ (Triton lowering)
  HMMA instruction (Warp Matrix Multiply Accumulate)
  ↓ (PTX instruction)
  wmma.mma.sync.m16n8k16.f16.f16.f32  ← Tensor Core instruction
  ↓ (hardware)
  16×8 result computed in 1 clock cycle by the Tensor Core unit
""")

# ─────────────────────────────────────────────────────────────────────────────

lesson(7, "Live Demo: Verify Your Triton Kernels")

if torch.cuda.is_available():
    explain("Running live kernel correctness check...")

    @triton.jit
    def scale_kernel(x_ptr, out_ptr, n_elements, scale, BLOCK: tl.constexpr):
        pid     = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask    = offsets < n_elements
        x       = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        tl.store(out_ptr + offsets, x * scale, mask=mask)

    N   = 1024 * 1024
    x   = torch.randn(N, device="cuda")
    out = torch.empty_like(x)
    scale = 3.14159

    grid = lambda meta: (triton.cdiv(N, meta["BLOCK"]),)
    scale_kernel[grid](x, out, N, scale, BLOCK=1024)

    torch_out = x * scale
    max_diff  = (out - torch_out).abs().max().item()

    show("scale_kernel(x, 3.14159):")
    print(f"      Max diff vs x * 3.14159: {max_diff:.2e}  {'✓ correct' if max_diff < 1e-4 else '✗ wrong'}")
    print(f"      x[:5]:   {x[:5].tolist()}")
    print(f"      out[:5]: {out[:5].tolist()}")
else:
    explain("CUDA not available — skipping live demo.")

# ─────────────────────────────────────────────────────────────────────────────

box("Triton Tour Complete")
print("""
  Key Takeaways:
    1. Triton operates on BLOCKS, not individual threads.
    2. tl.load/store + masking handles all boundary conditions.
    3. tl.dot uses Tensor Cores automatically.
    4. @triton.autotune finds the best tile size automatically.
    5. TorchInductor generates Triton kernels for you via torch.compile.
       Writing Triton manually lets you go further: custom ops,
       non-standard fusion patterns, or squeezing the last 10% of perf.

  Files to run:
    kernels/triton_basics.py   → vector add, softmax, fused bias+relu
    kernels/matmul.py          → tiled matrix multiplication
    kernels/fused_ops.py       → RMSNorm, SwiGLU, residual+norm
    kernels/flash_attention.py → tiled attention without O(T²) memory
""")
