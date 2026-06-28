"""
Triton Matrix Multiplication — The Core GPU Algorithm

Matrix multiplication is the most important GPU kernel. Every Linear layer,
every attention score computation, every weight update is a GEMM.

NAIVE MATMUL:
  C[i,j] = sum_k A[i,k] * B[k,j]
  Each element requires K multiply-adds.
  Total ops: M*N*K multiply-adds = 2*M*N*K FLOPs (multiply + add).
  Naive implementation loads A and B from HBM for every output element.
  Memory traffic: A (M*K) + B (K*N) per output element → O(M*N*K) reads.

BLOCKED (TILED) MATMUL:
  Divide A into (M/Tm)×(K/Tk) tiles,  B into (K/Tk)×(N/Tn) tiles.
  Each program computes one (Tm×Tn) output tile.
  Load A tile (Tm×Tk) and B tile (Tk×Tn) into SRAM (shared memory).
  Accumulate K/Tk times → Tm×Tn output.

  HBM traffic per output tile:
    A tile: Tm * K reads
    B tile: K * Tn reads
  But each tile is used to produce Tm*Tn outputs.
  Arithmetic intensity = (2*Tm*Tn*K) / (K*(Tm+Tn)) ≈ Tm*Tn/(Tm+Tn)
  With Tm=Tn=128: intensity = 128*128/256 = 64 FLOPs/byte
  RTX 2000 bandwidth: ~224 GB/s, peak: ~16 TFLOPS
  Roofline: need intensity > 16T/224G ≈ 71 FLOPs/byte to be compute-bound.

TENSOR CORES:
  Modern GPUs have Tensor Core units (NVIDIA) or Matrix Cores (AMD).
  They natively compute D = A@B + C for small tiles (e.g. 16×16×16) in
  one instruction. tl.dot() in Triton uses them automatically.

Run this file:
  python kernels/matmul.py
"""

import torch
import triton
import triton.language as tl
import time


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 32,  "GROUP_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 32,  "GROUP_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 32,  "GROUP_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32,  "GROUP_M": 8}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32,  "GROUP_M": 8}, num_stages=4, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,   # A: row stride, col stride
    stride_bk, stride_bn,   # B: row stride, col stride
    stride_cm, stride_cn,   # C: row stride, col stride
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,  # number of programs in a 'super-block' for L2 reuse
):
    """
    Tiled matrix multiply: C = A @ B
    A: (M, K)  B: (K, N)  C: (M, N)

    Each program computes a BLOCK_M × BLOCK_N tile of C.
    Grid: (ceil(M/BM) * ceil(N/BN),) — one program per output tile.

    GROUP_M controls 'grouped row ordering': programs in the same group
    share rows of A, improving L2 cache reuse for B.
    """
    # Which tile am I?
    pid   = tl.program_id(0)
    n_pid_m = tl.cdiv(M, BLOCK_M)
    n_pid_n = tl.cdiv(N, BLOCK_N)

    # Grouped row ordering: reorder programs to improve L2 reuse
    n_pid_in_group = GROUP_M * n_pid_n
    group_id       = pid // n_pid_in_group
    first_pid_m    = group_id * GROUP_M
    group_size_m   = min(n_pid_m - first_pid_m, GROUP_M)
    pid_m          = first_pid_m + (pid % group_size_m)
    pid_n          = (pid % n_pid_in_group) // group_size_m

    # Row and column indices for this tile in C
    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M))
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N))
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers into A and B for this tile's first K-block
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # Accumulate over K in BLOCK_K chunks
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Load tiles (with boundary masking)
        mask_a = (offs_m[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_K)
        mask_b = (offs_k[:, None] < K - k * BLOCK_K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)

        # tl.dot() uses Tensor Cores (fp16/bf16 input, fp32 accumulate)
        acc += tl.dot(a, b)

        # Advance pointers by BLOCK_K along K dimension
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Write accumulated tile to C
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    c = acc.to(tl.float16)   # cast down for storage
    c_ptrs = c_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
    tl.store(c_ptrs, c, mask=out_mask)


def triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """C = A @ B using our Triton kernel."""
    assert a.is_cuda and b.is_cuda
    assert a.ndim == 2 and b.ndim == 2
    assert a.shape[1] == b.shape[0]

    a = a.to(torch.float16).contiguous()
    b = b.to(torch.float16).contiguous()

    M, K = a.shape
    K_, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)

    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c


def benchmark(fn, *args, n_warmup=20, n_iter=200):
    for _ in range(n_warmup):
        fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iter * 1000


def tflops(M, N, K, ms):
    """Compute TFLOPS from matrix dimensions and latency."""
    return 2 * M * N * K / ms * 1e-9


def demo():
    print("\n── Triton vs torch.matmul benchmark ────────────────────────────")
    print(f"  {'Shape (M, N, K)':<25}  {'Triton':>10}  {'Torch':>10}  "
          f"{'Triton TFLOPS':>15}  {'Torch TFLOPS':>14}")
    print("  " + "─" * 80)

    shapes = [
        (512,  512,  512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (1024, 4096, 1024),   # GPT-3 inner FFN shape
    ]

    for M, N, K in shapes:
        a = torch.randn(M, K, device="cuda", dtype=torch.float16)
        b = torch.randn(K, N, device="cuda", dtype=torch.float16)

        t_triton = benchmark(triton_matmul, a, b)
        t_torch  = benchmark(torch.matmul, a, b)

        tf_triton = tflops(M, N, K, t_triton)
        tf_torch  = tflops(M, N, K, t_torch)

        print(f"  ({M:>4}, {N:>4}, {K:>4})           "
              f"  {t_triton:>8.3f}ms  {t_torch:>8.3f}ms  "
              f"  {tf_triton:>12.2f}T  {tf_torch:>12.2f}T")

    print()
    print("  RTX PRO 2000 peak (fp16): ~16 TFLOPS")
    print("  Note: torch.matmul uses cuBLAS which has deeply tuned GEMM kernels.")
    print("  Triton gets close — production models use cuBLAS for maximum perf.")
    print("  Triton shines when you need custom ops (e.g. FlashAttention).")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA required")
    else:
        demo()
        print("\nNext: kernels/fused_ops.py")
