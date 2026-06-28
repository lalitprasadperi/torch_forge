"""
Fused Kernels — Transformers Are All About Fusion

In a transformer, several operations naturally fuse because they share
the same iteration space (elementwise) or have arithmetic that simplifies.

KEY FUSION PATTERNS:
  1. Bias + Activation          (Linear layer output)
  2. RMSNorm                    (fuse: square, mean, rsqrt, scale)
  3. Attention bias              (fuse: causal mask + scale + softmax)
  4. Residual + LayerNorm       (fuse: x + residual, then norm)
  5. SwiGLU                     (fuse: x1*SiLU(x2) in one pass)

WHY FUSE?
  Every intermediate tensor that's written to HBM and read back
  costs bandwidth. For a 2048×4096 tensor at fp16:
    2048 * 4096 * 2 bytes = 16 MB
  On RTX 2000 at 224 GB/s: reading + writing = 32 MB = 0.14 ms
  Fusing 4 ops together saves 3 HBM round-trips = 0.42 ms per layer
  Over 32 transformer layers: 0.42 * 32 = 13 ms saved per step.

Run this file:
  python kernels/fused_ops.py
"""

import torch
import triton
import triton.language as tl
import time
import math


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 1: Fused RMSNorm
# out = x / RMS(x) * weight      where  RMS(x) = sqrt(mean(x^2) + eps)
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def rmsnorm_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    n_cols,
    eps: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """
    One program per row. Each program:
      1. Loads one row of x
      2. Computes RMS = sqrt(mean(x^2) + eps)
      3. Normalises and scales by weight
      4. Writes output

    All in ONE pass through HBM for x. Weight is tiny (fits in L1 cache).
    """
    row       = tl.program_id(0)
    row_start = row * n_cols
    cols      = tl.arange(0, BLOCK)
    mask      = cols < n_cols

    # Load row (float32 for precision)
    x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)

    # Compute RMS
    x_sq  = x * x
    mean_sq = tl.sum(x_sq, axis=0) / n_cols
    rsqrt   = tl.rsqrt(mean_sq + eps)

    # Normalise + scale
    x_norm = x * rsqrt
    w      = tl.load(weight_ptr + cols, mask=mask, other=1.0)
    out    = x_norm * w

    # Store (cast back to original dtype)
    tl.store(out_ptr + row_start + cols, out.to(tl.float16), mask=mask)


def triton_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Fused RMSNorm: O(N) memory, one HBM pass."""
    assert x.is_cuda
    x16 = x.to(torch.float16)
    rows, cols = x16.shape
    BLOCK = triton.next_power_of_2(cols)
    out   = torch.empty_like(x16)
    grid  = (rows,)
    rmsnorm_kernel[grid](x16, weight.to(torch.float16), out, cols, eps=eps, BLOCK=BLOCK)
    return out.to(x.dtype)


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 2: Fused SwiGLU
# out = (x1 * silu(x2))   where x = [x1 | x2] (split in half)
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def swiglu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    half_n,
    BLOCK: tl.constexpr,
):
    """
    x has shape (B, 2*d_ff). We split into x1, x2 (each shape B, d_ff).
    out = x1 * silu(x2)   where silu(x) = x * sigmoid(x)

    Each program handles BLOCK contiguous elements of the output.
    """
    pid     = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask    = offsets < n_elements

    # x1 is in first half, x2 in second half
    x1 = tl.load(x_ptr + offsets,         mask=mask, other=0.0)
    x2 = tl.load(x_ptr + offsets + half_n, mask=mask, other=0.0)

    # SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    silu_x2 = x2 * tl.sigmoid(x2)
    out      = x1 * silu_x2

    tl.store(out_ptr + offsets, out, mask=mask)


def triton_swiglu(x: torch.Tensor) -> torch.Tensor:
    """Fused SwiGLU: input has 2*d_ff cols, output has d_ff cols."""
    assert x.is_cuda and x.ndim == 2
    B, two_d = x.shape
    assert two_d % 2 == 0
    d    = two_d // 2
    out  = torch.empty(B, d, device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(B * d, meta["BLOCK"]),)

    swiglu_kernel[grid](x, out, B * d, B * d, BLOCK=512)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 3: Fused Residual + RMSNorm (pre-norm pattern)
# Common in transformer forward: out = rmsnorm(x + residual)
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def residual_rmsnorm_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    out_ptr,
    residual_out_ptr,   # also update the residual stream in-place
    n_cols,
    eps: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """
    Fuses: residual_new = x + residual
            out         = rmsnorm(residual_new) * weight

    TWO outputs from ONE pass:
      1. residual_out = x + residual (the new residual stream)
      2. out          = rmsnorm(residual_out)

    Llama / Mistral use this pattern in every transformer block.
    """
    row       = tl.program_id(0)
    row_start = row * n_cols
    cols      = tl.arange(0, BLOCK)
    mask      = cols < n_cols

    x   = tl.load(x_ptr        + row_start + cols, mask=mask, other=0.0).to(tl.float32)
    res = tl.load(residual_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)

    # Residual add
    res_new = x + res

    # RMSNorm
    rms   = tl.rsqrt(tl.sum(res_new * res_new, 0) / n_cols + eps)
    w     = tl.load(weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    normed = res_new * rms * w

    # Store both outputs
    tl.store(residual_out_ptr + row_start + cols, res_new.to(tl.float16), mask=mask)
    tl.store(out_ptr          + row_start + cols, normed.to(tl.float16), mask=mask)


def triton_residual_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
):
    assert x.is_cuda
    rows, cols = x.shape
    BLOCK = triton.next_power_of_2(cols)
    out         = torch.empty_like(x, dtype=torch.float16)
    residual_new = torch.empty_like(x, dtype=torch.float16)
    grid = (rows,)
    residual_rmsnorm_kernel[grid](
        x.to(torch.float16), residual.to(torch.float16),
        weight.to(torch.float16),
        out, residual_new,
        cols, eps=eps, BLOCK=BLOCK,
    )
    return out, residual_new


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark
# ─────────────────────────────────────────────────────────────────────────────

def bmark(fn, *args, n=100):
    for _ in range(20):
        fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000


def demo():
    B, d = 2048, 4096   # typical transformer hidden states

    # ── RMSNorm ───────────────────────────────────────────────────────────────
    print("\n── Fused RMSNorm ────────────────────────────────────────────────")
    x = torch.randn(B, d, device="cuda")
    w = torch.ones(d, device="cuda")

    def torch_rmsnorm(x, w, eps=1e-6):
        return (x / x.pow(2).mean(-1, keepdim=True).add(eps).rsqrt()) * w

    out_triton = triton_rmsnorm(x, w)
    out_torch  = torch_rmsnorm(x, w)
    print(f"  Max diff: {(out_triton.float() - out_torch.float()).abs().max():.2e}")

    t_triton = bmark(triton_rmsnorm, x, w)
    t_torch  = bmark(torch_rmsnorm, x, w)
    print(f"  Triton: {t_triton:.3f} ms  |  Torch (unfused): {t_torch:.3f} ms")

    # ── SwiGLU ────────────────────────────────────────────────────────────────
    print("\n── Fused SwiGLU ─────────────────────────────────────────────────")
    d_ff = d * 8 // 3   # SwiGLU hidden (rounds to ~10922)
    d_ff = (d_ff + 255) // 256 * 256   # round to multiple of 256
    x_swi = torch.randn(B, d_ff * 2, device="cuda", dtype=torch.float16)

    def torch_swiglu(x):
        x1, x2 = x.chunk(2, dim=-1)
        return x1 * torch.nn.functional.silu(x2)

    out_triton_swi = triton_swiglu(x_swi)
    out_torch_swi  = torch_swiglu(x_swi)
    print(f"  Max diff: {(out_triton_swi - out_torch_swi).abs().max():.2e}")

    t_triton_swi = bmark(triton_swiglu, x_swi)
    t_torch_swi  = bmark(torch_swiglu, x_swi)
    print(f"  Triton: {t_triton_swi:.3f} ms  |  Torch: {t_torch_swi:.3f} ms")

    # ── Residual + RMSNorm ────────────────────────────────────────────────────
    print("\n── Fused Residual + RMSNorm (Llama-style) ───────────────────────")
    x2  = torch.randn(B, d, device="cuda")
    res = torch.randn(B, d, device="cuda")
    w2  = torch.ones(d, device="cuda")

    def torch_res_rms(x, r, w, eps=1e-6):
        r_new = x + r
        return torch_rmsnorm(r_new, w, eps), r_new

    out_t, res_t = triton_residual_rmsnorm(x2, res, w2)
    out_e, res_e = torch_res_rms(x2, res, w2)
    print(f"  Max diff (norm out): {(out_t.float() - out_e.float()).abs().max():.2e}")
    print(f"  Max diff (residual): {(res_t.float() - res_e.float()).abs().max():.2e}")

    t_fused  = bmark(triton_residual_rmsnorm, x2, res, w2)
    t_unfused = bmark(torch_res_rms, x2, res, w2)
    print(f"  Triton (1 kernel):      {t_fused:.3f} ms")
    print(f"  Torch  (2 kernels+HBM): {t_unfused:.3f} ms")

    # Bandwidth saved
    B_elem = B * d
    bw_saved = 2 * B_elem * 2 / 1e9   # residual_new: one write+read avoided
    print(f"  HBM traffic saved per call: ~{bw_saved:.1f} GB")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA required")
    else:
        demo()
        print("\nNext: kernels/flash_attention.py")
