"""
Flash Attention in Triton — Tiled Attention for O(T) Memory

THE STANDARD ATTENTION PROBLEM:
  Attention(Q, K, V) = softmax(QK^T / sqrt(d)) @ V

  Standard implementation:
    1. S = Q @ K^T          shape (B, H, T, T) — the BIG matrix
    2. P = softmax(S)        shape (B, H, T, T)
    3. O = P @ V            shape (B, H, T, d)

  Memory: the (T, T) attention matrix is O(T^2).
  For T=8192 (8K context), H=32 heads: 8192^2 * 32 * 2B = 17 GB.
  This is why long-context models need special tricks.

FLASH ATTENTION:
  Key insight: We don't need to materialise the full (T, T) matrix.
  We can compute softmax incrementally using the 'online softmax' trick.

  Algorithm (simplified):
    For each query block Qi (rows BLOCK_Q at a time):
      For each key/value block Kj, Vj (cols BLOCK_KV at a time):
        Compute Sij = Qi @ Kj^T
        Update running max  m = max(m, row_max(Sij))
        Update running sum  l = l * exp(old_m - m) + sum(exp(Sij - m))
        Update output       O = O * exp(old_m - m) + exp(Sij - m) @ Vj
      Normalise: O = O / l

  Memory: only the BLOCK_Q and BLOCK_KV tiles in SRAM — O(T) total.
  The (T, T) matrix never exists in memory.

  FlashAttention-2 (Tri Dao, 2023): further optimises by reducing non-matrix ops,
  better causal masking, and improved parallelism across query blocks.

This file implements a SIMPLIFIED version for understanding.
Production code: use torch.nn.functional.scaled_dot_product_attention (SDPA)
which dispatches to flash_attn or cudnn flash attention automatically.

Run this file:
  python kernels/flash_attention.py
"""

import torch
import triton
import triton.language as tl
import math
import time


@triton.jit
def flash_attention_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    q_stride_b, q_stride_h, q_stride_t, q_stride_d,
    k_stride_b, k_stride_h, k_stride_t, k_stride_d,
    v_stride_b, v_stride_h, v_stride_t, v_stride_d,
    o_stride_b, o_stride_h, o_stride_t, o_stride_d,
    T,          # sequence length
    d,          # head dimension
    scale,      # 1 / sqrt(d)
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    D: tl.constexpr,            # head dim (constexpr for shared mem alloc)
):
    """
    Each program handles:  one (batch, head, query_block)
    Grid: (B * H * ceil(T / BLOCK_Q),)
    """
    pid    = tl.program_id(0)
    n_q_blocks = tl.cdiv(T, BLOCK_Q)

    # Which batch, head, query block?
    batch_head = pid // n_q_blocks
    q_block    = pid  % n_q_blocks

    b = batch_head // 1   # single head (simplified: 1 head)
    h = 0

    q_start = q_block * BLOCK_Q
    q_offs  = q_start + tl.arange(0, BLOCK_Q)
    d_offs  = tl.arange(0, D)

    # Load query tile: (BLOCK_Q, D)
    q_mask = (q_offs[:, None] < T) & (d_offs[None, :] < d)
    q_base = b * q_stride_b + h * q_stride_h
    Q = tl.load(
        q_ptr + q_base + q_offs[:, None] * q_stride_t + d_offs[None, :] * q_stride_d,
        mask=q_mask, other=0.0,
    )

    # Running statistics for online softmax
    m_i = tl.full((BLOCK_Q,), float("-inf"), dtype=tl.float32)   # running max
    l_i = tl.zeros((BLOCK_Q,),              dtype=tl.float32)    # running sum
    O   = tl.zeros((BLOCK_Q, D),            dtype=tl.float32)    # output accumulator

    # Iterate over key/value blocks
    n_kv_blocks = tl.cdiv(T, BLOCK_KV)
    for j in range(n_kv_blocks):
        kv_start = j * BLOCK_KV
        kv_offs  = kv_start + tl.arange(0, BLOCK_KV)
        kv_mask  = (kv_offs[:, None] < T) & (d_offs[None, :] < d)

        kv_base = b * k_stride_b + h * k_stride_h
        K = tl.load(
            k_ptr + kv_base + kv_offs[:, None] * k_stride_t + d_offs[None, :] * k_stride_d,
            mask=kv_mask, other=0.0,
        )
        V = tl.load(
            v_ptr + kv_base + kv_offs[:, None] * v_stride_t + d_offs[None, :] * v_stride_d,
            mask=kv_mask, other=0.0,
        )

        # Attention scores: S = Q @ K^T * scale  (BLOCK_Q × BLOCK_KV)
        S = tl.dot(Q, tl.trans(K)) * scale

        # Causal mask: q position j can't attend to kv position > q position
        causal_mask = q_offs[:, None] >= kv_offs[None, :]
        S = tl.where(causal_mask, S, float("-inf"))

        # Online softmax update
        m_j = tl.max(S, axis=1)             # max of this block
        m_new = tl.maximum(m_i, m_j)        # new running max

        # Correction factor for previous accumulator
        alpha = tl.exp(m_i - m_new)         # rescale old O
        beta  = tl.exp(S - m_new[:, None])  # softmax numerators

        # Update output and normaliser
        O   = O * alpha[:, None] + tl.dot(beta.to(tl.float16), V)
        l_i = l_i * alpha + tl.sum(beta, axis=1)
        m_i = m_new

    # Final normalisation
    O = O / l_i[:, None]

    # Write output
    out_base = b * o_stride_b + h * o_stride_h
    out_mask = (q_offs[:, None] < T) & (d_offs[None, :] < d)
    tl.store(
        out_ptr + out_base + q_offs[:, None] * o_stride_t + d_offs[None, :] * o_stride_d,
        O.to(tl.float16), mask=out_mask,
    )


def triton_flash_attention(Q, K, V, causal=True):
    """
    Q, K, V: (B, H, T, d) in fp16.
    Returns: (B, H, T, d).

    Simplified: single-head (H=1).
    """
    assert Q.is_cuda and Q.dtype == torch.float16
    B, H, T, d = Q.shape
    assert H == 1, "Simplified version: single head only"

    scale = 1.0 / math.sqrt(d)
    D     = triton.next_power_of_2(d)
    BLOCK_Q  = 32
    BLOCK_KV = 32

    out  = torch.empty_like(Q)
    grid = (B * H * triton.cdiv(T, BLOCK_Q),)

    flash_attention_kernel[grid](
        Q, K, V, out,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        T, d, scale,
        BLOCK_Q=BLOCK_Q, BLOCK_KV=BLOCK_KV, D=D,
    )
    return out


def standard_attention(Q, K, V):
    """Standard attention: materialises (T, T) matrix."""
    d     = Q.shape[-1]
    scale = 1.0 / math.sqrt(d)
    S     = (Q @ K.transpose(-2, -1)) * scale
    T     = S.shape[-1]
    causal_mask = torch.tril(torch.ones(T, T, device=Q.device)).bool()
    S = S.masked_fill(~causal_mask, float("-inf"))
    P   = torch.softmax(S.float(), dim=-1).to(Q.dtype)
    return P @ V


def demo():
    print("\n── Flash Attention vs Standard ──────────────────────────────────")
    B, T, d = 1, 512, 64   # single head, 512 tokens, dim 64
    Q = torch.randn(B, 1, T, d, device="cuda", dtype=torch.float16) * 0.1
    K = torch.randn(B, 1, T, d, device="cuda", dtype=torch.float16) * 0.1
    V = torch.randn(B, 1, T, d, device="cuda", dtype=torch.float16) * 0.1

    out_flash  = triton_flash_attention(Q, K, V)
    out_std    = standard_attention(Q, K, V)

    print(f"  Max diff (Flash vs Standard): {(out_flash.float() - out_std.float()).abs().max():.2e}")
    print(f"  Output shape: {out_flash.shape}")

    print("\n── Memory comparison ────────────────────────────────────────────")
    for seq_len in [512, 1024, 2048, 4096]:
        attn_mat_mb = seq_len * seq_len * 2 / 1e6   # fp16
        flash_mb    = 2 * 32 * 32 * 2 / 1e6         # only SRAM tiles
        print(f"  T={seq_len:>5}:  standard={attn_mat_mb:>8.1f} MB  "
              f"flash≈{flash_mb:.2f} MB  ratio={attn_mat_mb/flash_mb:.0f}×")

    print("\n── PyTorch SDPA (production FlashAttn) ──────────────────────────")
    B2, H, T2, d2 = 4, 8, 1024, 64
    Q2 = torch.randn(B2, H, T2, d2, device="cuda", dtype=torch.float16)
    K2 = torch.randn_like(Q2)
    V2 = torch.randn_like(Q2)

    def run_sdpa():
        return torch.nn.functional.scaled_dot_product_attention(
            Q2, K2, V2, is_causal=True)

    def run_manual():
        return standard_attention(Q2, K2, V2)

    n_warmup, n_iter = 10, 100
    for _ in range(n_warmup):
        run_sdpa(); run_manual()
    torch.cuda.synchronize()

    import time
    t0 = time.perf_counter()
    for _ in range(n_iter):
        run_sdpa()
    torch.cuda.synchronize()
    t_sdpa = (time.perf_counter() - t0) / n_iter * 1000

    t0 = time.perf_counter()
    for _ in range(n_iter):
        run_manual()
    torch.cuda.synchronize()
    t_manual = (time.perf_counter() - t0) / n_iter * 1000

    print(f"  SDPA (Flash):   {t_sdpa:.3f} ms")
    print(f"  Standard attn:  {t_manual:.3f} ms")
    print(f"  Speedup: {t_manual/t_sdpa:.2f}×")
    print()
    print("  torch.nn.functional.scaled_dot_product_attention() auto-dispatches")
    print("  to Flash Attention (CUDA backend) when available.")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA required")
    else:
        demo()
        print("\nNext: benchmarks/compile_speedup.py")
