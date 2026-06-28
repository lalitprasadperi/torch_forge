"""
Triton Paged Attention Kernel — Fast Decode Attention

WHY A CUSTOM KERNEL?
─────────────────────
The PyTorch reference in model/paged_attention.py has a Python loop over
batch items and materialises gathered K,V tensors. On GPU that's:
  1. Python overhead per sequence
  2. Extra HBM writes for the gathered tensors (then read again for attention)

A Triton kernel does the gather + attention in ONE fused pass, per head,
reading each KV block exactly once into SRAM.

THE ALGORITHM (online softmax):
  For each (batch, head) pair in parallel:
    m = -inf  (running max for numerical stability)
    l = 0     (running sum of exp)
    o = 0     (output accumulator)

    For each block in block_table[batch]:
      Load K block: (block_size, d_head)
      Load V block: (block_size, d_head)
      For each token t in block:
        score = dot(q, K[t]) * scale
        m_new = max(m, score)
        alpha = exp(m - m_new)         # rescale old accumulator
        o = o * alpha + exp(score - m_new) * V[t]
        l = l * alpha + exp(score - m_new)
        m = m_new

    out[batch, head] = o / l           # final normalisation

This is the ONLINE SOFTMAX technique — the same idea as Flash Attention,
applied to the gather case. No materialisation of gathered K,V needed.

Reference: vLLM paged_attention_v1 kernel (Apache 2.0)
"""

import math
import torch
import triton
import triton.language as tl


@triton.jit
def paged_attn_decode_kernel(
    # Output
    out_ptr,         # (B, H, D)
    # Inputs
    q_ptr,           # (B, H, D)
    k_cache_ptr,     # (num_blocks, block_size, H, D)
    v_cache_ptr,     # (num_blocks, block_size, H, D)
    # Block tables
    block_tables_ptr, # (B, max_blocks) int32
    seq_lens_ptr,     # (B,) int32
    # Dimensions
    scale,
    B, H, D,
    block_size: tl.constexpr,  # constexpr: unrolls the inner tok loop
    max_blocks,
    # Tile sizes
    BLOCK_D: tl.constexpr,   # must be >= D, power of 2
):
    """
    Each program handles ONE (batch, head) pair.
    Iterates over all KV blocks for that sequence using online softmax.

    Triton does not support `break` inside loops, so we loop to n_blocks
    (the actual number of blocks for this sequence) instead of max_blocks.
    The inner loop over tokens is unrolled because block_size is constexpr.
    """
    batch_idx = tl.program_id(0)
    head_idx  = tl.program_id(1)

    seq_len  = tl.load(seq_lens_ptr + batch_idx)
    n_blocks = (seq_len + block_size - 1) // block_size
    d_offs   = tl.arange(0, BLOCK_D)
    d_mask   = d_offs < D

    # Load query: (D,)
    q = tl.load(
        q_ptr + batch_idx * H * D + head_idx * D + d_offs,
        mask=d_mask, other=0.0
    ).to(tl.float32)

    # Online softmax state
    m   = float("-inf")
    l   = 0.0
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    # Dynamic loop bound: iterate only over allocated blocks (no break needed)
    for blk in range(n_blocks):
        block_id = tl.load(block_tables_ptr + batch_idx * max_blocks + blk)

        # Base pointer for this block's K and V
        k_block_base = k_cache_ptr + block_id * block_size * H * D + head_idx * D
        v_block_base = v_cache_ptr + block_id * block_size * H * D + head_idx * D

        for tok in range(block_size):  # unrolled (constexpr)
            global_pos = blk * block_size + tok
            valid      = global_pos < seq_len

            k_tok = tl.load(
                k_block_base + tok * H * D + d_offs,
                mask=d_mask & valid, other=0.0
            ).to(tl.float32)

            # Attention score: dot(q, k) * scale
            score = tl.sum(q * k_tok, axis=0) * scale
            score = tl.where(valid, score, float("-inf"))

            # Online softmax update
            m_new = tl.maximum(m, score)
            alpha = tl.exp(m - m_new)
            beta  = tl.exp(score - m_new)

            v_tok = tl.load(
                v_block_base + tok * H * D + d_offs,
                mask=d_mask & valid, other=0.0
            ).to(tl.float32)

            acc = acc * alpha + beta * v_tok
            l   = l   * alpha + beta
            m   = m_new

    # Normalise and write output
    out = (acc / tl.where(l > 0, l, 1.0)).to(tl.float16)
    tl.store(
        out_ptr + batch_idx * H * D + head_idx * D + d_offs,
        out, mask=d_mask
    )


def triton_paged_attention_decode(
    query:        torch.Tensor,   # (B, H, D) fp16
    k_cache:      torch.Tensor,   # (num_blocks, block_size, H, D) fp16
    v_cache:      torch.Tensor,   # (num_blocks, block_size, H, D) fp16
    block_tables: torch.Tensor,   # (B, max_blocks) int32
    seq_lens:     torch.Tensor,   # (B,) int32
    scale:        float,
) -> torch.Tensor:
    B, H, D     = query.shape
    max_blocks  = block_tables.shape[1]
    block_size  = k_cache.shape[1]
    BLOCK_D     = triton.next_power_of_2(D)

    out  = torch.empty_like(query)
    grid = (B, H)

    paged_attn_decode_kernel[grid](
        out, query, k_cache, v_cache,
        block_tables, seq_lens,
        scale,
        B, H, D,
        block_size, max_blocks,
        BLOCK_D=BLOCK_D,
    )
    return out


# ── Correctness check ──────────────────────────────────────────────────────────

def test_vs_reference():
    """Compare Triton kernel against PyTorch reference."""
    from model.paged_attention import paged_attention_decode

    torch.manual_seed(42)
    B, H, D      = 4, 8, 64
    block_size   = 16
    num_blocks   = 64
    max_seq_len  = 128
    scale        = 1.0 / math.sqrt(D)

    # Fill cache with random KV
    k_cache = torch.randn(num_blocks, block_size, H, D, device="cuda", dtype=torch.float16)
    v_cache = torch.randn(num_blocks, block_size, H, D, device="cuda", dtype=torch.float16)

    # Combine for PyTorch reference shape: (num_blocks, 2, block_size, H, D)
    kv_cache_ref = torch.stack([k_cache, v_cache], dim=1)

    # Random block tables and seq lens
    max_blocks_needed = max_seq_len // block_size
    block_tables = torch.randint(0, num_blocks, (B, max_blocks_needed), dtype=torch.int32, device="cuda")
    seq_lens     = torch.randint(16, max_seq_len + 1, (B,), dtype=torch.int32, device="cuda")

    query = torch.randn(B, H, D, device="cuda", dtype=torch.float16)

    # Reference
    ref_out = paged_attention_decode(query, kv_cache_ref, block_tables, seq_lens, scale)

    # Triton
    triton_out = triton_paged_attention_decode(query, k_cache, v_cache, block_tables, seq_lens, scale)

    max_diff = (ref_out.float() - triton_out.float()).abs().max().item()
    print(f"  Max diff (Triton vs PyTorch reference): {max_diff:.3e}")
    assert max_diff < 0.05, f"Too large: {max_diff}"
    print("  ✓ Triton paged attention matches reference")


if __name__ == "__main__":
    if torch.cuda.is_available():
        test_vs_reference()
    else:
        print("CUDA required")
