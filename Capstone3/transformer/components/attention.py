"""
Scaled Dot-Product Attention — The Core Operation

This is THE most important function in the transformer. Every piece of the
architecture exists to feed into this one operation.

THE INTUITION:
──────────────
Given a set of queries Q, keys K, and values V, attention answers:
  "For each query, how much should I attend to each key, and what
   weighted combination of values do I return?"

Think of it like a soft dictionary lookup:
  - Hard lookup:  dict[key] → exact match only
  - Soft lookup:  find the keys most similar to the query, return a weighted
                  average of their values, weighted by similarity score

In a self-attention layer:
  Q = xW_Q   (what am I looking for?)
  K = xW_K   (what do I have to offer?)
  V = xW_V   (what information do I carry?)

FORMULA:
  Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) V

Step by step:
  1. Q K^T          — dot product similarity between every query and every key
                      shape: (B, H, T_q, T_k)
  2. / sqrt(d_k)    — scale down to prevent softmax saturation
  3. + mask         — set future positions to -inf (causal masking)
  4. softmax(...)   — normalise to get attention weights summing to 1
  5. @ V            — weighted sum of values

WHY sqrt(d_k) SCALING?
  Q and K are initialised ~N(0,1). Their dot product grows as:
  E[q·k] = 0,  Var[q·k] = d_k
  → std of dot products = sqrt(d_k)

  Without scaling, large d_k pushes softmax into saturation (one weight
  near 1, all others near 0 → nearly discrete, tiny gradients).
  Dividing by sqrt(d_k) brings variance back to 1.

COMPLEXITY:
  Time:   O(T² × d_k) — quadratic in sequence length T
  Memory: O(T²)        — the attention matrix
  This is the bottleneck that FlashAttention was designed to address.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .causal_mask import apply_causal_mask


def scaled_dot_product_attention(
    Q:       torch.Tensor,          # (B, H, T_q, d_k)
    K:       torch.Tensor,          # (B, H, T_k, d_k)
    V:       torch.Tensor,          # (B, H, T_k, d_v)
    mask:    torch.Tensor = None,   # (T_q, T_k) or (B, 1, T_q, T_k)
    dropout: float = 0.0,
    training: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pure PyTorch scaled dot-product attention.

    Returns:
        output:  (B, H, T_q, d_v)   weighted combination of values
        weights: (B, H, T_q, T_k)   attention weight matrix (for visualisation)
    """
    d_k = Q.size(-1)

    # Step 1: similarity scores — (B, H, T_q, T_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # Step 2: causal mask — future positions → -inf → softmax → 0
    if mask is not None:
        scores = apply_causal_mask(scores, mask)
    else:
        # Apply causal mask by default if T_q == T_k (training)
        if Q.size(-2) == K.size(-2):
            scores = apply_causal_mask(scores)

    # Step 3: softmax over the key dimension
    weights = F.softmax(scores, dim=-1)   # (B, H, T_q, T_k)

    # Step 4: attention dropout (regularises which keys get attended to)
    if dropout > 0.0 and training:
        weights = F.dropout(weights, p=dropout, training=True)

    # Step 5: weighted sum of values
    output = torch.matmul(weights, V)     # (B, H, T_q, d_v)

    return output, weights


def flash_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    causal: bool = True,
    dropout: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    """
    Use PyTorch's built-in scaled_dot_product_attention which dispatches to
    FlashAttention on CUDA when available (PyTorch >= 2.0).

    This is the production path — mathematically identical to our manual
    implementation but 2-4x faster and uses O(N) memory instead of O(N²).

    See tours/flash_tour.py for a detailed comparison and memory analysis.
    """
    return F.scaled_dot_product_attention(
        Q, K, V,
        attn_mask  = None,
        dropout_p  = dropout if training else 0.0,
        is_causal  = causal,
    )
