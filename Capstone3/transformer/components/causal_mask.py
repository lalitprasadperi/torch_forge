"""
Causal Masking — Make Attention Autoregressive

WHY CAUSAL MASKING?
────────────────────
In a language model, token at position i must only attend to positions
≤ i (itself and earlier tokens). It must NOT see future tokens — that
would be cheating. The model should predict token i using only tokens 0..i-1.

This is called autoregressive or causal attention.

HOW IT WORKS:
  Before applying softmax, we add -inf to all "future" positions:

  scores (pre-mask):              after masking:
  [[s00  s01  s02  s03]          [[s00  -inf -inf -inf]
   [s10  s11  s12  s13]    →      [s10  s11  -inf -inf]
   [s20  s21  s22  s23]           [s20  s21  s22  -inf]
   [s30  s31  s32  s33]]          [s30  s31  s32  s33]]

  After softmax(-inf) = 0, so future positions contribute 0 to the output.

  The mask is a lower-triangular matrix of True/False values.

EFFICIENCY:
  The mask is computed once and cached (register_buffer in the model).
  For batch inference we slice [:T, :T] to get the right size.
"""

import torch


def causal_mask(T: int, device: torch.device = None) -> torch.Tensor:
    """
    Return boolean lower-triangular mask of shape (T, T).
    True = keep, False = mask out (will become -inf).

    Example for T=4:
    [[True,  False, False, False],
     [True,  True,  False, False],
     [True,  True,  True,  False],
     [True,  True,  True,  True ]]
    """
    return torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))


def apply_causal_mask(
    scores: torch.Tensor,       # (B, H, T_q, T_k)
    mask:   torch.Tensor = None # (T_q, T_k) or (B, 1, T_q, T_k)
) -> torch.Tensor:
    """
    Apply causal mask to attention scores: set future positions to -inf.

    -inf → softmax → 0: future tokens contribute nothing to the output.
    We use -1e9 instead of -inf to avoid NaN when all positions are masked
    (e.g. during padding handling).
    """
    T_q, T_k = scores.shape[-2], scores.shape[-1]
    if mask is None:
        mask = causal_mask(T_q, device=scores.device)   # (T_q, T_k) but usually T_q==T_k
        # During inference with KV cache, T_q=1 but T_k=past+1
        if T_q != T_k:
            # new query can see all past keys — no masking needed for T_q=1
            mask = torch.ones(T_q, T_k, dtype=torch.bool, device=scores.device)
    return scores.masked_fill(~mask, -1e9)


def make_padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """
    Boolean mask for variable-length sequences: True = real token, False = padding.

    lengths: (B,) tensor of actual sequence lengths per sample
    Returns: (B, 1, 1, max_len) for broadcasting over (B, H, T_q, T_k)
    """
    B = lengths.size(0)
    ids = torch.arange(max_len, device=lengths.device).unsqueeze(0)  # (1, max_len)
    mask = ids < lengths.unsqueeze(1)                                  # (B, max_len)
    return mask.unsqueeze(1).unsqueeze(1)                              # (B, 1, 1, max_len)
