"""
Embedding — Token Embeddings + Positional Encodings

The first layer of every transformer: convert token indices into
dense vectors, then add position information.

WHY EMBEDDINGS?
───────────────
A transformer operates on continuous vectors, not discrete tokens.
An embedding table is a learned lookup: token_id → d_model-dim vector.
Think of it as a trainable one-hot encoder: embed(id) = W[id, :] where
W is shape (vocab_size, d_model). This is exactly nn.Embedding.

WHY POSITIONAL ENCODING?
─────────────────────────
Self-attention is permutation-invariant: attending to tokens [A,B,C] gives
the same result as [C,A,B]. Transformers have NO built-in sense of order.
We inject position information by adding a position-dependent vector to
each token embedding:

    input_i = token_embed(id_i) + position_embed(i)

Two variants implemented here:
  1. SinusoidalPE  — fixed, mathematical (original "Attention is All You Need")
  2. LearnedPE     — trainable parameters (GPT-2 style)
  3. RoPE          — see transformer/modern/rope.py (applied inside attention)

SINUSOIDAL FORMULA:
  PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
  PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

  Each dimension oscillates at a different frequency. Low dimensions
  change rapidly with position (high frequency). High dimensions change
  slowly (low frequency). Together they uniquely encode any position.

  Key property: PE(pos+k) can be expressed as a linear function of PE(pos)
  for any offset k — the model can learn relative positions from absolute ones.
"""

import math
import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """
    Learnable token embedding table.

    Maps discrete token IDs → continuous d_model-dimensional vectors.
    Initialised with N(0, 1/sqrt(d_model)) so that the initial embedding
    norm ≈ 1, matching the scale of positional encodings.
    """

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        # Scale initialisation: variance 1/d_model keeps output norm stable
        nn.init.normal_(self.embed.weight, mean=0.0, std=d_model ** -0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T) integer token ids
        # out: (B, T, d_model) float embeddings
        return self.embed(x)


class SinusoidalPE(nn.Module):
    """
    Fixed sinusoidal positional encoding (Vaswani et al. 2017).

    Stored as a buffer (not a parameter): no gradients, moves with model.to(device),
    saved in state_dict for exact reproducibility.

    Shape: (1, max_len, d_model) — broadcast over batch dimension.
    """

    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Compute encoding matrix once
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()           # (max_len, 1)
        # div_term[i] = 1 / 10000^(2i/d_model)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)   # even dims: sin
        pe[:, 1::2] = torch.cos(pos * div)   # odd  dims: cos

        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        x = x + self.pe[:, :x.size(1)]   # broadcast: adds position to each token
        return self.dropout(x)


class LearnedPE(nn.Module):
    """
    Learned positional embedding (GPT-2 style).

    A simple nn.Embedding over position indices [0, 1, ..., max_len-1].
    The model learns which position vectors are most useful during training.
    Slightly more expressive than sinusoidal but cannot generalise
    to sequence lengths longer than max_len.
    """

    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.dropout   = nn.Dropout(dropout)
        nn.init.normal_(self.pos_embed.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        T   = x.size(1)
        pos = torch.arange(T, device=x.device).unsqueeze(0)  # (1, T)
        return self.dropout(x + self.pos_embed(pos))
