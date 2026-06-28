"""
GPT — Full Decoder-Only Transformer

Assembles all components into a complete language model:

    Input: token ids (B, T)
        ↓
    Token Embedding + Positional Encoding → (B, T, d_model)
        ↓
    N × DecoderBlock (attn + ffn, pre-norm, residual)
        ↓
    Final LayerNorm
        ↓
    Language Model Head: Linear(d_model, vocab_size) → (B, T, vocab_size)
        ↓
    Output: logits (B, T, vocab_size)

LOSS:
  CrossEntropy(logits[:, :-1], tokens[:, 1:])
  We shift by 1: the model predicts the NEXT token at each position.
  logits[i] should predict tokens[i+1].

WEIGHT TYING:
  The language model head (W_lm) and token embedding (W_embed) share weights.
  This is called "weight tying" or "input-output embedding tying".
  Intuition: the embed matrix maps tokens to vectors; W_lm maps vectors back
  to token logits. They do inverse operations — it makes sense to share weights.
  Also reduces parameters by vocab_size × d_model (significant for large vocab).

GPT-2 SPECIFIC INITIALISATIONS:
  • All weights: N(0, 0.02)
  • Residual projection weights: N(0, 0.02 / sqrt(2 × n_layers))
    The 1/sqrt(2n) scaling compensates for variance growth from stacking
    n residual connections that each add variance.
"""

import math
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from .decoder_block import DecoderBlock
from ..components.embedding import TokenEmbedding, SinusoidalPE, LearnedPE
from ..modern.rope import RotaryEmbedding
from ..modern.rmsnorm import RMSNorm


@dataclass
class GPTConfig:
    # Vocabulary + sequence
    vocab_size:  int   = 50257   # GPT-2 tokenizer
    max_len:     int   = 1024    # max context length

    # Architecture
    d_model:     int   = 768
    n_layers:    int   = 12
    n_heads:     int   = 12
    d_ff:        int   = None    # None → 4 × d_model

    # Regularisation
    dropout:     float = 0.1
    bias:        bool  = False   # False = GPT-2 style (no bias)

    # Modern variants
    use_flash:   bool  = True    # torch SDPA → FlashAttention on CUDA
    use_rope:    bool  = False   # Rotary PE (replaces sinusoidal/learned PE)
    use_rmsnorm: bool  = False   # RMSNorm instead of LayerNorm
    use_swiglu:  bool  = False   # SwiGLU FFN instead of GELU FFN
    pos_enc:     str   = "learned"  # "sinusoidal" | "learned" | "rope"

    def __post_init__(self):
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model
        if self.pos_enc == "rope":
            self.use_rope = True


# Convenient named presets
def gpt_nano() -> GPTConfig:
    """Tiny GPT for quick experiments (~10M params)."""
    return GPTConfig(vocab_size=65, max_len=256, d_model=384, n_layers=6,
                     n_heads=6, d_ff=1536, dropout=0.1, bias=True, pos_enc="learned")

def gpt_small() -> GPTConfig:
    """Small GPT-2 equivalent (~117M params)."""
    return GPTConfig(vocab_size=50257, max_len=1024, d_model=768, n_layers=12,
                     n_heads=12, dropout=0.1, bias=False)

def gpt_modern() -> GPTConfig:
    """Modern GPT (LLaMA-style): RoPE + RMSNorm + SwiGLU (~120M params)."""
    return GPTConfig(vocab_size=50257, max_len=2048, d_model=768, n_layers=12,
                     n_heads=12, dropout=0.0, bias=False,
                     use_rope=True, use_rmsnorm=True, use_swiglu=True)


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # ── Embeddings ─────────────────────────────────────────────────────────
        self.token_embed = TokenEmbedding(config.vocab_size, config.d_model)

        if config.pos_enc == "sinusoidal":
            self.pos_enc = SinusoidalPE(config.d_model, config.max_len, config.dropout)
        elif config.pos_enc == "learned":
            self.pos_enc = LearnedPE(config.d_model, config.max_len, config.dropout)
        else:  # rope — no additive PE; handled inside blocks
            self.pos_enc = nn.Dropout(config.dropout)

        if config.use_rope:
            self.rope = RotaryEmbedding(config.d_model // config.n_heads)
        else:
            self.rope = None

        # ── Decoder blocks ─────────────────────────────────────────────────────
        self.blocks = nn.ModuleList([
            DecoderBlock(
                d_model    = config.d_model,
                n_heads    = config.n_heads,
                d_ff       = config.d_ff,
                dropout    = config.dropout,
                use_flash  = config.use_flash,
                use_swiglu = config.use_swiglu,
                use_rmsnorm= config.use_rmsnorm,
                use_rope   = config.use_rope,
                bias       = config.bias,
            )
            for _ in range(config.n_layers)
        ])

        # ── Final norm + LM head ───────────────────────────────────────────────
        norm_cls    = RMSNorm if config.use_rmsnorm else nn.LayerNorm
        self.norm   = norm_cls(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: share embedding and LM head weights
        self.lm_head.weight = self.token_embed.embed.weight

        # ── Initialise weights ─────────────────────────────────────────────────
        self.apply(self._init_weights)
        # Scale residual projections by 1/sqrt(2 × n_layers)
        scale = (2 * config.n_layers) ** -0.5
        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("w2.weight"):
                p.data.mul_(scale)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        tokens:    torch.Tensor,         # (B, T) integer token ids
        targets:   torch.Tensor = None,  # (B, T) for loss computation
        kv_cache:  dict = None,          # for inference
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = tokens.shape
        assert T <= self.config.max_len, \
            f"Sequence length {T} > max_len {self.config.max_len}"

        # ── Embed tokens ───────────────────────────────────────────────────────
        x = self.token_embed(tokens)   # (B, T, d_model)
        x = self.pos_enc(x)            # add position info (or just dropout for RoPE)

        # ── Precompute RoPE frequencies ────────────────────────────────────────
        freqs = None
        if self.rope is not None:
            start = 0
            if kv_cache and "layer_0" in kv_cache:
                start = kv_cache["layer_0"]["k"].shape[2]
            freqs = self.rope(T, device=tokens.device, offset=start)

        # ── Pass through decoder blocks ────────────────────────────────────────
        for i, block in enumerate(self.blocks):
            x, _ = block(x, kv_cache=kv_cache, layer_idx=i, freqs=freqs)

        # ── Final norm + logits ────────────────────────────────────────────────
        x      = self.norm(x)                   # (B, T, d_model)
        logits = self.lm_head(x)                # (B, T, vocab_size)

        # ── Loss (optional) ────────────────────────────────────────────────────
        loss = None
        if targets is not None:
            # Shift: logits[:, :-1] predicts targets[:, 1:]
            loss = nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, self.config.vocab_size),
                targets[:, 1:].reshape(-1),
            )

        return logits, loss

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_summary(self) -> str:
        n = self.parameter_count()
        embed_params = (self.config.vocab_size * self.config.d_model
                        + self.config.max_len * self.config.d_model)
        other_params = n - embed_params
        return (f"  Total parameters    : {n:>14,}\n"
                f"  Embedding params    : {embed_params:>14,}\n"
                f"  Non-embed params    : {other_params:>14,}")
