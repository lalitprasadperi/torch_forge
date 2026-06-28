"""
Configuration dataclasses for Mini vLLM.

Three independent configs that mirror the real vLLM structure:
  ModelConfig   — transformer architecture hyperparameters
  CacheConfig   — KV cache block layout
  SchedulerConfig — batching limits and policies
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Transformer architecture."""
    vocab_size:  int   = 50257   # GPT-2 tokenizer
    d_model:     int   = 768
    n_heads:     int   = 12
    n_layers:    int   = 12
    d_ff:        int   = 3072    # 4 * d_model
    context_len: int   = 1024
    dropout:     float = 0.0
    dtype:       str   = "float16"

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0
        self.d_head = self.d_model // self.n_heads

    # Named presets
    @classmethod
    def gpt2_small(cls):
        return cls(vocab_size=50257, d_model=768,  n_heads=12, n_layers=12, d_ff=3072)

    @classmethod
    def gpt2_medium(cls):
        return cls(vocab_size=50257, d_model=1024, n_heads=16, n_layers=24, d_ff=4096)

    @classmethod
    def nano(cls):
        """Tiny model for fast testing (char-level tokenizer friendly)."""
        return cls(vocab_size=256, d_model=128, n_heads=4, n_layers=4, d_ff=512, context_len=512)


@dataclass
class CacheConfig:
    """
    KV cache is divided into fixed-size BLOCKS (pages).

    Each block stores `block_size` tokens of K and V for ONE layer.
    A sequence's KV cache is tracked as a list of block IDs — like a page table.

    Memory per block (all layers, fp16):
        2 * block_size * n_heads * d_head * 2 bytes * n_layers
    """
    block_size:          int   = 16      # tokens per block
    num_gpu_blocks:      int   = 1024    # total physical blocks on GPU
    num_cpu_blocks:      int   = 256     # blocks on CPU (for swapping)
    gpu_memory_util:     float = 0.90    # fraction of free GPU VRAM to use for KV

    def blocks_needed(self, num_tokens: int) -> int:
        """How many blocks does a sequence of num_tokens need?"""
        return (num_tokens + self.block_size - 1) // self.block_size


@dataclass
class SchedulerConfig:
    """
    Controls how many requests and tokens are processed per step.

    max_num_seqs limits batch size (memory).
    max_num_batched_tokens limits total tokens per step (compute).
    Setting max_num_batched_tokens = max_seq_len * max_num_seqs allows
    full prefills without throttling; lower values enforce chunked prefill.
    """
    max_num_seqs:           int = 32     # max concurrent sequences
    max_num_batched_tokens: int = 4096   # max total tokens per scheduler step
    max_model_len:          int = 1024   # max sequence length
    preemption_mode:        str = "recompute"  # "recompute" or "swap"


@dataclass
class SamplingParams:
    """Per-request generation parameters."""
    temperature:    float          = 1.0
    top_k:          int            = -1       # -1 = disabled
    top_p:          float          = 1.0      # 1.0 = disabled
    max_tokens:     int            = 256
    stop_token_ids: list           = field(default_factory=list)
    repetition_penalty: float      = 1.0     # 1.0 = no penalty
    seed:           Optional[int]  = None

    def is_greedy(self) -> bool:
        return self.temperature == 0.0 or (self.top_k == 1)
