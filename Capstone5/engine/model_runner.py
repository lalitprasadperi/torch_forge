"""
ModelRunner — Prepares batch tensors and runs the model + sampler.

The ModelRunner sits between the Scheduler and the GPU.
It takes a SchedulerOutput (list of Sequence objects) and:

  1. Prepares input tensors:
       - token_ids:    (B, T) — what tokens to feed the model
       - block_tables: (B, max_blocks) — page table for each sequence
       - seq_lens:     (B,) — full sequence lengths after this step
       - cache_offset: (B,) — tokens already in the KV cache
       - is_prefill:   bool

  2. Copies KV blocks if swapping occurred (swap in/out on CPU↔GPU)

  3. Runs the model forward pass

  4. Runs the sampler to get next token IDs

  5. Returns {seq_id → next_token_id} back to the engine

PREFILL vs DECODE BATCHING:
  vLLM runs prefill and decode in the SAME forward pass with a mixed batch.
  Our simpler implementation runs them in SEPARATE forward passes per step:
    - If any sequence is doing prefill, run ALL prefills first
    - Then run decode for remaining sequences
  This avoids complex padding and masking but is slightly less efficient.
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

from engine.sequence import Sequence, SchedulerOutput
from model.config import CacheConfig, ModelConfig
from model.gpt import PagedGPT
from sampling.sampler import Sampler
from model.config import SamplingParams


class ModelRunner:
    """
    Owns the model weights and KV cache tensors.
    Prepares batch inputs and executes forward + sampling.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        device:       torch.device,
    ):
        self.model_config  = model_config
        self.cache_config  = cache_config
        self.device        = device
        self.dtype         = torch.float16 if model_config.dtype == "float16" else torch.float32

        # ── Load model ────────────────────────────────────────────────────────
        self.model = PagedGPT(model_config).to(device=device, dtype=self.dtype)
        self.model.eval()

        # ── Pre-allocate KV caches ────────────────────────────────────────────
        # One tensor per transformer layer:
        #   shape: (num_blocks, 2, block_size, n_heads, d_head)
        self.kv_caches: List[torch.Tensor] = self.model.allocate_kv_caches(
            num_blocks=cache_config.num_gpu_blocks,
            block_size=cache_config.block_size,
            device=device,
            dtype=self.dtype,
        )

        # CPU-side cache for swapped blocks
        self.cpu_kv_caches: List[torch.Tensor] = self.model.allocate_kv_caches(
            num_blocks=cache_config.num_cpu_blocks,
            block_size=cache_config.block_size,
            device=torch.device("cpu"),
            dtype=self.dtype,
        )

        self.sampler    = Sampler()
        self.block_size = cache_config.block_size

        kv_bytes = self.model.kv_cache_bytes(
            cache_config.num_gpu_blocks, cache_config.block_size, self.dtype
        )
        print(f"[ModelRunner] GPU KV cache: {kv_bytes / 1e9:.2f} GB  "
              f"({cache_config.num_gpu_blocks} blocks × {cache_config.block_size} tokens)")

    # ── Main entry point ──────────────────────────────────────────────────────

    @torch.no_grad()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
    ) -> Dict[int, int]:
        """
        Run one model step.
        Returns {seq_id: next_token_id} for all non-finished sequences.
        """
        # ── Perform KV block copies (swapping) ────────────────────────────────
        self._swap_blocks(
            scheduler_output.blocks_to_swap_in,
            scheduler_output.blocks_to_swap_out,
        )

        scheduled   = scheduler_output.scheduled_seqs
        prefill_ids = scheduler_output.prefill_seq_ids

        if not scheduled:
            return {}

        prefill_seqs = [s for s in scheduled if s.seq_id in prefill_ids]
        decode_seqs  = [s for s in scheduled if s.seq_id not in prefill_ids]

        results: Dict[int, int] = {}

        # ── Run prefill sequences ─────────────────────────────────────────────
        if prefill_seqs:
            results.update(self._run_prefill(prefill_seqs))

        # ── Run decode sequences ──────────────────────────────────────────────
        if decode_seqs:
            results.update(self._run_decode(decode_seqs))

        return results

    # ── Prefill ───────────────────────────────────────────────────────────────

    def _run_prefill(self, seqs: List[Sequence]) -> Dict[int, int]:
        """
        Run prefill for a batch of sequences.
        Each sequence may have a different prompt length.
        We pad to the longest prompt in the batch.
        """
        B = len(seqs)
        max_len = max(s.prompt_len for s in seqs)

        # Pad token_ids to (B, max_len)
        token_ids = torch.zeros(B, max_len, dtype=torch.long, device=self.device)
        for i, seq in enumerate(seqs):
            ids = seq.prompt_token_ids
            token_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)

        block_tables, seq_lens, cache_offsets = self._build_block_tables(seqs, is_prefill=True)

        logits = self.model(
            token_ids,
            self.kv_caches,
            block_tables,
            seq_lens,
            is_prefill=True,
            cache_offset=cache_offsets,
        )  # (B, max_len, vocab)

        # Sample from the last REAL token of each sequence
        last_token_logits = torch.stack([
            logits[i, seqs[i].prompt_len - 1] for i in range(B)
        ])  # (B, vocab)

        next_tokens = self.sampler(
            last_token_logits,
            [s.sampling_params for s in seqs],
            [s.output_token_ids for s in seqs],
        )

        return {seqs[i].seq_id: next_tokens[i] for i in range(B)}

    # ── Decode ────────────────────────────────────────────────────────────────

    def _run_decode(self, seqs: List[Sequence]) -> Dict[int, int]:
        """
        Run decode for a batch of sequences (one token per sequence).
        All sequences contribute a single token: their last generated token.
        """
        B = len(seqs)
        token_ids = torch.tensor(
            [s.get_last_token_id() for s in seqs],
            dtype=torch.long, device=self.device
        ).unsqueeze(1)  # (B, 1)

        block_tables, seq_lens, cache_offsets = self._build_block_tables(seqs, is_prefill=False)

        logits = self.model(
            token_ids,
            self.kv_caches,
            block_tables,
            seq_lens,
            is_prefill=False,
            cache_offset=cache_offsets,
        )  # (B, 1, vocab)

        last_logits = logits[:, 0, :]  # (B, vocab)

        next_tokens = self.sampler(
            last_logits,
            [s.sampling_params for s in seqs],
            [s.output_token_ids for s in seqs],
        )

        return {seqs[i].seq_id: next_tokens[i] for i in range(B)}

    # ── Tensor preparation ────────────────────────────────────────────────────

    def _build_block_tables(
        self,
        seqs:       List[Sequence],
        is_prefill: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build block_tables, seq_lens, cache_offsets tensors for the model.

        block_tables: (B, max_blocks_per_seq) int32 — padded with 0
        seq_lens:     (B,) int32 — full length AFTER adding current tokens
        cache_offsets:(B,) int32 — tokens already in cache (= seq_len - new_tokens)
        """
        B             = len(seqs)
        max_blocks    = max(len(s.block_table) for s in seqs)
        if max_blocks == 0:
            max_blocks = 1

        block_tables = torch.zeros(B, max_blocks, dtype=torch.int32, device=self.device)
        for i, seq in enumerate(seqs):
            bt = seq.block_table
            block_tables[i, :len(bt)] = torch.tensor(bt, dtype=torch.int32)

        if is_prefill:
            seq_lens     = torch.tensor([s.prompt_len for s in seqs],
                                         dtype=torch.int32, device=self.device)
            cache_offsets = torch.zeros(B, dtype=torch.int32, device=self.device)
        else:
            # During decode, cache already has prompt + prev generated tokens
            seq_lens      = torch.tensor([s.length for s in seqs],
                                          dtype=torch.int32, device=self.device)
            cache_offsets = torch.tensor([s.length - 1 for s in seqs],
                                          dtype=torch.int32, device=self.device)

        return block_tables, seq_lens, cache_offsets

    # ── KV block copying ──────────────────────────────────────────────────────

    def _swap_blocks(
        self,
        swap_in:  Dict[int, int],   # cpu_block → gpu_block
        swap_out: Dict[int, int],   # gpu_block → cpu_block
    ) -> None:
        """Copy KV blocks between CPU and GPU tensors."""
        for layer_idx in range(self.model_config.n_layers):
            gpu_cache = self.kv_caches[layer_idx]
            cpu_cache = self.cpu_kv_caches[layer_idx]

            for cpu_id, gpu_id in swap_in.items():
                gpu_cache[gpu_id].copy_(cpu_cache[cpu_id], non_blocking=True)

            for gpu_id, cpu_id in swap_out.items():
                cpu_cache[cpu_id].copy_(gpu_cache[gpu_id], non_blocking=True)
