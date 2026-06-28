"""
LLMEngine — Top-Level Inference Engine

This is the main entry point: it ties together the Scheduler,
BlockManager, and ModelRunner into a single `step()` loop.

USAGE:
    engine = LLMEngine.from_config(model_config, cache_config, scheduler_config)
    engine.add_request("req-1", "The meaning of life is", SamplingParams(max_tokens=100))
    engine.add_request("req-2", "Hello, my name is",       SamplingParams(temperature=0.8))

    while engine.has_unfinished_requests():
        outputs = engine.step()
        for out in outputs:
            if out.finished:
                print(out.request_id, out.outputs[0].text)

STEP INTERNALS:
    1. Scheduler.schedule() → decide who runs this step
    2. ModelRunner.execute_model(scheduler_output) → logits → sample
    3. For each sampled token:
         - Append to sequence
         - Check stop conditions
         - Mark FINISHED if done
    4. Scheduler.free_finished_seqs() → free KV blocks
    5. Build and return RequestOutput objects

STREAMING:
    Each step() returns partial outputs for all running sequences.
    Use `finished=False` outputs for streaming, `finished=True` as the final.
"""

from __future__ import annotations
import time
import uuid
from typing import Callable, Dict, Iterator, List, Optional

import torch

from engine.kv_cache import BlockManager
from engine.model_runner import ModelRunner
from engine.scheduler import Scheduler
from engine.sequence import (
    CompletionOutput, RequestOutput, Sequence, SequenceGroup, SequenceStatus
)
from model.config import CacheConfig, ModelConfig, SamplingParams, SchedulerConfig


class LLMEngine:
    """
    Continuous-batching inference engine.

    Manages the request lifecycle:
      add_request() → step() × N → RequestOutput(finished=True)
    """

    def __init__(
        self,
        model_runner:     ModelRunner,
        scheduler:        Scheduler,
        tokenizer,
    ):
        self.model_runner = model_runner
        self.scheduler    = scheduler
        self.tokenizer    = tokenizer

        # request_id → SequenceGroup
        self._request_id_to_group: Dict[str, SequenceGroup] = {}
        self._next_seq_id = 0

        self._step_count   = 0
        self._total_tokens = 0
        self._start_time   = time.monotonic()

    # ── Public API ────────────────────────────────────────────────────────────

    def add_request(
        self,
        request_id:      str,
        prompt:          str,
        sampling_params: SamplingParams,
        prompt_token_ids: Optional[List[int]] = None,
    ) -> None:
        """
        Enqueue a new generation request.

        prompt_token_ids can be provided directly (e.g. for testing
        without a real tokenizer). If None, self.tokenizer.encode() is used.
        """
        if prompt_token_ids is None:
            prompt_token_ids = self.tokenizer.encode(prompt)

        seq = Sequence(
            seq_id           = self._next_seq_id,
            prompt_token_ids = prompt_token_ids,
            sampling_params  = sampling_params,
        )
        self._next_seq_id += 1

        group = SequenceGroup(
            request_id   = request_id,
            sequences    = [seq],
            arrival_time = time.monotonic(),
        )
        self._request_id_to_group[request_id] = group
        self.scheduler.add_seq_group(group)

    def step(self) -> List[RequestOutput]:
        """
        Run one iteration of the engine.

        Returns partial or complete outputs for each active request.
        Finished requests have output.finished = True.
        """
        # ── Schedule ──────────────────────────────────────────────────────────
        scheduler_output = self.scheduler.schedule()

        if not scheduler_output.scheduled_seqs:
            return []

        # ── Run model ─────────────────────────────────────────────────────────
        seq_id_to_token = self.model_runner.execute_model(scheduler_output)

        # ── Process sampled tokens ────────────────────────────────────────────
        for seq in scheduler_output.scheduled_seqs:
            if seq.seq_id not in seq_id_to_token:
                continue

            token_id = seq_id_to_token[seq.seq_id]
            seq.append_token(token_id)
            self._total_tokens += 1

            if seq.check_stop():
                seq.status    = SequenceStatus.FINISHED_STOPPED
                seq.finish_time = time.monotonic()

        # ── Free finished sequences ────────────────────────────────────────────
        self.scheduler.free_finished_seqs()

        # ── Build output objects ──────────────────────────────────────────────
        outputs = self._build_outputs()

        self._step_count += 1
        return outputs

    def generate(
        self,
        prompt:          str,
        sampling_params: Optional[SamplingParams] = None,
    ) -> RequestOutput:
        """Convenience: add one request, run until done, return result."""
        if sampling_params is None:
            sampling_params = SamplingParams()

        request_id = str(uuid.uuid4())
        self.add_request(request_id, prompt, sampling_params)

        while self.has_unfinished_requests():
            outputs = self.step()
            for out in outputs:
                if out.request_id == request_id and out.finished:
                    return out

        raise RuntimeError("Engine finished without completing request")

    def stream(
        self,
        prompt:          str,
        sampling_params: Optional[SamplingParams] = None,
        on_token:        Optional[Callable[[str], None]] = None,
    ) -> Iterator[str]:
        """
        Streaming generation: yields each new token text as it's produced.
        Optionally calls on_token(text) callback too.
        """
        if sampling_params is None:
            sampling_params = SamplingParams()

        request_id = str(uuid.uuid4())
        self.add_request(request_id, prompt, sampling_params)

        prev_output_len = 0
        while self.has_unfinished_requests():
            outputs = self.step()
            for out in outputs:
                if out.request_id != request_id:
                    continue
                curr_text = out.outputs[0].text if out.outputs else ""
                new_text  = curr_text[prev_output_len:]
                if new_text:
                    if on_token:
                        on_token(new_text)
                    yield new_text
                    prev_output_len = len(curr_text)
                if out.finished:
                    return

    def has_unfinished_requests(self) -> bool:
        return self.scheduler.has_unfinished_seqs()

    def abort_request(self, request_id: str) -> None:
        group = self._request_id_to_group.pop(request_id, None)
        if group is None:
            return
        for seq in group.sequences:
            seq.status = SequenceStatus.FINISHED_ABORTED
        self.scheduler.free_finished_seqs()

    @property
    def stats(self) -> dict:
        elapsed = time.monotonic() - self._start_time
        return {
            "steps":         self._step_count,
            "total_tokens":  self._total_tokens,
            "tokens_per_sec": self._total_tokens / max(elapsed, 1e-6),
            **self.scheduler.stats,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_outputs(self) -> List[RequestOutput]:
        outputs: List[RequestOutput] = []
        for request_id, group in list(self._request_id_to_group.items()):
            seq        = group.sequences[0]
            token_ids  = seq.output_token_ids
            text       = self.tokenizer.decode(token_ids)
            finished   = seq.is_finished()

            finish_reason = None
            if seq.status == SequenceStatus.FINISHED_STOPPED:
                if seq.output_token_ids and seq.output_token_ids[-1] in seq.sampling_params.stop_token_ids:
                    finish_reason = "stop"
                else:
                    finish_reason = "length"

            completion = CompletionOutput(
                index        = 0,
                text         = text,
                token_ids    = list(token_ids),
                finish_reason = finish_reason,
            )
            outputs.append(RequestOutput(
                request_id = request_id,
                prompt     = self.tokenizer.decode(seq.prompt_token_ids),
                outputs    = [completion],
                finished   = finished,
            ))

            if finished:
                del self._request_id_to_group[request_id]

        return outputs

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_config(
        cls,
        model_config:     ModelConfig,
        cache_config:     CacheConfig,
        scheduler_config: SchedulerConfig,
        tokenizer,
        device:           Optional[torch.device] = None,
    ) -> "LLMEngine":
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        block_manager = BlockManager(cache_config)
        scheduler     = Scheduler(scheduler_config, block_manager)
        model_runner  = ModelRunner(model_config, cache_config, device)

        return cls(model_runner, scheduler, tokenizer)
