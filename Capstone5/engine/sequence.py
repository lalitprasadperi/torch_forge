"""
Sequence — the core data structure of an inference engine.

A REQUEST comes in from the user.
A SEQUENCE is the internal representation: it tracks generated tokens,
block assignments, status, and per-request sampling parameters.
A SEQUENCE GROUP groups sequences that share a prompt (for beam search).
For greedy/sampling inference, each group has exactly one sequence.

STATE MACHINE:
    WAITING ──► RUNNING ──► FINISHED_STOPPED  (EOS or max_tokens)
                   │    └──► FINISHED_ABORTED  (cancelled)
                   │
                   ▼
                SWAPPED ──► RUNNING            (rescheduled)
"""

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional

from model.config import SamplingParams


class SequenceStatus(Enum):
    WAITING          = auto()   # queued, not yet started
    RUNNING          = auto()   # active: either being prefilled or decoded
    SWAPPED          = auto()   # preempted, KV blocks swapped to CPU
    FINISHED_STOPPED = auto()   # ended naturally (EOS or max_tokens)
    FINISHED_ABORTED = auto()   # cancelled by caller


FINISHED_STATUSES = {SequenceStatus.FINISHED_STOPPED, SequenceStatus.FINISHED_ABORTED}


class Sequence:
    """
    One sequence being generated.

    Tracks:
      - prompt + generated tokens
      - which physical KV blocks are allocated to this sequence
      - current status in the state machine
      - timing for latency measurement
    """

    def __init__(
        self,
        seq_id:          int,
        prompt_token_ids: List[int],
        sampling_params: SamplingParams,
    ):
        self.seq_id          = seq_id
        self.prompt_token_ids = list(prompt_token_ids)
        self.output_token_ids: List[int] = []
        self.sampling_params  = sampling_params

        self.status = SequenceStatus.WAITING

        # KV block IDs assigned by the BlockManager.
        # Index i holds the physical block ID for the i-th block of this sequence.
        self.block_table: List[int] = []

        # Timing
        self.arrival_time  = time.monotonic()
        self.start_time:  Optional[float] = None   # first token issued
        self.finish_time: Optional[float] = None

    # ── Token accessors ────────────────────────────────────────────────────────

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def output_len(self) -> int:
        return len(self.output_token_ids)

    @property
    def length(self) -> int:
        """Total tokens so far: prompt + generated."""
        return self.prompt_len + self.output_len

    def get_all_token_ids(self) -> List[int]:
        return self.prompt_token_ids + self.output_token_ids

    def get_last_token_id(self) -> int:
        if self.output_token_ids:
            return self.output_token_ids[-1]
        return self.prompt_token_ids[-1]

    def append_token(self, token_id: int) -> None:
        self.output_token_ids.append(token_id)
        if self.start_time is None:
            self.start_time = time.monotonic()

    # ── Status helpers ─────────────────────────────────────────────────────────

    def is_finished(self) -> bool:
        return self.status in FINISHED_STATUSES

    def check_stop(self) -> bool:
        """Return True if this sequence should stop generating."""
        if self.output_len >= self.sampling_params.max_tokens:
            return True
        if self.output_token_ids and self.output_token_ids[-1] in self.sampling_params.stop_token_ids:
            return True
        return False

    def __repr__(self) -> str:
        return (f"Sequence(id={self.seq_id}, status={self.status.name}, "
                f"len={self.length}, blocks={self.block_table})")


@dataclass
class SequenceGroup:
    """
    A group of sequences that share the same prompt.
    For non-beam-search inference this always holds exactly one sequence.
    Grouped here to support future beam search (multiple output sequences
    from the same prompt can share prompt KV blocks via copy-on-write).
    """
    request_id:  str
    sequences:   List[Sequence]
    arrival_time: float = field(default_factory=time.monotonic)

    def get_seqs(self, status: Optional[SequenceStatus] = None) -> List[Sequence]:
        if status is None:
            return self.sequences
        return [s for s in self.sequences if s.status == status]

    def is_finished(self) -> bool:
        return all(s.is_finished() for s in self.sequences)

    @property
    def sampling_params(self) -> SamplingParams:
        return self.sequences[0].sampling_params


@dataclass
class SchedulerOutput:
    """
    What the scheduler decided to do this step.
    The model runner uses this to prepare the batch.
    """
    scheduled_seqs:    List[Sequence]    # sequences to run this step (prefill or decode)
    prefill_seq_ids:   set               # subset doing prefill (first time through)
    blocks_to_swap_in:  dict             # cpu_block_id → gpu_block_id
    blocks_to_swap_out: dict             # gpu_block_id → cpu_block_id
    preempted:         List[Sequence]    # sequences preempted this step


@dataclass
class CompletionOutput:
    """One sequence's contribution to the final response."""
    index:        int
    text:         str
    token_ids:    List[int]
    finish_reason: Optional[str] = None


@dataclass
class RequestOutput:
    """Response returned to the caller for one request."""
    request_id: str
    prompt:     str
    outputs:    List[CompletionOutput]
    finished:   bool = False
