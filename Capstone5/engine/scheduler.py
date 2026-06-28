"""
Scheduler — Continuous Batching Policy

THE BIG IDEA:
─────────────
Static batching: wait until all sequences in a batch finish before starting
new ones. The slowest sequence in the batch holds everyone else hostage.

    Batch │ Seq A ████████████████████████ done │
          │ Seq B ████████ done                  │  <-- GPU idle while waiting for A
          │ Seq C ████████████████ done          │
          └────────────────────────────────────────
                                  ^ all finish, THEN start new batch

Continuous batching (Orca, Yu et al. 2022): as soon as ANY sequence
finishes, the freed slot is immediately filled with the next waiting request.
Sequences are decoded at the ITERATION level, not the REQUEST level.

    Batch │ Seq A ████████████████████████ done │
          │ Seq B ████████ done ► Seq D █████   │  <-- D starts immediately
          │ Seq C ████████████████ done ► Seq E │  <-- E starts mid-batch
          └────────────────────────────────────────
                GPU busy at all times ↑

SCHEDULER POLICY (FCFS with memory awareness):
  Each step:
    1. Try to resume any SWAPPED sequences (if GPU has room)
    2. Try to admit WAITING sequences for prefill (if within token budget)
    3. Decode all RUNNING sequences
    4. If total memory pressure: preempt lowest-priority RUNNING sequence

PREEMPTION MODES:
  recompute: drop the KV cache, re-run prefill when rescheduled (cheap storage, costly on reschedule)
  swap:      move KV blocks to CPU RAM (costly now, free reschedule)
"""

from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple

from engine.kv_cache import BlockManager
from engine.sequence import (
    SchedulerOutput, Sequence, SequenceGroup, SequenceStatus
)
from model.config import SchedulerConfig


class Scheduler:
    """
    Decides which sequences to run each step.

    Maintains three queues:
      waiting  — new requests, not yet started
      running  — sequences being decoded (or about to be prefilled)
      swapped  — preempted, KV on CPU

    Each call to `schedule()` returns a SchedulerOutput describing
    what the ModelRunner should do this step.
    """

    def __init__(self, config: SchedulerConfig, block_manager: BlockManager):
        self.config        = config
        self.block_manager = block_manager

        self.waiting: List[SequenceGroup] = []   # FCFS order
        self.running: List[SequenceGroup] = []
        self.swapped: List[SequenceGroup] = []

    def add_seq_group(self, seq_group: SequenceGroup) -> None:
        """Enqueue a new request."""
        self.waiting.append(seq_group)

    def has_unfinished_seqs(self) -> bool:
        return bool(self.waiting or self.running or self.swapped)

    def get_num_unfinished_seqs(self) -> int:
        return (sum(len(g.sequences) for g in self.waiting) +
                sum(len(g.sequences) for g in self.running) +
                sum(len(g.sequences) for g in self.swapped))

    # ── Main scheduling entry point ───────────────────────────────────────────

    def schedule(self) -> SchedulerOutput:
        """
        Build this step's batch.

        Returns SchedulerOutput with:
          scheduled_seqs   — all sequences that will run (prefill + decode)
          prefill_seq_ids  — which of those are doing prefill (first step)
          blocks_to_swap_* — memory ops to perform before the model step
          preempted        — sequences kicked out to make room
        """
        blocks_to_swap_in:  Dict[int, int] = {}
        blocks_to_swap_out: Dict[int, int] = {}
        preempted: List[Sequence] = []

        # ── 1. Try to swap swapped sequences back in ──────────────────────────
        still_swapped: List[SequenceGroup] = []
        for group in self.swapped:
            if len(self.running) >= self.config.max_num_seqs:
                still_swapped.append(group)
                continue
            seqs = group.get_seqs(SequenceStatus.SWAPPED)
            if all(self.block_manager.can_swap_in(s) for s in seqs):
                for s in seqs:
                    mapping = self.block_manager.swap_in(s)
                    blocks_to_swap_in.update(mapping)
                self.running.append(group)
            else:
                still_swapped.append(group)
        self.swapped = still_swapped

        # ── 2. Ensure running sequences have room for one more token ──────────
        #       Preempt if we can't append a slot.
        running_with_room: List[SequenceGroup] = []
        for group in self.running:
            seqs = group.get_seqs(SequenceStatus.RUNNING)
            can_append = all(self.block_manager.can_append_slot(s) for s in seqs)
            if can_append:
                for s in seqs:
                    self.block_manager.append_slot(s)
                running_with_room.append(group)
            else:
                self._preempt(group, blocks_to_swap_out, preempted)
        self.running = running_with_room

        # ── 3. Admit new sequences from waiting queue ──────────────────────────
        #       Budget: max_num_seqs total, max_num_batched_tokens prefill budget
        prefill_seq_ids: Set[int] = set()
        token_budget    = self.config.max_num_batched_tokens
        new_running:    List[SequenceGroup] = []

        for group in list(self.waiting):
            if len(self.running) + len(new_running) >= self.config.max_num_seqs:
                break

            seq = group.sequences[0]  # one seq per group in greedy mode
            if seq.prompt_len > token_budget:
                break   # not enough token budget for this prefill
            if not self.block_manager.can_allocate(seq):
                break   # not enough KV blocks

            self.waiting.remove(group)
            self.block_manager.allocate(seq)
            seq.status = SequenceStatus.RUNNING
            prefill_seq_ids.add(seq.seq_id)
            token_budget -= seq.prompt_len
            new_running.append(group)

        self.running.extend(new_running)

        # ── 4. Collect all scheduled sequences ────────────────────────────────
        scheduled_seqs: List[Sequence] = []
        for group in self.running:
            scheduled_seqs.extend(group.get_seqs(SequenceStatus.RUNNING))

        return SchedulerOutput(
            scheduled_seqs    = scheduled_seqs,
            prefill_seq_ids   = prefill_seq_ids,
            blocks_to_swap_in  = blocks_to_swap_in,
            blocks_to_swap_out = blocks_to_swap_out,
            preempted         = preempted,
        )

    # ── Post-step: update queue state ─────────────────────────────────────────

    def free_finished_seqs(self) -> None:
        """Remove finished sequences and free their KV blocks."""
        still_running: List[SequenceGroup] = []
        for group in self.running:
            seqs_done = [s for s in group.sequences if s.is_finished()]
            for s in seqs_done:
                self.block_manager.free(s)
            if not group.is_finished():
                still_running.append(group)
        self.running = still_running

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _preempt(
        self,
        group:            SequenceGroup,
        swap_out_map:     Dict[int, int],
        preempted_list:   List[Sequence],
    ) -> None:
        seqs = group.get_seqs(SequenceStatus.RUNNING)
        if self.config.preemption_mode == "swap":
            for s in seqs:
                mapping = self.block_manager.swap_out(s)
                swap_out_map.update(mapping)
            self.swapped.append(group)
        else:
            # recompute: drop KV cache entirely, move back to waiting
            for s in seqs:
                self.block_manager.free(s)
                s.status = SequenceStatus.WAITING
                s.output_token_ids = []   # reset: will be re-prefilled
            self.waiting.insert(0, group)   # priority re-entry
        preempted_list.extend(seqs)

    @property
    def stats(self) -> dict:
        return {
            "waiting": len(self.waiting),
            "running": len(self.running),
            "swapped": len(self.swapped),
            **self.block_manager.stats,
        }
