"""
Block Manager — Paged Memory for KV Cache

THE CORE IDEA (PagedAttention, Kwon et al. 2023):
─────────────────────────────────────────────────
Traditional inference pre-allocates one contiguous KV buffer per sequence:

    kv = torch.zeros(max_seqs, max_len, 2, n_heads, d_head)

Problems:
  1. INTERNAL FRAGMENTATION: a sequence of 100 tokens occupies space for max_len
  2. NO SHARING: two requests with the same system prompt store duplicate KV
  3. RESERVATION: must know max_len up front

PagedAttention borrows from OS virtual memory:
  - Physical memory = pool of fixed-size BLOCKS (like page frames)
  - Each sequence gets a BLOCK TABLE (like a page table): seq_pos → block_id
  - Blocks are allocated on demand, freed when the sequence finishes
  - Blocks can be SHARED across sequences with the same prefix (copy-on-write)

RESULT:
  - Waste drops from ~40% to <4% in real workloads
  - Average batch size increases by 2–4× at the same GPU memory
  - Enables prefix caching: identical prompts share KV blocks

BLOCK TABLE EXAMPLE:
  block_size = 4, sequence = 11 tokens

  Logical:  [ 0  1  2  3 | 4  5  6  7 | 8  9 10  _]
  Physical: [  block 7   |  block 2   |  block 15  ]

  block_table = [7, 2, 15]    (grows as sequence generates more tokens)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from engine.sequence import Sequence, SequenceStatus
from model.config import CacheConfig


@dataclass
class PhysicalBlock:
    """One page frame in the KV cache pool."""
    block_id:  int
    ref_count: int = 0   # >1 when shared across sequences (prefix caching)

    @property
    def is_free(self) -> bool:
        return self.ref_count == 0


class BlockAllocator:
    """Manages a pool of physical blocks for one device (GPU or CPU)."""

    def __init__(self, device: str, num_blocks: int):
        self.device = device
        self.num_blocks = num_blocks
        self.blocks: List[PhysicalBlock] = [
            PhysicalBlock(block_id=i) for i in range(num_blocks)
        ]
        self.free_block_ids: List[int] = list(range(num_blocks))

    def allocate(self) -> PhysicalBlock:
        if not self.free_block_ids:
            raise MemoryError(f"No free blocks on {self.device}")
        block_id = self.free_block_ids.pop()
        block = self.blocks[block_id]
        block.ref_count = 1
        return block

    def free(self, block: PhysicalBlock) -> None:
        block.ref_count -= 1
        if block.ref_count == 0:
            self.free_block_ids.append(block.block_id)

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_block_ids)

    @property
    def num_used_blocks(self) -> int:
        return self.num_blocks - self.num_free_blocks


class BlockManager:
    """
    Allocates, frees, and tracks KV cache blocks for all sequences.

    Maintains a mapping:  sequence → list of physical block IDs
    Each block stores `block_size` tokens of KV for ONE transformer layer.
    The actual tensors live in ModelRunner.kv_caches (allocated once).

    OPERATIONS:
      can_allocate(seq)    → does the pool have enough free blocks?
      allocate(seq)        → assign blocks to a new sequence
      append_slot(seq)     → add one more block when the last one is full
      free(seq)            → return all blocks to the pool
      fork(parent, child)  → share parent's blocks with child (copy-on-write)
      swap_out(seq)        → move GPU blocks → CPU
      swap_in(seq)         → move CPU blocks → GPU
    """

    def __init__(self, cache_config: CacheConfig):
        self.block_size = cache_config.block_size
        self.gpu_allocator = BlockAllocator("gpu", cache_config.num_gpu_blocks)
        self.cpu_allocator = BlockAllocator("cpu", cache_config.num_cpu_blocks)

        # seq_id → list of PhysicalBlock (in order)
        self._gpu_blocks: Dict[int, List[PhysicalBlock]] = {}
        self._cpu_blocks: Dict[int, List[PhysicalBlock]] = {}

    # ── Core operations ──────────────────────────────────────────────────────

    def can_allocate(self, seq: Sequence) -> bool:
        """Check whether we have enough free GPU blocks for seq's full prompt."""
        blocks_needed = self._blocks_needed(seq.prompt_len)
        return self.gpu_allocator.num_free_blocks >= blocks_needed

    def allocate(self, seq: Sequence) -> None:
        """
        Allocate GPU blocks for a sequence's prompt.
        Called when a WAITING sequence transitions to RUNNING.
        """
        blocks_needed = self._blocks_needed(seq.prompt_len)
        blocks = [self.gpu_allocator.allocate() for _ in range(blocks_needed)]
        self._gpu_blocks[seq.seq_id] = blocks
        seq.block_table = [b.block_id for b in blocks]

    def can_append_slot(self, seq: Sequence) -> bool:
        """
        Check whether the decode write position has an allocated block.

        During decode, we write K,V to position (seq.length - 1) — the last
        appended token.  A new physical block is needed when that position
        falls outside the currently allocated range, i.e. when
          (seq.length - 1) // block_size  >=  len(seq.block_table)
        """
        write_block = (seq.length - 1) // self.block_size
        if write_block < len(seq.block_table):
            return True   # block already allocated
        return self.gpu_allocator.num_free_blocks >= 1

    def append_slot(self, seq: Sequence) -> Optional[Tuple[int, int]]:
        """
        Allocate a new physical block if the decode write position overflows
        the current block table.

        Called by the scheduler before each decode step.  After append_token()
        increments seq.length, the write target is position seq.length-1.
        If that position's block index equals len(seq.block_table) we are at
        the first slot of a new (unallocated) block.
        """
        write_block = (seq.length - 1) // self.block_size
        if write_block < len(seq.block_table):
            return None   # block already allocated, nothing to do

        # Need a new block for the upcoming write
        new_block = self.gpu_allocator.allocate()
        self._gpu_blocks[seq.seq_id].append(new_block)
        seq.block_table.append(new_block.block_id)
        return None

    def free(self, seq: Sequence) -> None:
        """Return all of a sequence's GPU blocks to the free pool."""
        blocks = self._gpu_blocks.pop(seq.seq_id, [])
        for block in blocks:
            self.gpu_allocator.free(block)
        seq.block_table = []

    def fork(self, parent: Sequence, child: Sequence) -> None:
        """
        Give child a reference to parent's blocks (for beam search).
        Uses copy-on-write: blocks are shared until one side writes.
        """
        parent_blocks = self._gpu_blocks[parent.seq_id]
        for block in parent_blocks:
            block.ref_count += 1
        self._gpu_blocks[child.seq_id] = list(parent_blocks)
        child.block_table = list(parent.block_table)

    # ── Swapping (for preemption) ────────────────────────────────────────────

    def swap_out(self, seq: Sequence) -> Dict[int, int]:
        """
        Move a sequence's KV blocks from GPU → CPU (preemption).
        Returns mapping {gpu_block_id: cpu_block_id} for the copy kernel.
        """
        mapping: Dict[int, int] = {}
        gpu_blocks = self._gpu_blocks.pop(seq.seq_id, [])
        cpu_blocks = []

        for gpu_block in gpu_blocks:
            cpu_block = self.cpu_allocator.allocate()
            mapping[gpu_block.block_id] = cpu_block.block_id
            self.gpu_allocator.free(gpu_block)
            cpu_blocks.append(cpu_block)

        self._cpu_blocks[seq.seq_id] = cpu_blocks
        seq.block_table = [b.block_id for b in cpu_blocks]
        seq.status = SequenceStatus.SWAPPED
        return mapping

    def swap_in(self, seq: Sequence) -> Dict[int, int]:
        """
        Move a preempted sequence's KV blocks from CPU → GPU (rescheduled).
        Returns mapping {cpu_block_id: gpu_block_id} for the copy kernel.
        """
        mapping: Dict[int, int] = {}
        cpu_blocks = self._cpu_blocks.pop(seq.seq_id, [])
        gpu_blocks = []

        for cpu_block in cpu_blocks:
            gpu_block = self.gpu_allocator.allocate()
            mapping[cpu_block.block_id] = gpu_block.block_id
            self.cpu_allocator.free(cpu_block)
            gpu_blocks.append(gpu_block)

        self._gpu_blocks[seq.seq_id] = gpu_blocks
        seq.block_table = [b.block_id for b in gpu_blocks]
        seq.status = SequenceStatus.RUNNING
        return mapping

    def can_swap_in(self, seq: Sequence) -> bool:
        blocks_needed = len(self._cpu_blocks.get(seq.seq_id, []))
        return self.gpu_allocator.num_free_blocks >= blocks_needed

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _blocks_needed(self, num_tokens: int) -> int:
        return (num_tokens + self.block_size - 1) // self.block_size

    def get_block_table(self, seq: Sequence) -> List[int]:
        return seq.block_table

    def num_free_gpu_blocks(self) -> int:
        return self.gpu_allocator.num_free_blocks

    def num_free_cpu_blocks(self) -> int:
        return self.cpu_allocator.num_free_blocks

    @property
    def stats(self) -> dict:
        return {
            "gpu_used":  self.gpu_allocator.num_used_blocks,
            "gpu_free":  self.gpu_allocator.num_free_blocks,
            "gpu_total": self.gpu_allocator.num_blocks,
            "cpu_used":  self.cpu_allocator.num_used_blocks,
            "cpu_free":  self.cpu_allocator.num_free_blocks,
        }
