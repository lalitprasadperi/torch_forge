"""
Example 05 — Block Table Inspector

Traces how a sequence's block table grows as it generates tokens.
Visualises the physical block layout at each step.

Key concepts shown:
  - When a new block is allocated (every block_size tokens)
  - How the block table maps logical positions to physical blocks
  - That free blocks decrease, then recover when sequences finish

Run:  python examples/05_block_table_inspector.py
"""
import sys; sys.path.insert(0, ".")
from engine.kv_cache import BlockManager
from engine.sequence import Sequence, SequenceStatus
from model.config import CacheConfig, SamplingParams


BLOCK_SIZE = 4


def render_sequence(seq, block_size, pool_size):
    """ASCII art of this sequence's block layout."""
    lines = []
    for slot_idx, blk_id in enumerate(seq.block_table):
        start = slot_idx * block_size
        end   = min(start + block_size, seq.length)
        filled = end - start
        bar    = "█" * filled + "░" * (block_size - filled)
        lines.append(f"  block {blk_id:2d}: [{bar}]  tokens {start}–{end-1}")
    return lines


def render_pool(bm, pool_size):
    used  = pool_size - bm.num_free_gpu_blocks()
    free  = bm.num_free_gpu_blocks()
    bar   = "█" * used + "░" * free
    return f"  Pool [{bar}]  {used}/{pool_size} used"


# ── Setup ──────────────────────────────────────────────────────────────────────
POOL_SIZE = 16
bm  = BlockManager(CacheConfig(block_size=BLOCK_SIZE, num_gpu_blocks=POOL_SIZE, num_cpu_blocks=4))
sp  = SamplingParams(max_tokens=50)
seq = Sequence(0, list(range(3)), sp)   # 3-token prompt

bm.allocate(seq)
seq.status = SequenceStatus.RUNNING

print("=" * 55)
print("Block Table Inspector")
print(f"block_size={BLOCK_SIZE}, pool={POOL_SIZE} blocks")
print("=" * 55)
print()

def show_state(label):
    print(f"  ── {label} (seq.length={seq.length}) ──")
    for line in render_sequence(seq, BLOCK_SIZE, POOL_SIZE):
        print(line)
    print(render_pool(bm, POOL_SIZE))
    print()

show_state("After allocate (3-token prompt)")

# Generate 12 tokens, showing block allocations
for tok_num in range(12):
    # append_slot BEFORE we generate (ensure room exists)
    prev_blocks = len(seq.block_table)
    bm.append_slot(seq)
    seq.output_token_ids.append(tok_num + 100)

    new_block = len(seq.block_table) > prev_blocks
    label = f"After token {tok_num+1}"
    if new_block:
        label += f"  ← NEW BLOCK {seq.block_table[-1]} allocated"
    show_state(label)

# ── Second sequence: show independent allocation ──────────────────────────────
print("=" * 55)
print("Second sequence starts (independent block table)")
print("=" * 55)
seq2 = Sequence(1, list(range(5)), sp)
bm.allocate(seq2)
seq2.status = SequenceStatus.RUNNING

print(f"\n  seq1 block_table: {seq.block_table}")
print(f"  seq2 block_table: {seq2.block_table}")
print(f"  No block IDs overlap — each gets physically separate blocks")
print(render_pool(bm, POOL_SIZE))

# ── Free seq1 and watch blocks return to pool ─────────────────────────────────
print()
print("  Freeing seq1...")
bm.free(seq)
seq.status = SequenceStatus.FINISHED_STOPPED
print(f"  seq1 block_table after free: {seq.block_table}")
print(render_pool(bm, POOL_SIZE))
print("  Blocks immediately available for new requests!")
