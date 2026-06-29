"""
Example 15 — Prefix Caching (Block Sharing)

When many requests share the same prefix (e.g. a system prompt),
PagedAttention can SHARE those KV blocks across sequences via reference
counting — only computing and storing the prefix once.

This example demonstrates:
  1. BlockManager.fork() — child sequence inherits parent's block table
  2. Ref counting — blocks freed only when ALL references are gone
  3. Memory savings calculation for a real system prompt scenario

In production (vLLM v0.2+), prefix caching is triggered automatically
by hashing the prefix token IDs. Here we show the mechanics manually.

Run:  python examples/15_prefix_caching_demo.py
"""
import sys; sys.path.insert(0, ".")
from engine.kv_cache import BlockManager, BlockAllocator
from engine.sequence import Sequence, SequenceStatus
from model.config import CacheConfig, SamplingParams


BLOCK_SIZE = 4
POOL_SIZE  = 32
bm         = BlockManager(CacheConfig(block_size=BLOCK_SIZE, num_gpu_blocks=POOL_SIZE, num_cpu_blocks=8))
sp         = SamplingParams(max_tokens=10)


def pool_bar():
    used = POOL_SIZE - bm.num_free_gpu_blocks()
    return f"Pool: {'█'*used + '░'*(POOL_SIZE-used)} {used}/{POOL_SIZE} used"


# ── Step 1: "System prompt" sequence is prefilled and stored ───────────────────
print("=" * 60)
print("Prefix Caching Demo")
print("=" * 60)
print()
print("  System prompt: 8 tokens (occupies 2 blocks of size 4)")
system_seq = Sequence(0, list(range(8)), sp)
bm.allocate(system_seq)
system_seq.status = SequenceStatus.RUNNING
print(f"  system_seq.block_table = {system_seq.block_table}")
print(f"  {pool_bar()}")

# ── Step 2: User request A forks the system prompt's blocks ───────────────────
print()
print("  Request A arrives — shares system prompt via fork()")
req_a = Sequence(1, list(range(8)), sp)  # same prefix
bm.fork(system_seq, req_a)               # shares blocks, ref_count++
req_a.status = SequenceStatus.RUNNING
print(f"  req_a.block_table      = {req_a.block_table}")
print(f"  (same block IDs as system_seq!)")
print(f"  {pool_bar()}  ← no new blocks used!")

# Inspect ref counts
shared_blocks = set(system_seq.block_table) & set(req_a.block_table)
for blk_id in shared_blocks:
    ref = bm.gpu_allocator.blocks[blk_id].ref_count
    print(f"  block {blk_id} ref_count = {ref}")

# ── Step 3: Request B also forks ─────────────────────────────────────────────
print()
print("  Request B arrives — also shares system prompt")
req_b = Sequence(2, list(range(8)), sp)
bm.fork(system_seq, req_b)
req_b.status = SequenceStatus.RUNNING
print(f"  req_b.block_table      = {req_b.block_table}")
print(f"  {pool_bar()}  ← still no new blocks!")

# ── Step 4: Requests generate tokens (need private blocks) ────────────────────
print()
print("  Both requests generate new tokens (private blocks allocated)")
for i in range(5):
    bm.append_slot(req_a); req_a.output_token_ids.append(i+10)
    bm.append_slot(req_b); req_b.output_token_ids.append(i+20)

print(f"  req_a.block_table = {req_a.block_table}")
print(f"  req_b.block_table = {req_b.block_table}")
print(f"  (prefix blocks shared; new blocks private)")
print(f"  {pool_bar()}")

# ── Step 5: Free request A — shared blocks not yet freed ─────────────────────
print()
print("  Request A finishes — shared blocks ref_count drops to 1")
bm.free(req_a)
req_a.status = SequenceStatus.FINISHED_STOPPED
for blk_id in system_seq.block_table:
    ref = bm.gpu_allocator.blocks[blk_id].ref_count
    print(f"  block {blk_id} ref_count = {ref}  (still held by system_seq and req_b)")
print(f"  {pool_bar()}")

# ── Step 6: Free system_seq and req_b — now blocks are fully freed ────────────
print()
print("  System prompt and Request B finish — blocks returned to pool")
bm.free(system_seq); system_seq.status = SequenceStatus.FINISHED_STOPPED
bm.free(req_b);      req_b.status      = SequenceStatus.FINISHED_STOPPED
print(f"  {pool_bar()}")
print()

# ── Memory savings analysis ────────────────────────────────────────────────────
print("=" * 60)
print("Memory Savings from Prefix Caching")
print("=" * 60)

system_prompt_tokens = 500
n_heads, d_head, n_layers = 12, 64, 12
block_size = 16
n_concurrent = 50
fp16_bytes = 2

blocks_for_prompt   = (system_prompt_tokens + block_size - 1) // block_size
bytes_per_block_all_layers = 2 * block_size * n_heads * d_head * fp16_bytes * n_layers

without_sharing = n_concurrent * blocks_for_prompt * bytes_per_block_all_layers
with_sharing    = blocks_for_prompt * bytes_per_block_all_layers  # one shared copy

print(f"\n  System prompt:    {system_prompt_tokens} tokens → {blocks_for_prompt} blocks")
print(f"  Concurrent users: {n_concurrent}")
print()
print(f"  Without prefix caching: {without_sharing/1e6:.0f} MB  ({n_concurrent} full copies)")
print(f"  With prefix caching:    {with_sharing/1e6:.0f} MB    (1 shared copy)")
print(f"  Memory saved:           {(without_sharing-with_sharing)/1e6:.0f} MB  "
      f"({without_sharing/with_sharing:.0f}× reduction)")
print()
print("  These freed blocks can serve ~"
      f"{(without_sharing-with_sharing)//bytes_per_block_all_layers * block_size:,}"
      " additional output tokens")
