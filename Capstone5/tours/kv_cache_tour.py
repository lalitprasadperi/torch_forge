"""
KV Cache Tour — From Naive to Paged Memory Management

Lessons:
  1. Why KV cache exists (the computation we're avoiding)
  2. Naive KV cache (pre-allocated, wasteful)
  3. Memory fragmentation problem
  4. PagedAttention: the OS analogy
  5. Block allocation and block tables live demo
  6. Prefix caching: sharing blocks across requests
  7. Measuring fragmentation and efficiency

Run:
  cd Capstone5
  python tours/kv_cache_tour.py
"""

import sys
sys.path.insert(0, "/home/jmd/Desktop/TrainHard/LearnTorch/Capstone5")

import math
import time
import torch

from engine.kv_cache import BlockManager
from engine.sequence import Sequence, SequenceStatus
from model.config import CacheConfig, ModelConfig, SamplingParams


DIVIDER = "=" * 65


def lesson1_why_kv_cache():
    print(DIVIDER)
    print("Lesson 1: Why the KV Cache Exists")
    print(DIVIDER)
    print("""
Autoregressive generation: each new token attends to ALL previous tokens.

  t=1: attention over [token_1]                  — 1 key, 1 value
  t=2: attention over [token_1, token_2]         — 2 keys, 2 values
  t=3: attention over [token_1, token_2, token_3]— 3 keys, 3 values
  ...

Without KV cache:
  At step t, we'd recompute K and V for ALL t tokens.
  Cost: O(t²) — quadratic in sequence length.
  For t=1000, step 1000 recomputes 999 tokens unnecessarily.

With KV cache:
  Store K,V at each step. At step t, recompute only the NEW token.
  Cost: O(t) per step — linear.
  Tradeoff: memory grows as O(t × n_layers × n_heads × d_head).

For GPT-2 small generating 1000 tokens:
""")
    n_heads, d_head, n_layers = 12, 64, 12
    seq_len = 1000
    bytes_per_el = 2  # fp16

    kv_per_layer = 2 * seq_len * n_heads * d_head * bytes_per_el
    total_kv     = kv_per_layer * n_layers
    print(f"  KV cache per layer:   {kv_per_layer / 1024:.1f} KB")
    print(f"  Total KV cache:       {total_kv / 1024 / 1024:.2f} MB")
    print(f"  (compare: GPT-2 small weights ≈ 240 MB)\n")


def lesson2_naive_kv():
    print(DIVIDER)
    print("Lesson 2: Naive KV Cache — Pre-allocated Buffers")
    print(DIVIDER)
    print("""
The simple approach: pre-allocate a buffer big enough for the longest
possible sequence. Every request gets a contiguous slice.

  kv = torch.zeros(max_seqs, max_len, n_layers, 2, n_heads, d_head)

Problems:
""")
    max_seqs, max_len = 32, 2048
    n_heads, d_head, n_layers = 12, 64, 12
    bytes_per_el = 2

    total = max_seqs * max_len * n_layers * 2 * n_heads * d_head * bytes_per_el
    print(f"  Allocation for {max_seqs} seqs × {max_len} tokens × GPT-2 small:")
    print(f"  {total / 1e9:.2f} GB  — always, regardless of actual use\n")

    print("  FRAGMENTATION: each request gets max_len slots,")
    print("  but average request only uses 20% of them.")
    print("  Wasted memory = 80% of the KV cache buffer!")
    print()
    print("  Wasted:", total * 0.8 / 1e9, "GB for requests averaging 400 tokens")
    print()
    print("  RESERVATION: must know max_len at request start.")
    print("  Can't add more capacity if the sequence grows longer.")
    print()
    print("  NO SHARING: two requests with the same 1000-token system prompt")
    print("  each store a full copy of those 1000 K,V pairs.")


def lesson3_fragmentation():
    print(DIVIDER)
    print("Lesson 3: Internal Fragmentation in Practice")
    print(DIVIDER)

    import random
    random.seed(42)

    max_seqs    = 16
    max_len     = 512
    n_requests  = 100
    lengths     = [random.randint(10, max_len) for _ in range(n_requests)]

    # Static allocation: each request wastes (max_len - actual_len) slots
    total_allocated = max_seqs * max_len
    total_used_avg  = sum(lengths) / n_requests

    waste_pct = (1 - total_used_avg / max_len) * 100
    print(f"""
  Simulating {n_requests} requests with random lengths 10–{max_len}:
  Average request length: {total_used_avg:.0f} tokens
  Max slot allocation:    {max_len} tokens
  Wasted per slot:        {waste_pct:.0f}%

  With batch size {max_seqs}:
  Effective tokens served: {max_seqs * total_used_avg:.0f}
  Allocated tokens:        {total_allocated}
  GPU KV memory wasted:    {waste_pct:.0f}%
""")
    print("  → With this waste, you can only serve", max_seqs, "concurrent users.")
    print("    PagedAttention drops waste to <4%, letting you serve")
    print(f"   ~{int(max_seqs / (1 - waste_pct / 100)):.0f} concurrent users on the same GPU.")


def lesson4_paged_attention():
    print(DIVIDER)
    print("Lesson 4: PagedAttention — The OS Virtual Memory Analogy")
    print(DIVIDER)
    print("""
Insight (Kwon et al., 2023): The KV cache fragmentation problem is
IDENTICAL to the memory fragmentation problem in operating systems.

OS solution: VIRTUAL MEMORY
  - Physical RAM is divided into fixed-size PAGE FRAMES
  - Each process has a PAGE TABLE mapping virtual → physical pages
  - Pages are allocated on demand, freed when no longer needed
  - Pages can be SHARED (copy-on-write for fork())

vLLM solution: PAGED KV CACHE (PagedAttention)
  - KV memory is divided into fixed-size BLOCKS (our "page frames")
  - Each sequence has a BLOCK TABLE mapping position → block ID
  - Blocks are allocated on demand as sequences grow
  - Blocks can be SHARED for common prefixes (prefix caching)

  Traditional:     Paged:
  ┌────────────┐   ┌──┐ ┌──┐
  │ Req A [███]│   │B7│ │B2│  ← Req A gets 2 non-contiguous blocks
  │     wasted │   └──┘ └──┘
  │ Req B [██] │   ┌──┐
  │     wasted │   │B5│       ← Req B gets 1 block
  │ Req C [█]  │   └──┘
  │     wasted │   ┌──┐
  └────────────┘   │B9│       ← Req C gets 1 block
                   └──┘
                   7 blocks free for new requests!
""")


def lesson5_block_manager_demo():
    print(DIVIDER)
    print("Lesson 5: Block Manager Live Demo")
    print(DIVIDER)

    config = CacheConfig(block_size=4, num_gpu_blocks=20, num_cpu_blocks=5)
    bm     = BlockManager(config)
    sp     = SamplingParams(max_tokens=50)

    print(f"  Pool: {config.num_gpu_blocks} GPU blocks × {config.block_size} tokens/block")
    print(f"  Total capacity: {config.num_gpu_blocks * config.block_size} tokens\n")

    # Create 3 sequences of different lengths
    seqs = [
        Sequence(0, list(range(7)),  sp),  # 7 tokens → 2 blocks
        Sequence(1, list(range(3)),  sp),  # 3 tokens → 1 block
        Sequence(2, list(range(10)), sp),  # 10 tokens → 3 blocks
    ]

    for seq in seqs:
        bm.allocate(seq)
        seq.status = SequenceStatus.RUNNING
        print(f"  Seq {seq.seq_id} ({seq.prompt_len} tokens): blocks = {seq.block_table}")

    print(f"\n  Free blocks remaining: {bm.num_free_gpu_blocks()}")

    # Simulate generating tokens
    print("\n  Generating tokens for seq 0:")
    for step in range(6):
        bm.append_slot(seqs[0])
        seqs[0].output_token_ids.append(step)
        print(f"    step {step+1}: length={seqs[0].length}, blocks={seqs[0].block_table}")

    print(f"\n  After generating 6 tokens, seq 0 has {len(seqs[0].block_table)} blocks")
    print(f"  Free blocks: {bm.num_free_gpu_blocks()}")

    # Free a sequence
    bm.free(seqs[1])
    seqs[1].status = SequenceStatus.FINISHED_STOPPED
    print(f"\n  Freed seq 1: {bm.num_free_gpu_blocks()} blocks now free")
    print(f"  Stats: {bm.stats}")


def lesson6_prefix_caching():
    print(DIVIDER)
    print("Lesson 6: Prefix Caching — Sharing Blocks Across Requests")
    print(DIVIDER)
    print("""
Many production workloads have the same system prompt for every request:
  "You are a helpful assistant. Today is January 1st, 2024..."

Without prefix caching:
  100 concurrent users × 500-token system prompt
  = 50,000 KV tokens duplicated in memory (wasteful!)

With prefix caching:
  1 copy of the system prompt's KV blocks, shared by all users
  Ref-counted blocks: freed only when ALL users release them

The BlockManager.fork() method implements this:
  parent.block_table = [3, 7, 2]  ← system prompt blocks
  child.block_table  = [3, 7, 2]  ← SAME blocks, ref_count = 2
  ref_count[3]++, ref_count[7]++, ref_count[2]++

When child appends a new token to block 2 and that block is full:
  - Allocate a new block (say, block 11)
  - child.block_table = [3, 7, 2, 11]  (copy-on-write in parent stays [3, 7, 2])
  - parent still uses block 2; child gets a new block 11

Memory savings:
""")
    system_prompt_tokens = 500
    block_size           = 16
    n_heads, d_head, n_layers = 12, 64, 12
    concurrent_users     = 100
    bytes_per_el         = 2

    blocks_for_prompt = math.ceil(system_prompt_tokens / block_size)
    kv_per_block      = 2 * block_size * n_heads * d_head * n_layers * bytes_per_el

    no_sharing = concurrent_users * blocks_for_prompt * kv_per_block
    with_sharing = blocks_for_prompt * kv_per_block

    print(f"  System prompt: {system_prompt_tokens} tokens → {blocks_for_prompt} blocks")
    print(f"  Without prefix caching: {no_sharing / 1e6:.0f} MB (100 copies)")
    print(f"  With prefix caching:    {with_sharing / 1e6:.0f} MB (1 shared copy)")
    print(f"  Savings: {(no_sharing - with_sharing) / 1e6:.0f} MB ({concurrent_users}× reduction)")


def lesson7_memory_efficiency():
    print(DIVIDER)
    print("Lesson 7: Measuring Memory Efficiency")
    print(DIVIDER)

    import random
    random.seed(0)
    config = CacheConfig(block_size=16, num_gpu_blocks=100, num_cpu_blocks=20)
    bm     = BlockManager(config)
    sp     = SamplingParams(max_tokens=200)

    # Simulate 20 sequences of varying lengths
    seqs_active = []
    seq_id = 0

    for _ in range(20):
        length = random.randint(8, 64)
        seq = Sequence(seq_id, list(range(length)), sp)
        seq_id += 1
        if bm.can_allocate(seq):
            bm.allocate(seq)
            seq.status = SequenceStatus.RUNNING
            seqs_active.append(seq)

    stats = bm.stats
    print(f"\n  {len(seqs_active)} sequences active")
    print(f"  GPU blocks used:  {stats['gpu_used']}")
    print(f"  GPU blocks free:  {stats['gpu_free']}")
    print(f"  GPU blocks total: {stats['gpu_total']}")

    total_tokens = sum(s.length for s in seqs_active)
    slots_used   = stats['gpu_used'] * config.block_size
    efficiency   = total_tokens / slots_used * 100
    print(f"\n  Total tokens:   {total_tokens}")
    print(f"  Allocated slots:{slots_used}")
    print(f"  Efficiency:     {efficiency:.1f}%")
    print(f"  (100% = perfect packing; typical: 90–97% with paged alloc)")

    # Free some, check efficiency
    for seq in seqs_active[:5]:
        bm.free(seq)
    stats2 = bm.stats
    print(f"\n  After freeing 5 sequences:")
    print(f"  Free blocks:  {stats2['gpu_free']} (immediately reusable!)")
    print(f"  → No compaction needed — this is the key advantage of paging.")


def main():
    lesson1_why_kv_cache()
    lesson2_naive_kv()
    lesson3_fragmentation()
    lesson4_paged_attention()
    lesson5_block_manager_demo()
    lesson6_prefix_caching()
    lesson7_memory_efficiency()

    print(DIVIDER)
    print("Summary")
    print(DIVIDER)
    print("""
  KV cache enables autoregressive generation in O(t) per step.
  Naive pre-allocation wastes 40–80% of KV memory to fragmentation.
  PagedAttention (vLLM, 2023) solves this with OS-style page tables:
    • Fixed-size blocks allocated on demand
    • Block tables per sequence (logical → physical mapping)
    • Prefix caching via shared blocks + ref counting
    • Swap to CPU for preemption
  Result: 2–4× more concurrent users on the same GPU.
""")


if __name__ == "__main__":
    main()
