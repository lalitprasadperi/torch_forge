"""
Scheduler Tour — Continuous Batching and Preemption

Lessons:
  1. Static batching — the problem
  2. Continuous batching — how it works
  3. Scheduling with memory constraints
  4. Preemption modes (recompute vs swap)
  5. Live simulation of the scheduler
  6. Chunked prefill — hiding prefill latency
  7. Priority scheduling

Run:
  cd Capstone5
  python tours/scheduler_tour.py
"""

import sys
sys.path.insert(0, "/home/jmd/Desktop/TrainHard/LearnTorch/Capstone5")

import random
import time
from typing import List

from engine.kv_cache import BlockManager
from engine.scheduler import Scheduler
from engine.sequence import Sequence, SequenceGroup, SequenceStatus
from model.config import CacheConfig, SamplingParams, SchedulerConfig


DIVIDER = "=" * 65


def lesson1_static_batching_problem():
    print(DIVIDER)
    print("Lesson 1: The Static Batching Problem")
    print(DIVIDER)
    print("""
Traditional ML serving uses STATIC BATCHING:
  1. Wait for a full batch of N requests
  2. Run them all together until ALL are done
  3. Then accept the next batch

The problem: requests have wildly different output lengths.
  Seq A: 10 output tokens  (fast)
  Seq B: 200 output tokens (slow)
  Seq C: 50 output tokens  (medium)

  With static batching (batch_size=3):
  ┌─────────────────────────────────────────┐
  │ A: ██ done ■■■■■■■■■■■■■■■■ idle       │
  │ B: █████████████████████████████ done   │
  │ C: ████████████ done ■■■■■■■■■■■■ idle │
  └─────────────────────────────────────────┘
         ↑ GPU wasted while A and C wait for B

  GPU utilisation: 60% (tokens generated / total slots × time)
""")

    # Simulate the waste
    batch_size = 4
    random.seed(42)
    output_lens = sorted([random.randint(10, 300) for _ in range(batch_size)], reverse=True)
    max_len     = output_lens[0]

    total_slots = batch_size * max_len
    useful_work = sum(output_lens)
    util        = useful_work / total_slots * 100

    print(f"  Example batch: {output_lens}")
    print(f"  Max length:  {max_len}")
    print(f"  Useful work: {useful_work} tokens")
    print(f"  Total slots: {total_slots}")
    print(f"  Utilisation: {util:.1f}%")


def lesson2_continuous_batching():
    print(DIVIDER)
    print("Lesson 2: Continuous Batching (Orca, Yu et al. 2022)")
    print(DIVIDER)
    print("""
Key insight: GPU can operate at the ITERATION level, not the REQUEST level.

Instead of waiting for all sequences to finish:
  • After each decode step, check for newly finished sequences
  • Immediately fill empty slots with NEW waiting requests
  • Different sequences can be at different generation stages

  ┌─────────────────────────────────────────┐
  │ A: ██ done ──► D: ████████ done ──► F  │
  │ B: █████████████████████████████ done   │
  │ C: ████████████ done ──► E: ████████   │
  └─────────────────────────────────────────┘
         ↑ GPU always busy!

  Utilisation: ~95–99%

The key: we batch at DECODE time, not REQUEST time.
  At step t: batch = [all sequences currently being decoded]
  When sequence i finishes: remove from batch, add next waiting sequence
  When sequence j starts:   run its PREFILL, then add to decode batch
""")


def lesson3_scheduler_demo():
    print(DIVIDER)
    print("Lesson 3: Scheduler Live Simulation")
    print(DIVIDER)

    # Set up a small engine
    cache_config     = CacheConfig(block_size=4, num_gpu_blocks=30, num_cpu_blocks=10)
    scheduler_config = SchedulerConfig(
        max_num_seqs           = 4,
        max_num_batched_tokens = 128,
        preemption_mode        = "recompute",
    )
    bm        = BlockManager(cache_config)
    scheduler = Scheduler(scheduler_config, bm)
    sp        = SamplingParams(max_tokens=8)

    # Enqueue 6 requests
    for i in range(6):
        prompt_len = 3 + i
        seq   = Sequence(i, list(range(prompt_len)), sp)
        group = SequenceGroup(request_id=f"req-{i}", sequences=[seq])
        scheduler.add_seq_group(group)

    print(f"  Queued 6 requests. Max concurrent: {scheduler_config.max_num_seqs}")
    print(f"  GPU blocks: {cache_config.num_gpu_blocks}")
    print()

    for step in range(12):
        out = scheduler.schedule()
        n_prefill = len(out.prefill_seq_ids)
        n_decode  = len(out.scheduled_seqs) - n_prefill

        # Simulate: append one token per decode seq, mark some as finished
        for seq in out.scheduled_seqs:
            if seq.seq_id not in out.prefill_seq_ids:
                seq.append_token(99)
            if seq.check_stop():
                seq.status = SequenceStatus.FINISHED_STOPPED

        scheduler.free_finished_seqs()

        stats = scheduler.stats
        print(f"  Step {step:2d}: "
              f"prefill={n_prefill}, decode={n_decode}, "
              f"waiting={stats['waiting']}, running={stats['running']}, "
              f"gpu_free={stats['gpu_free']}")

        if not scheduler.has_unfinished_seqs():
            break

    print("\n  All requests completed!")


def lesson4_preemption():
    print(DIVIDER)
    print("Lesson 4: Preemption — Managing Memory Pressure")
    print(DIVIDER)
    print("""
What happens when you run out of KV cache blocks?

Without preemption: wait (stall new requests).
With preemption: evict one running sequence to make room.

TWO PREEMPTION MODES:

  RECOMPUTE:
    Drop the KV cache, put sequence back in waiting queue.
    Pros: no extra memory for swap buffer, simple.
    Cons: sequence must re-run prefill when rescheduled (wastes compute).
    Best for: short prompts, GPU memory too tight for swap buffers.

  SWAP:
    Copy KV blocks from GPU → CPU RAM.
    Pros: reschedule is instant (copy back and continue).
    Cons: copy bandwidth, need free CPU memory.
    Best for: long prompts where recompute is expensive.

vLLM uses SWAP by default. Our scheduler supports both.

PREEMPTION POLICY:
  vLLM uses FCFS + youngest-first preemption:
    "Preempt the most recently started sequence."
    Rationale: younger seqs have fewer generated tokens → cheaper to recompute.

  Our scheduler preempts from the end of the running queue.
""")

    print("  Simulation: tight memory, many requests...")
    cache_config     = CacheConfig(block_size=4, num_gpu_blocks=12, num_cpu_blocks=8)
    scheduler_config = SchedulerConfig(
        max_num_seqs           = 6,
        max_num_batched_tokens = 256,
        preemption_mode        = "swap",
    )
    bm        = BlockManager(cache_config)
    scheduler = Scheduler(scheduler_config, bm)
    sp        = SamplingParams(max_tokens=20)

    for i in range(8):
        seq   = Sequence(i, list(range(5 + i)), sp)
        group = SequenceGroup(request_id=f"req-{i}", sequences=[seq])
        scheduler.add_seq_group(group)

    for step in range(15):
        out   = scheduler.schedule()
        stats = scheduler.stats

        for seq in out.scheduled_seqs:
            if seq.seq_id not in out.prefill_seq_ids:
                seq.append_token(99)
            if seq.check_stop():
                seq.status = SequenceStatus.FINISHED_STOPPED

        if out.blocks_to_swap_out:
            print(f"  Step {step:2d}: PREEMPTED! Swapping out blocks: {list(out.blocks_to_swap_out.keys())[:3]}...")
        if out.blocks_to_swap_in:
            print(f"  Step {step:2d}: Swapping in blocks: {list(out.blocks_to_swap_in.keys())[:3]}...")

        scheduler.free_finished_seqs()
        if not scheduler.has_unfinished_seqs():
            print(f"\n  All requests done in {step+1} steps")
            break


def lesson5_scheduling_budget():
    print(DIVIDER)
    print("Lesson 5: The Token Budget — Balancing Prefill and Decode")
    print(DIVIDER)
    print("""
PROBLEM: A large prefill monopolises the GPU.
  If a 2000-token prompt arrives while 8 sequences are decoding,
  running the prefill takes 8× longer than one decode step.
  During that time, decode users see no tokens → high TPOT.

SOLUTION: max_num_batched_tokens budget.
  scheduler_config.max_num_batched_tokens = 512

  If a pending prefill has 2000 tokens, it must wait until the
  batch has enough token budget (or be chunked into pieces).

CHUNKED PREFILL (vLLM v0.3+):
  Split long prefills into chunks of max_chunk_size tokens.
  Each step runs ONE chunk of prefill + ALL decode.

  Step 1: chunk_1 [0:512] of prompt + decode batch
  Step 2: chunk_2 [512:1024] + decode batch
  Step 3: chunk_3 [1024:1536] + decode batch
  Step 4: chunk_4 [1536:2000] + decode batch → token ready!

  TTFT increases slightly (4 steps vs 1) but TPOT stays smooth.
  This is the "disaggregated prefill" idea taken to its logical end.

  In our scheduler: if seq.prompt_len > token_budget, skip it.
  A proper chunked prefill would track chunk position. Exercise!
""")
    print("  Our scheduler config example:")
    cfg = SchedulerConfig(
        max_num_seqs           = 16,
        max_num_batched_tokens = 512,
    )
    print(f"    max_num_seqs:           {cfg.max_num_seqs}")
    print(f"    max_num_batched_tokens: {cfg.max_num_batched_tokens}")
    print()
    print("  A request with 800-token prompt will wait until the batch")
    print("  has enough remaining budget (or implement chunked prefill).")


def lesson6_priority_scheduling():
    print(DIVIDER)
    print("Lesson 6: Beyond FCFS — Priority and SLO-Aware Scheduling")
    print(DIVIDER)
    print("""
vLLM currently uses FCFS (First Come First Served) scheduling.
More advanced schedulers consider:

  SLO-AWARE (Alizadeh et al.):
    Each request has a latency SLO (e.g., TPOT < 20ms).
    Schedule to minimise SLO violations, not just FCFS.
    Trade off: which request is most "at risk" of missing its SLO?

  PRIORITY QUEUES:
    Premium users → higher priority queue.
    Background jobs → low priority, can be preempted freely.

  PREDICTIVE SCHEDULING:
    Estimate output length from the prompt.
    Short outputs first (Shortest Job First) → higher throughput.
    Problem: length prediction is hard without running the model.

  DISAGGREGATED SERVING:
    Run PREFILL machines and DECODE machines separately.
    Prefill is compute-bound → pack GPUs with matmul-heavy work.
    Decode is memory-bound → use GPUs with high HBM bandwidth.
    KV cache transferred between machines over NVLink/InfiniBand.

Current research frontier: Sarathi (chunked prefill),
  DistServe (disaggregation), Llumnix (live migration), TetriInfer.
""")


def main():
    lesson1_static_batching_problem()
    lesson2_continuous_batching()
    lesson3_scheduler_demo()
    lesson4_preemption()
    lesson5_scheduling_budget()
    lesson6_priority_scheduling()

    print(DIVIDER)
    print("Summary")
    print(DIVIDER)
    print("""
  Static batching: wait for all → 60% GPU utilisation, high latency.
  Continuous batching: fill slots immediately → 95%+ GPU utilisation.
  PagedAttention: memory-efficient KV → larger batches.
  Preemption: swap or recompute to handle memory pressure.
  Token budget: prevents large prefills from starving decode.
  Chunked prefill: smooth TPOT under mixed workloads.

  The scheduler is the "OS kernel" of an LLM serving system.
""")


if __name__ == "__main__":
    main()
