"""
Example 07 — Preemption Demo

Forces memory pressure by using a tiny GPU block pool and many requests.
Shows the scheduler swapping sequences out to CPU and back in.

Demonstrates:
  - Swap mode: KV blocks moved GPU → CPU, then CPU → GPU on reschedule
  - Recompute mode: KV blocks dropped, sequence re-prefills from scratch
  - The fallback: swap → recompute when CPU is also full

Run:  python examples/07_preemption_demo.py
"""
import sys; sys.path.insert(0, ".")
from engine.kv_cache import BlockManager
from engine.scheduler import Scheduler
from engine.sequence import Sequence, SequenceGroup, SequenceStatus
from model.config import CacheConfig, SamplingParams, SchedulerConfig


def run_simulation(mode, gpu_blocks, cpu_blocks, label):
    cache_cfg = CacheConfig(block_size=4, num_gpu_blocks=gpu_blocks, num_cpu_blocks=cpu_blocks)
    sched_cfg = SchedulerConfig(
        max_num_seqs=10,
        max_num_batched_tokens=256,
        preemption_mode=mode,
    )
    bm        = BlockManager(cache_cfg)
    scheduler = Scheduler(sched_cfg, bm)

    for i in range(8):
        sp  = SamplingParams(max_tokens=12)
        seq = Sequence(i, list(range(4 + i)), sp)
        grp = SequenceGroup(request_id=f"req-{i}", sequences=[seq])
        scheduler.add_seq_group(grp)

    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"  GPU blocks: {gpu_blocks}  CPU blocks: {cpu_blocks}  mode: {mode}")
    print(f"{'='*55}")
    print(f"  {'Step':>4}  {'Run':>3}  {'Wait':>4}  {'Swap':>4}  Events")
    print(f"  {'-'*50}")

    swap_count     = 0
    preempt_count  = 0

    for step in range(40):
        out   = scheduler.schedule()
        stats = scheduler.stats

        events = []
        if out.blocks_to_swap_out:
            swap_count    += len(out.blocks_to_swap_out)
            preempt_count += 1
            events.append(f"PREEMPT→CPU ({len(out.blocks_to_swap_out)} blks)")
        if out.blocks_to_swap_in:
            events.append(f"RESTORE←CPU ({len(out.blocks_to_swap_in)} blks)")
        if out.preempted and mode == "recompute":
            preempt_count += 1
            events.append(f"RECOMPUTE ({len(out.preempted)} seq re-queued)")

        for seq in out.scheduled_seqs:
            if seq.seq_id not in out.prefill_seq_ids:
                seq.append_token(99)
            if seq.check_stop():
                seq.status = SequenceStatus.FINISHED_STOPPED

        scheduler.free_finished_seqs()

        print(f"  {step:4d}  {stats['running']:3d}  {stats['waiting']:4d}"
              f"  {stats['swapped']:4d}  {', '.join(events) if events else ''}")

        if not scheduler.has_unfinished_seqs():
            print(f"\n  Done at step {step}. Preemptions: {preempt_count}")
            break


# ── Mode 1: swap (KV moved to CPU) ────────────────────────────────────────────
run_simulation(
    mode="swap", gpu_blocks=10, cpu_blocks=12,
    label="Swap Mode — KV blocks moved GPU→CPU on preemption"
)

# ── Mode 2: recompute (KV dropped, re-prefill) ────────────────────────────────
run_simulation(
    mode="recompute", gpu_blocks=10, cpu_blocks=4,
    label="Recompute Mode — KV dropped, sequence re-prefills from scratch"
)

# ── Mode 3: swap with tiny CPU pool → falls back to recompute ─────────────────
run_simulation(
    mode="swap", gpu_blocks=10, cpu_blocks=2,
    label="Swap with tiny CPU pool — falls back to recompute when CPU full"
)

print()
print("Takeaway:")
print("  Swap:      cheaper on reschedule, expensive on preemption (copy BW)")
print("  Recompute: cheaper on preemption, expensive on reschedule (re-prefill)")
print("  Fallback:  our scheduler gracefully degrades swap→recompute when needed")
