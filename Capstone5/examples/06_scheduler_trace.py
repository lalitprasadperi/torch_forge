"""
Example 06 — Scheduler Trace

Prints a step-by-step trace of the scheduler's decisions:
  - Which sequences are being prefilled vs decoded
  - When sequences finish and free their slots
  - When new waiting requests are admitted
  - GPU block utilisation over time

Run:  python examples/06_scheduler_trace.py
"""
import sys; sys.path.insert(0, ".")
from engine.kv_cache import BlockManager
from engine.scheduler import Scheduler
from engine.sequence import Sequence, SequenceGroup, SequenceStatus
from model.config import CacheConfig, SamplingParams, SchedulerConfig


BLOCK_SIZE  = 4
GPU_BLOCKS  = 24
MAX_SEQS    = 3

cache_cfg     = CacheConfig(block_size=BLOCK_SIZE, num_gpu_blocks=GPU_BLOCKS, num_cpu_blocks=8)
sched_cfg     = SchedulerConfig(max_num_seqs=MAX_SEQS, max_num_batched_tokens=64)
bm            = BlockManager(cache_cfg)
scheduler     = Scheduler(sched_cfg, bm)

# Queue 5 requests with different lengths
requests = [
    ("A", 3,  6),   # (name, prompt_tokens, max_output_tokens)
    ("B", 5,  4),
    ("C", 2,  8),
    ("D", 4,  5),
    ("E", 6,  3),
]

sp_map = {}
for name, prompt_len, max_out in requests:
    sp  = SamplingParams(max_tokens=max_out)
    seq = Sequence(ord(name), list(range(prompt_len)), sp)
    grp = SequenceGroup(request_id=name, sequences=[seq])
    scheduler.add_seq_group(grp)
    sp_map[ord(name)] = max_out

print(f"{'Step':>4}  {'Prefill':^12}  {'Decode':^20}  {'Wait':>4}  {'Run':>3}  {'Blks':>4}")
print("-" * 60)

def bar(used, total, width=12):
    filled = int(used / total * width)
    return "█" * filled + "░" * (width - filled)

for step in range(30):
    out   = scheduler.schedule()
    stats = scheduler.stats

    prefill_names = [chr(s.seq_id) for s in out.scheduled_seqs if s.seq_id in out.prefill_seq_ids]
    decode_names  = [chr(s.seq_id) for s in out.scheduled_seqs if s.seq_id not in out.prefill_seq_ids]

    # Simulate: append one token per decode seq
    for seq in out.scheduled_seqs:
        if seq.seq_id not in out.prefill_seq_ids:
            seq.append_token(99)
        if seq.check_stop():
            seq.status = SequenceStatus.FINISHED_STOPPED

    scheduler.free_finished_seqs()
    finished = [chr(s.seq_id) for s in out.scheduled_seqs if s.is_finished()]

    blk_bar = bar(GPU_BLOCKS - stats["gpu_free"], GPU_BLOCKS, 8)
    note    = f"  ← {','.join(finished)} done" if finished else ""
    print(f"  {step:2d}  {','.join(prefill_names):^12}  {','.join(decode_names):^20}"
          f"  {stats['waiting']:4d}  {stats['running']:3d}  [{blk_bar}]{note}")

    if not scheduler.has_unfinished_seqs():
        print()
        print(f"  All requests completed at step {step}")
        break
