"""
Example 08 — Continuous vs Static Batching Throughput

Simulates both strategies on the same workload and measures:
  - Total time to completion
  - GPU slot utilisation (useful tokens / total token-steps)
  - Throughput (tokens/sec)

Static batching: process N requests at a time, wait for ALL to finish.
Continuous batching: fill slots immediately when any request finishes.

Run:  python examples/08_continuous_vs_static.py
"""
import sys; sys.path.insert(0, ".")
import random
import time
import torch
from engine.llm_engine import LLMEngine
from model.config import ModelConfig, CacheConfig, SchedulerConfig, SamplingParams


class ByteTokenizer:
    vocab_size = 256
    def encode(self, t): return list(t.encode("utf-8"))
    def decode(self, ids): return bytes([max(0,min(255,i)) for i in ids]).decode("utf-8","replace")


def make_engine(max_seqs):
    tok = ByteTokenizer()
    mc  = ModelConfig.nano(); mc.vocab_size = tok.vocab_size
    return LLMEngine.from_config(
        mc,
        CacheConfig(block_size=16, num_gpu_blocks=512),
        SchedulerConfig(max_num_seqs=max_seqs, max_num_batched_tokens=2048),
        tok,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )


random.seed(7)
BATCH_SIZE   = 6
N_REQUESTS   = 18
PROMPT_LEN   = 20
OUTPUT_LENS  = [random.randint(5, 60) for _ in range(N_REQUESTS)]  # variable!

print("=" * 60)
print("Continuous vs Static Batching Comparison")
print("=" * 60)
print(f"Requests:    {N_REQUESTS}")
print(f"Batch size:  {BATCH_SIZE}")
print(f"Prompt len:  {PROMPT_LEN} tokens each")
print(f"Output lens: min={min(OUTPUT_LENS)}  max={max(OUTPUT_LENS)}  "
      f"mean={sum(OUTPUT_LENS)/len(OUTPUT_LENS):.0f}")
print()


# ── Static Batching Simulation ─────────────────────────────────────────────────
# Process BATCH_SIZE requests at a time; start next batch only when all done.
print("--- Static Batching ---")
engine_static = make_engine(BATCH_SIZE)
total_tokens_static = 0
total_steps_static  = 0
waste_steps_static  = 0

static_start = time.monotonic()
for batch_start in range(0, N_REQUESTS, BATCH_SIZE):
    batch = list(range(batch_start, min(batch_start + BATCH_SIZE, N_REQUESTS)))
    for i in batch:
        sp = SamplingParams(temperature=0.0, max_tokens=OUTPUT_LENS[i])
        engine_static.add_request(f"s-req-{i}", "x" * PROMPT_LEN, sp)

    batch_steps     = 0
    batch_tokens    = 0
    active_per_step = []

    while engine_static.has_unfinished_requests():
        outputs = engine_static.step()
        batch_steps += 1
        active_this_step = sum(1 for o in outputs if not o.finished or o.finished)
        active_per_step.append(engine_static.stats["running"])
        for o in outputs:
            if o.finished:
                batch_tokens += len(o.outputs[0].token_ids)

    # Waste = steps where some slots were idle (finished seqs held up the batch)
    max_per_step = max(active_per_step) if active_per_step else 1
    for active in active_per_step:
        waste_steps_static += max_per_step - active

    total_steps_static  += batch_steps
    total_tokens_static += batch_tokens
    print(f"  Batch {batch_start//BATCH_SIZE + 1}: {len(batch)} reqs, "
          f"{batch_steps} steps, {batch_tokens} tokens")

static_elapsed = time.monotonic() - static_start
static_throughput = total_tokens_static / static_elapsed


# ── Continuous Batching ────────────────────────────────────────────────────────
# All requests submitted up front; scheduler fills slots immediately.
print()
print("--- Continuous Batching ---")
engine_cont   = make_engine(BATCH_SIZE)
total_steps_c = 0

for i in range(N_REQUESTS):
    sp = SamplingParams(temperature=0.0, max_tokens=OUTPUT_LENS[i])
    engine_cont.add_request(f"c-req-{i}", "x" * PROMPT_LEN, sp)

cont_start = time.monotonic()
while engine_cont.has_unfinished_requests():
    engine_cont.step()
    total_steps_c += 1

cont_elapsed    = time.monotonic() - cont_start
cont_stats      = engine_cont.stats
cont_throughput = cont_stats["total_tokens"] / cont_elapsed

print(f"  {N_REQUESTS} requests, {total_steps_c} steps, "
      f"{cont_stats['total_tokens']} tokens")


# ── Comparison ─────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print(f"{'Metric':<30}  {'Static':>10}  {'Continuous':>10}")
print("-" * 55)
print(f"{'Total steps':<30}  {total_steps_static:>10d}  {total_steps_c:>10d}")
print(f"{'Total output tokens':<30}  {total_tokens_static:>10d}  {cont_stats['total_tokens']:>10d}")
print(f"{'Wall time (s)':<30}  {static_elapsed:>10.2f}  {cont_elapsed:>10.2f}")
print(f"{'Throughput (tok/s)':<30}  {static_throughput:>10.0f}  {cont_throughput:>10.0f}")
speedup = cont_throughput / static_throughput
print(f"\n  Continuous batching is {speedup:.2f}× faster")
print(f"  (higher with more variable output lengths)")
