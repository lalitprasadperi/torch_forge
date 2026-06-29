"""
Example 09 — Throughput Sweep (Batch Size vs Tokens/Sec)

Sweeps batch size from 1 to 16 and plots throughput.
Demonstrates the memory-bound → compute-bound transition:
  - At small batches: GPU underutilised (memory-bound decode)
  - At large batches: throughput plateaus (compute or KV-bandwidth bound)

Run:  python examples/09_throughput_sweep.py
"""
import sys; sys.path.insert(0, ".")
import time
import torch
from engine.llm_engine import LLMEngine
from model.config import ModelConfig, CacheConfig, SchedulerConfig, SamplingParams


class ByteTokenizer:
    vocab_size = 256
    def encode(self, t): return list(t.encode("utf-8"))
    def decode(self, ids): return bytes([max(0,min(255,i)) for i in ids]).decode("utf-8","replace")


def measure(batch_size, n_requests=32, prompt_len=32, output_len=64):
    tok = ByteTokenizer()
    mc  = ModelConfig.nano(); mc.vocab_size = tok.vocab_size
    engine = LLMEngine.from_config(
        mc,
        CacheConfig(block_size=16, num_gpu_blocks=1024),
        SchedulerConfig(max_num_seqs=batch_size, max_num_batched_tokens=4096),
        tok,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    sp = SamplingParams(temperature=0.0, max_tokens=output_len)
    for i in range(n_requests):
        engine.add_request(f"req-{i}", "x" * prompt_len, sp)

    # Warmup: let the first batch settle
    for _ in range(3):
        if engine.has_unfinished_requests():
            engine.step()

    start = time.monotonic()
    total = 0
    while engine.has_unfinished_requests():
        for o in engine.step():
            if o.finished:
                total += len(o.outputs[0].token_ids)

    elapsed = max(time.monotonic() - start, 1e-6)
    return total / elapsed


BATCH_SIZES  = [1, 2, 4, 8, 12, 16]
PROMPT_LEN   = 32
OUTPUT_LEN   = 64
N_REQUESTS   = 32

print("=" * 55)
print("Throughput Sweep — Batch Size vs Tokens/Sec")
print(f"Prompt={PROMPT_LEN} tokens, Output={OUTPUT_LEN} tokens, N={N_REQUESTS} requests")
print("=" * 55)
print()

results = {}
for bs in BATCH_SIZES:
    tps = measure(bs, N_REQUESTS, PROMPT_LEN, OUTPUT_LEN)
    results[bs] = tps

# ── Text chart ─────────────────────────────────────────────────────────────────
max_tps = max(results.values())
print(f"  {'Batch':>6}  {'Tokens/s':>10}  Bar")
print(f"  {'-'*45}")
for bs, tps in results.items():
    bar_len = int(tps / max_tps * 30)
    bar     = "█" * bar_len
    print(f"  {bs:6d}  {tps:10.0f}  {bar}")

print()
bs1   = BATCH_SIZES[0]
bsmax = BATCH_SIZES[-1]
speedup = results[bsmax] / results[bs1]
print(f"  Speedup (batch={bsmax} vs batch={bs1}): {speedup:.1f}×")
print()
print("  Expected pattern:")
print("  • Small batches: throughput scales near-linearly with batch size")
print("    (each new sequence adds nearly one extra token per step)")
print("  • Large batches: throughput growth slows — either:")
print("    - KV cache reads saturate HBM bandwidth (memory-bound), or")
print("    - Attention FLOPs saturate compute (compute-bound)")
print("  • Very large batches may OOM if block pool too small")
