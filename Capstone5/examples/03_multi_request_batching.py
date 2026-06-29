"""
Example 03 — Multiple Concurrent Requests (Continuous Batching)

Shows that the engine processes multiple requests in parallel,
interleaving their decode steps rather than waiting for one to finish.

Key observation: shorter requests finish first, freeing slots for new ones.

Run:  python examples/03_multi_request_batching.py
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


def make_engine(max_seqs=8):
    tok = ByteTokenizer()
    mc  = ModelConfig.nano()
    mc.vocab_size = tok.vocab_size
    return LLMEngine.from_config(
        mc,
        CacheConfig(block_size=16, num_gpu_blocks=512),
        SchedulerConfig(max_num_seqs=max_seqs, max_num_batched_tokens=1024),
        tok,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )


engine = make_engine()

# Submit 5 requests with different output lengths
requests = [
    ("req-short",   "Hi",        SamplingParams(temperature=0.0, max_tokens=5)),
    ("req-medium",  "Hello",     SamplingParams(temperature=0.0, max_tokens=20)),
    ("req-long",    "Once upon", SamplingParams(temperature=0.0, max_tokens=40)),
    ("req-longer",  "def fib",   SamplingParams(temperature=0.0, max_tokens=50)),
    ("req-longest", "import",    SamplingParams(temperature=0.0, max_tokens=60)),
]

print(f"Submitting {len(requests)} requests simultaneously")
print("-" * 55)
for rid, prompt, sp in requests:
    engine.add_request(rid, prompt, sp)
    print(f"  {rid:<15}  prompt={prompt!r:10}  max_tokens={sp.max_tokens}")

print()
print(f"{'Step':>5}  {'Running':>8}  {'Finished this step'}")
print("-" * 55)

step         = 0
finished_all = []
start        = time.monotonic()

while engine.has_unfinished_requests():
    outputs = engine.step()
    step   += 1
    finished_this_step = [o for o in outputs if o.finished]

    if finished_this_step or step % 10 == 0:
        stats = engine.stats
        names = [o.request_id for o in finished_this_step]
        print(f"  {step:3d}  {stats['running']:8d}  {names if names else ''}")
        finished_all.extend(finished_this_step)

elapsed = time.monotonic() - start
print()
print(f"All done in {step} steps, {elapsed:.2f}s")
print(f"Throughput: {engine.stats['total_tokens'] / elapsed:.0f} tokens/sec")
print()
print("Completion order (continuous batching fills slots immediately):")
for i, out in enumerate(finished_all, 1):
    print(f"  {i}. {out.request_id:<15}  {len(out.outputs[0].token_ids)} tokens generated")
