"""
Example 02 — generate() and stream() Convenience Methods

LLMEngine.generate() blocks until done and returns the full output.
LLMEngine.stream()   is a generator that yields each new token text.

Run:  python examples/02_generate_convenience.py
"""
import sys; sys.path.insert(0, ".")
import torch
from engine.llm_engine import LLMEngine
from model.config import ModelConfig, CacheConfig, SchedulerConfig, SamplingParams


class ByteTokenizer:
    vocab_size = 256
    def encode(self, t): return list(t.encode("utf-8"))
    def decode(self, ids): return bytes([max(0,min(255,i)) for i in ids]).decode("utf-8","replace")


def make_engine():
    tok          = ByteTokenizer()
    model_config = ModelConfig.nano()
    model_config.vocab_size = tok.vocab_size
    return LLMEngine.from_config(
        model_config,
        CacheConfig(block_size=16, num_gpu_blocks=256),
        SchedulerConfig(max_num_seqs=4, max_num_batched_tokens=512),
        tok,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )


engine = make_engine()

# ── Method 1: generate() — blocks until done ──────────────────────────────────
print("=== generate() — blocks until the full output is ready ===")
sp  = SamplingParams(temperature=0.0, max_tokens=20)
out = engine.generate("Hello world", sp)
print(f"  prompt:       {out.prompt!r}")
print(f"  output:       {out.outputs[0].text!r}")
print(f"  finish_reason: {out.outputs[0].finish_reason}")

# ── Method 2: stream() — yields token text as each is produced ────────────────
print("\n=== stream() — yields each token as it is decoded ===")
sp2 = SamplingParams(temperature=0.0, max_tokens=30)
print("  Streaming: ", end="", flush=True)
for fragment in engine.stream("The meaning of", sp2):
    print(fragment, end="", flush=True)
print()
print("  (random model weights produce garbled text, but plumbing is real)")
