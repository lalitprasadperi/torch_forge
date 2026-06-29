"""
Example 01 — Hello Engine

The absolute minimum to use the mini vLLM engine:
  1. Build an engine
  2. Add a request
  3. Step until done

Run:  python examples/01_hello_engine.py
"""
import sys; sys.path.insert(0, ".")
import torch
from engine.llm_engine import LLMEngine
from model.config import ModelConfig, CacheConfig, SchedulerConfig, SamplingParams


class ByteTokenizer:
    vocab_size = 256
    def encode(self, text): return list(text.encode("utf-8"))
    def decode(self, ids):
        return bytes([max(0, min(255, i)) for i in ids]).decode("utf-8", errors="replace")


# ── Build the engine ──────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok    = ByteTokenizer()

model_config     = ModelConfig.nano()
model_config.vocab_size = tok.vocab_size
cache_config     = CacheConfig(block_size=16, num_gpu_blocks=256)
scheduler_config = SchedulerConfig(max_num_seqs=4, max_num_batched_tokens=512)

engine = LLMEngine.from_config(model_config, cache_config, scheduler_config, tok, device=device)

# ── One request ───────────────────────────────────────────────────────────────
engine.add_request(
    request_id  = "my-first-request",
    prompt      = "Hello, world!",
    sampling_params = SamplingParams(temperature=0.0, max_tokens=30),
)

# ── Run until finished ────────────────────────────────────────────────────────
while engine.has_unfinished_requests():
    for out in engine.step():
        if out.finished:
            print(f"Prompt:  {out.prompt!r}")
            print(f"Output:  {out.outputs[0].text!r}")
            print(f"Tokens:  {out.outputs[0].token_ids}")
            print(f"Reason:  {out.outputs[0].finish_reason}")
