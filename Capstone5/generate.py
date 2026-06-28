"""
CLI — Generate text with the mini vLLM engine.

Usage:
  cd Capstone5
  python generate.py --prompt "Hello world" --max-tokens 100
  python generate.py --prompt "def fibonacci" --temperature 0.7 --top-k 50
  python generate.py --stream --prompt "Once upon a time"
"""

import argparse
import sys

import torch

from engine.llm_engine import LLMEngine
from model.config import CacheConfig, ModelConfig, SamplingParams, SchedulerConfig


class ByteTokenizer:
    vocab_size = 256
    def encode(self, text): return list(text.encode("utf-8"))
    def decode(self, ids):
        try: return bytes([max(0, min(255, i)) for i in ids]).decode("utf-8", errors="replace")
        except: return ""


def main():
    parser = argparse.ArgumentParser(description="Mini vLLM generation CLI")
    parser.add_argument("--prompt",       required=True)
    parser.add_argument("--max-tokens",   type=int,   default=100)
    parser.add_argument("--temperature",  type=float, default=0.8)
    parser.add_argument("--top-k",        type=int,   default=50)
    parser.add_argument("--top-p",        type=float, default=0.95)
    parser.add_argument("--num-requests", type=int,   default=1,
                        help="Send the same prompt N times (throughput test)")
    parser.add_argument("--stream",       action="store_true")
    parser.add_argument("--model",        default="nano",
                        choices=["nano", "gpt2-small"])
    parser.add_argument("--num-gpu-blocks", type=int, default=512)
    args = parser.parse_args()

    # ── Setup ──────────────────────────────────────────────────────────────────
    tok = ByteTokenizer()

    if args.model == "nano":
        model_config = ModelConfig.nano()
        model_config.vocab_size = tok.vocab_size
    else:
        model_config = ModelConfig.gpt2_small()

    cache_config     = CacheConfig(block_size=16, num_gpu_blocks=args.num_gpu_blocks)
    scheduler_config = SchedulerConfig(max_num_seqs=max(1, args.num_requests), max_num_batched_tokens=4096)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    engine = LLMEngine.from_config(model_config, cache_config, scheduler_config, tok, device=device)

    sp = SamplingParams(
        temperature = args.temperature,
        top_k       = args.top_k,
        top_p       = args.top_p,
        max_tokens  = args.max_tokens,
    )

    # ── Generate ───────────────────────────────────────────────────────────────
    if args.stream and args.num_requests == 1:
        print(f"Prompt: {args.prompt!r}")
        print("Output: ", end="", flush=True)
        for tok_text in engine.stream(args.prompt, sp):
            print(tok_text, end="", flush=True)
        print()
        return

    import time
    for i in range(args.num_requests):
        engine.add_request(f"req-{i}", args.prompt, sp)

    start = time.monotonic()
    total_tokens = 0
    results = {}

    while engine.has_unfinished_requests():
        outputs = engine.step()
        for out in outputs:
            if out.finished:
                results[out.request_id] = out.outputs[0].text
                total_tokens += len(out.outputs[0].token_ids)

    elapsed = time.monotonic() - start

    if args.num_requests == 1:
        print(f"\nPrompt: {args.prompt!r}")
        print(f"Output: {results.get('req-0', '')!r}")
    else:
        print(f"\nCompleted {args.num_requests} requests")
        print(f"Total output tokens: {total_tokens}")
        print(f"Elapsed: {elapsed:.2f}s")
        print(f"Throughput: {total_tokens / elapsed:.0f} tokens/sec")
        print(f"\nFirst response: {list(results.values())[0]!r}")


if __name__ == "__main__":
    main()
