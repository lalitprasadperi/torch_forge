"""
Throughput Benchmark — Tokens Per Second at Different Batch Sizes

Measures the system throughput (total output tokens / wall-clock time)
for different batch sizes and prompt lengths.

KEY METRICS:
  • Throughput (tokens/sec): higher is better, measures server capacity
  • GPU utilisation: should stay near 100% during decode
  • Batch efficiency: continuous vs static batching comparison

EXPECTED BEHAVIOUR:
  Larger batch → more tokens per step → better GPU utilisation
  But: larger batch needs more KV blocks → less max seq length
  Sweet spot depends on model size and GPU memory.

Run:
  cd Capstone5
  python -m benchmarks.throughput --batch-sizes 1 4 8 16 --seq-len 128
"""

import argparse
import sys
import time
from typing import List

import torch

sys.path.insert(0, "/home/jmd/Desktop/TrainHard/LearnTorch/Capstone5")

from engine.llm_engine import LLMEngine
from model.config import CacheConfig, ModelConfig, SamplingParams, SchedulerConfig


class DummyTokenizer:
    """Simple tokenizer for benchmark: char → ordinal."""
    def encode(self, text: str) -> List[int]:
        return [ord(c) % 256 for c in text]
    def decode(self, ids: List[int]) -> str:
        return "".join(chr(max(32, i)) for i in ids)


def make_engine(batch_size: int, model_config: ModelConfig) -> LLMEngine:
    cache_config     = CacheConfig(block_size=16, num_gpu_blocks=512, num_cpu_blocks=64)
    scheduler_config = SchedulerConfig(
        max_num_seqs           = batch_size,
        max_num_batched_tokens = 4096,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return LLMEngine.from_config(model_config, cache_config, scheduler_config,
                                  DummyTokenizer(), device=device)


def run_benchmark(
    engine:       LLMEngine,
    num_requests: int,
    prompt_len:   int,
    output_len:   int,
) -> dict:
    prompt = "x" * prompt_len
    sp     = SamplingParams(temperature=0.0, max_tokens=output_len)

    for i in range(num_requests):
        engine.add_request(f"req-{i}", prompt, sp)

    start_time    = time.monotonic()
    total_tokens  = 0
    step_count    = 0

    while engine.has_unfinished_requests():
        outputs = engine.step()
        for out in outputs:
            if out.finished:
                total_tokens += len(out.outputs[0].token_ids)
        step_count += 1

    elapsed = time.monotonic() - start_time
    return {
        "total_tokens":   total_tokens,
        "elapsed_sec":    elapsed,
        "throughput":     total_tokens / elapsed,
        "avg_latency_ms": elapsed / num_requests * 1000,
        "steps":          step_count,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--prompt-len",  type=int, default=64)
    parser.add_argument("--output-len",  type=int, default=128)
    parser.add_argument("--num-requests", type=int, default=32)
    args = parser.parse_args()

    model_config = ModelConfig.nano()
    model_config.vocab_size = 256

    print("=" * 65)
    print("Mini vLLM Throughput Benchmark")
    print("=" * 65)
    print(f"Prompt length:  {args.prompt_len} tokens")
    print(f"Output length:  {args.output_len} tokens")
    print(f"Num requests:   {args.num_requests}")
    print()
    print(f"{'Batch':>6}  {'Throughput':>14}  {'Avg latency':>12}  {'Steps':>6}")
    print("-" * 45)

    for batch_size in args.batch_sizes:
        engine = make_engine(batch_size, model_config)
        result = run_benchmark(engine, args.num_requests, args.prompt_len, args.output_len)
        print(f"{batch_size:>6}  {result['throughput']:>11.0f} t/s"
              f"  {result['avg_latency_ms']:>9.0f} ms"
              f"  {result['steps']:>6}")


if __name__ == "__main__":
    main()
