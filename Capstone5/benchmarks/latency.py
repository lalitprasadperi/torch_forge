"""
Latency Benchmark — TTFT, TPOT, and End-to-End Latency

For LLM serving, there are two distinct latency metrics:

  TTFT (Time To First Token):
    Time from request submission to the FIRST generated token.
    Dominated by PREFILL: the cost of computing KV for the whole prompt.
    User-visible as "response lag" before text starts appearing.

  TPOT (Time Per Output Token):
    Average time between each successive output token.
    Dominated by DECODE: one token per sequence per step.
    User-visible as streaming "typing speed".

  E2E Latency = TTFT + (output_len - 1) × TPOT

THROUGHPUT vs LATENCY TRADEOFF:
  • Large batches maximise GPU utilisation → high throughput
  • But large batches increase TPOT (decode step slowed by more sequences)
  • Prefill for one request blocks the decode of others
    → chunked prefill breaks large prompts into smaller pieces

IDEAL BEHAVIOUR:
  TTFT should scale with prompt_len (compute-bound during prefill)
  TPOT should be roughly constant (memory-bound during decode)
  Large batches increase TPOT but improve throughput

Run:
  cd Capstone5
  python -m benchmarks.latency --prompt-lens 16 64 256 --output-len 128
"""

import argparse
import sys
import time
from typing import List

import torch

sys.path.insert(0, "/home/jmd/Desktop/TrainHard/LearnTorch/Capstone5")

from engine.llm_engine import LLMEngine
from engine.sequence import SequenceStatus
from model.config import CacheConfig, ModelConfig, SamplingParams, SchedulerConfig


class DummyTokenizer:
    def encode(self, text: str) -> List[int]:
        return [ord(c) % 256 for c in text]
    def decode(self, ids: List[int]) -> str:
        return "".join(chr(max(32, i)) for i in ids)


def measure_latency(
    model_config:  ModelConfig,
    prompt_len:    int,
    output_len:    int,
    batch_size:    int = 1,
    warmup_runs:   int = 2,
    measure_runs:  int = 5,
) -> dict:
    cache_config     = CacheConfig(block_size=16, num_gpu_blocks=512, num_cpu_blocks=64)
    scheduler_config = SchedulerConfig(
        max_num_seqs           = batch_size,
        max_num_batched_tokens = 4096,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    engine = LLMEngine.from_config(model_config, cache_config, scheduler_config,
                                    DummyTokenizer(), device=device)

    prompt = "x" * prompt_len
    sp     = SamplingParams(temperature=0.0, max_tokens=output_len)

    ttft_values: List[float] = []
    tpot_values: List[float] = []

    for run_idx in range(warmup_runs + measure_runs):
        request_id = f"req-{run_idx}"
        engine.add_request(request_id, prompt, sp)

        tokens_generated = 0
        first_token_time = None
        request_start    = time.perf_counter()
        step_times: List[float] = []

        while engine.has_unfinished_requests():
            step_start = time.perf_counter()
            outputs    = engine.step()
            step_end   = time.perf_counter()

            for out in outputs:
                if out.request_id != request_id:
                    continue
                curr_tokens = len(out.outputs[0].token_ids)
                if curr_tokens > tokens_generated:
                    if first_token_time is None:
                        first_token_time = step_end
                    step_times.append(step_end - step_start)
                    tokens_generated = curr_tokens
                if out.finished:
                    break

        if run_idx >= warmup_runs:
            ttft = (first_token_time - request_start) * 1000  # ms
            e2e  = (time.perf_counter() - request_start) * 1000
            tpot = (e2e - ttft) / max(output_len - 1, 1)
            ttft_values.append(ttft)
            tpot_values.append(tpot)

    return {
        "prompt_len": prompt_len,
        "output_len": output_len,
        "ttft_ms":    sum(ttft_values) / len(ttft_values),
        "tpot_ms":    sum(tpot_values) / len(tpot_values),
        "e2e_ms":     sum(ttft_values) / len(ttft_values) + sum(tpot_values) / len(tpot_values) * (output_len - 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-lens", type=int, nargs="+", default=[16, 64, 128])
    parser.add_argument("--output-len",  type=int, default=64)
    parser.add_argument("--batch-size",  type=int, default=1)
    args = parser.parse_args()

    model_config = ModelConfig.nano()
    model_config.vocab_size = 256

    print("=" * 65)
    print("Mini vLLM Latency Benchmark")
    print("=" * 65)
    print(f"Output length: {args.output_len} tokens")
    print(f"Batch size:    {args.batch_size}")
    print()
    print(f"{'Prompt':>8}  {'TTFT':>10}  {'TPOT':>10}  {'E2E':>10}")
    print("-" * 45)

    for prompt_len in args.prompt_lens:
        result = measure_latency(model_config, prompt_len, args.output_len, args.batch_size)
        print(f"{prompt_len:>8}  {result['ttft_ms']:>8.1f}ms"
              f"  {result['tpot_ms']:>8.2f}ms"
              f"  {result['e2e_ms']:>8.1f}ms")

    print()
    print("TTFT should increase with prompt length (compute-bound).")
    print("TPOT should stay roughly constant (memory-bound).")


if __name__ == "__main__":
    main()
