"""
Inference Tour — Prefill, Decode, and the Full Engine

Lessons:
  1. Prefill vs decode: two very different computations
  2. Memory-bound vs compute-bound: where the time goes
  3. Arithmetic intensity and the roofline model
  4. The full engine loop end-to-end
  5. Streaming generation
  6. Paged attention in practice (verify correctness)
  7. GPU memory anatomy: weights + KV + activations

Run:
  cd Capstone5
  python tours/inference_tour.py
"""

import sys
sys.path.insert(0, "/home/jmd/Desktop/TrainHard/LearnTorch/Capstone5")

import math
import time
from typing import List

import torch

from engine.llm_engine import LLMEngine
from model.config import CacheConfig, ModelConfig, SamplingParams, SchedulerConfig
from model.paged_attention import paged_attention_decode, write_to_cache


DIVIDER = "=" * 65


def lesson1_prefill_vs_decode():
    print(DIVIDER)
    print("Lesson 1: Prefill vs Decode — Two Very Different Computations")
    print(DIVIDER)
    print("""
LLM inference has TWO distinct phases:

  PREFILL (prompt processing):
    Input:   T prompt tokens
    Compute: full causal self-attention over T tokens — O(T²) ops
    Output:  1 new token (the first generation)
    KV write: T tokens written to cache

    Properties:
      • High arithmetic intensity (many FLOPs per byte loaded)
      • COMPUTE-BOUND on modern GPUs
      • Parallelisable over the T tokens
      • Runs once per request

  DECODE (token generation):
    Input:   1 new token
    Compute: attention over ALL cached tokens — O(T) ops
    Output:  1 new token
    KV write: 1 token appended to cache

    Properties:
      • Low arithmetic intensity (1 new FLOP per cached byte)
      • MEMORY-BOUND: limited by KV cache read bandwidth
      • Cannot be parallelised (each token depends on the previous)
      • Runs once per OUTPUT token

  This fundamental asymmetry is why continuous batching matters:
    Prefill  → few steps, high compute, parallelise across batch
    Decode   → many steps, memory-bound, limited by HBM bandwidth
""")


def lesson2_arithmetic_intensity():
    print(DIVIDER)
    print("Lesson 2: Arithmetic Intensity — Why Decode is Memory-Bound")
    print(DIVIDER)
    print("""
ARITHMETIC INTENSITY (AI) = FLOPs / bytes of memory traffic

ROOFLINE MODEL:
  If AI < ridge_point → MEMORY-BOUND (throughput = bandwidth × AI)
  If AI > ridge_point → COMPUTE-BOUND (throughput = peak TFLOPS)

For RTX PRO 2000 (approx):
  Peak FP16 TFLOPS: ~40 TFLOPS
  HBM Bandwidth:    ~448 GB/s
  Ridge point: 40e12 / 448e9 ≈ 89 FLOPs/byte
""")

    ridge_point = 89   # FLOP/byte for RTX PRO 2000 (approx)
    peak_tflops = 40
    bw_gb       = 448

    # Prefill case
    T, d = 512, 768   # seq_len, d_model
    # QKV matmul: 2 × T × d × 3d FLOPs, load weight (3d × d × 2 bytes)
    flops_prefill = 2 * T * d * 3 * d
    bytes_prefill = 3 * d * d * 2 + T * d * 2  # weight + activation
    ai_prefill = flops_prefill / bytes_prefill

    # Decode case
    T_cached = 512
    flops_decode = 2 * 1 * d * 3 * d       # same matmul, but T=1
    bytes_decode = 3 * d * d * 2 + 1 * d * 2
    ai_decode = flops_decode / bytes_decode

    # KV attention in decode: read all cached K,V
    n_heads, d_head = 12, 64
    flops_attn = 2 * T_cached * n_heads * d_head   # dot products
    bytes_attn = 2 * T_cached * n_heads * d_head * 2  # load K,V from HBM
    ai_attn = flops_attn / bytes_attn

    print(f"  QKV projection, prefill (T={T}):  AI = {ai_prefill:.0f} FLOPs/byte")
    print(f"  QKV projection, decode  (T=1):    AI = {ai_decode:.0f} FLOPs/byte")
    print(f"  KV attention,   decode (ctx={T_cached}): AI = {ai_attn:.1f} FLOPs/byte")
    print(f"  Ridge point:    {ridge_point} FLOPs/byte")
    print()

    for name, ai in [("Prefill QKV", ai_prefill), ("Decode QKV", ai_decode), ("Decode attention", ai_attn)]:
        if ai > ridge_point:
            bound = "COMPUTE-BOUND"
            tput  = peak_tflops * 1e12 / 1e12  # TFLOPS
            print(f"  {name}: {bound} (AI={ai:.0f} > {ridge_point}) → {tput:.0f} TFLOPS achievable")
        else:
            bound = "MEMORY-BOUND"
            tput  = bw_gb * ai / 1e9  # TFLOPS
            print(f"  {name}: {bound} (AI={ai:.1f} < {ridge_point}) → {tput:.1f} TFLOPS effective")

    print("""
  TAKEAWAY:
    Decode is memory-bound. Bigger batches amortise weight reads.
    At batch_size=1: ~1 TFLOP effective (vs 40 peak)
    At batch_size=64: weights read once, 64 tokens generated → 25× more efficient.
    This is why vLLM's continuous batching matters so much.
""")


def lesson3_full_engine_demo():
    print(DIVIDER)
    print("Lesson 3: Full Engine Loop — End to End")
    print(DIVIDER)

    class ByteTokenizer:
        vocab_size = 256
        def encode(self, text): return list(text.encode("utf-8"))
        def decode(self, ids):
            try: return bytes(ids).decode("utf-8", errors="replace")
            except: return ""

    model_config = ModelConfig.nano()
    model_config.vocab_size = 256
    cache_config     = CacheConfig(block_size=16, num_gpu_blocks=256, num_cpu_blocks=64)
    scheduler_config = SchedulerConfig(max_num_seqs=8, max_num_batched_tokens=1024)
    tok = ByteTokenizer()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    engine = LLMEngine.from_config(model_config, cache_config, scheduler_config, tok, device=device)
    print(f"  Engine ready on {device}")

    # Add 3 requests
    prompts = ["Hello", "The meaning of life", "def fibonacci"]
    params  = [
        SamplingParams(temperature=0.0, max_tokens=20),
        SamplingParams(temperature=0.8, max_tokens=30),
        SamplingParams(temperature=1.0, top_k=10, max_tokens=25),
    ]

    for i, (p, sp) in enumerate(zip(prompts, params)):
        engine.add_request(f"req-{i}", p, sp)
    print(f"  Added {len(prompts)} requests")

    # Run the engine loop
    print("\n  Running engine loop:")
    step = 0
    while engine.has_unfinished_requests():
        outputs = engine.step()
        step += 1
        for out in outputs:
            if out.finished:
                text = out.outputs[0].text
                print(f"    [step {step:3d}] {out.request_id} DONE: {repr(text[:40])}")

    stats = engine.stats
    print(f"\n  Engine stats:")
    print(f"    Total steps:   {stats['steps']}")
    print(f"    Total tokens:  {stats['total_tokens']}")
    print(f"    Throughput:    {stats['tokens_per_sec']:.0f} tokens/sec")


def lesson4_streaming():
    print(DIVIDER)
    print("Lesson 4: Streaming Generation")
    print(DIVIDER)
    print("  Streaming: each token is returned as it's generated.")
    print("  This is how ChatGPT feels 'live' — you see each word appear.")
    print()

    class ByteTokenizer:
        vocab_size = 256
        def encode(self, text): return list(text.encode("utf-8"))
        def decode(self, ids):
            try: return bytes(ids).decode("utf-8", errors="replace")
            except: return ""

    model_config = ModelConfig.nano()
    model_config.vocab_size = 256
    cache_config     = CacheConfig(block_size=16, num_gpu_blocks=256, num_cpu_blocks=64)
    scheduler_config = SchedulerConfig(max_num_seqs=4, max_num_batched_tokens=512)
    tok = ByteTokenizer()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    engine = LLMEngine.from_config(model_config, cache_config, scheduler_config, tok, device=device)

    print("  Streaming output: ", end="", flush=True)
    sp = SamplingParams(temperature=0.0, max_tokens=40)
    for token_text in engine.stream("Hello world", sp):
        print(token_text, end="", flush=True)
    print()
    print("  (random weights → random output, but the plumbing works!)")


def lesson5_paged_attention_correctness():
    print(DIVIDER)
    print("Lesson 5: Paged Attention — Verifying Correctness")
    print(DIVIDER)
    print("""
  Core claim: paged attention (gathering K,V from scattered blocks)
  gives the SAME result as standard attention over a contiguous buffer.

  Let's verify this:
    1. Fill contiguous KV buffer with random values
    2. Copy the same values into paged blocks
    3. Run standard attention vs paged attention
    4. Compare outputs (should match within fp16 tolerance)
""")

    torch.manual_seed(0)
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, H, D    = 2, 4, 32
    seq_len    = 32
    block_size = 8
    scale      = 1.0 / math.sqrt(D)
    num_blocks = 32

    # Contiguous KV (standard approach)
    k_cont = torch.randn(B, seq_len, H, D, device=device, dtype=torch.float32)
    v_cont = torch.randn(B, seq_len, H, D, device=device, dtype=torch.float32)

    # Paged KV cache: (num_blocks, 2, block_size, H, D)
    kv_cache    = torch.zeros(num_blocks, 2, block_size, H, D, device=device)
    block_tables = torch.zeros(B, seq_len // block_size + 1, dtype=torch.int32, device=device)
    seq_lens     = torch.tensor([seq_len, seq_len], dtype=torch.int32, device=device)

    # Assign blocks and copy KV data
    for i in range(B):
        n_blocks = seq_len // block_size
        assigned = list(range(i * n_blocks, (i + 1) * n_blocks))
        block_tables[i, :n_blocks] = torch.tensor(assigned, dtype=torch.int32)
        for blk_idx, blk_id in enumerate(assigned):
            start = blk_idx * block_size
            kv_cache[blk_id, 0] = k_cont[i, start:start + block_size]  # K
            kv_cache[blk_id, 1] = v_cont[i, start:start + block_size]  # V

    # Standard attention (B, H, D query attending over seq_len)
    q = torch.randn(B, H, D, device=device)

    # Standard: full contiguous attention
    k_h  = k_cont.permute(0, 2, 1, 3)   # (B, H, seq_len, D)
    v_h  = v_cont.permute(0, 2, 1, 3)
    q_h  = q.unsqueeze(2)                # (B, H, 1, D)
    scores_ref = torch.matmul(q_h, k_h.transpose(-1, -2)) * scale
    out_ref    = torch.matmul(torch.softmax(scores_ref, dim=-1), v_h).squeeze(2)  # (B, H, D)

    # Paged attention
    out_paged = paged_attention_decode(q, kv_cache, block_tables, seq_lens, scale)

    diff = (out_ref - out_paged).abs().max().item()
    print(f"  Max diff (standard vs paged): {diff:.2e}")
    assert diff < 0.01, f"Too large: {diff}"
    print("  ✓ Paged attention matches standard attention")


def lesson6_gpu_memory_anatomy():
    print(DIVIDER)
    print("Lesson 6: GPU Memory Anatomy — Where Does It All Go?")
    print(DIVIDER)

    if not torch.cuda.is_available():
        print("  (CUDA not available — showing estimates)")
        gpu_gb = 16.0
    else:
        gpu_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Total memory: {gpu_gb:.1f} GB")

    # GPT-2 small estimates (fp16)
    vocab, d, H, n_heads, n_layers = 50257, 768, 64, 12, 12
    d_ff = 3072

    embed_bytes    = vocab * d * 2
    attn_bytes     = n_layers * 4 * d * d * 2   # QKV + out proj
    ffn_bytes      = n_layers * 2 * d * d_ff * 2
    norm_bytes     = n_layers * 2 * d * 4
    total_weights  = embed_bytes + attn_bytes + ffn_bytes + norm_bytes

    print(f"\n  GPT-2 Small (fp16):")
    print(f"    Embedding:   {embed_bytes / 1e6:.0f} MB")
    print(f"    Attention:   {attn_bytes / 1e6:.0f} MB")
    print(f"    FFN:         {ffn_bytes / 1e6:.0f} MB")
    print(f"    Total model: {total_weights / 1e6:.0f} MB")

    # KV cache
    available_for_kv = gpu_gb * 0.85 * 1e9 - total_weights
    block_size = 16
    per_block_per_layer = 2 * block_size * n_heads * H * 2
    per_block = per_block_per_layer * n_layers
    num_blocks = int(available_for_kv / per_block)

    print(f"\n  Available for KV cache: {available_for_kv / 1e9:.1f} GB")
    print(f"  Per block (all layers):  {per_block / 1024:.1f} KB")
    print(f"  Max KV blocks:           {num_blocks}")
    print(f"  Max tokens (block_size={block_size}): {num_blocks * block_size:,}")

    # Activations (rough estimate)
    batch_size = 32
    max_len    = 2048
    activ_per_token = d * n_layers * 4 * 2  # rough: 4 tensors × d × fp16
    activ_total     = batch_size * max_len * activ_per_token
    print(f"\n  Activations (batch={batch_size}, len={max_len}): {activ_total / 1e9:.2f} GB")
    print()
    print("  Summary:")
    print(f"    Model weights:    {total_weights / 1e9:.2f} GB")
    print(f"    KV cache:         {available_for_kv / 1e9:.1f} GB")
    print(f"    Activations:      {activ_total / 1e9:.2f} GB (cleared after each step)")


def main():
    lesson1_prefill_vs_decode()
    lesson2_arithmetic_intensity()
    lesson3_full_engine_demo()
    lesson4_streaming()
    lesson5_paged_attention_correctness()
    lesson6_gpu_memory_anatomy()

    print(DIVIDER)
    print("Summary")
    print(DIVIDER)
    print("""
  Prefill: compute-bound, runs once per request.
  Decode:  memory-bound, runs once per output token.
  Paged attention: mathematically identical to standard attention.
  Continuous batching: saturates GPU during decode.
  KV cache dominates GPU memory for large contexts.
  Streaming: yield each token as it's decoded.

  You have now built every component of a production inference engine!
""")


if __name__ == "__main__":
    main()
