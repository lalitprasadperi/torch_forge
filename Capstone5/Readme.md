# Capstone 5 — Build a Mini vLLM from Scratch

Build a complete LLM inference engine implementing the core innovations
of [vLLM](https://arxiv.org/abs/2309.06180) (Kwon et al., 2023):
paged KV cache, continuous batching, and efficient token scheduling.

---

## Architecture Overview

```
Request (text) ──► LLMEngine ──► Scheduler ──► ModelRunner ──► GPU
                      │              │               │
                      │         BlockManager     KV Caches
                      │              │           (paged blocks)
                      ▼              ▼
                  RequestOutput   Sequence state machine
                  (streaming)
```

### Core Concepts

| Concept | Analogy | File |
|---------|---------|------|
| KV Cache | CPU cache for transformer computation | `model/paged_attention.py` |
| Physical blocks | OS page frames | `engine/kv_cache.py` |
| Block table | OS page table | `engine/sequence.py` |
| Continuous batching | OS scheduling | `engine/scheduler.py` |
| Swap in/out | Demand paging | `engine/kv_cache.py` |

---

## Directory Structure

```
Capstone5/
├── engine/
│   ├── sequence.py        ← Sequence, SequenceGroup, state machine
│   ├── kv_cache.py        ← BlockManager, PhysicalBlock, BlockAllocator
│   ├── scheduler.py       ← Continuous batching, preemption
│   ├── model_runner.py    ← Batch prep, prefill/decode dispatch
│   └── llm_engine.py      ← Main engine: add_request → step → output
├── model/
│   ├── config.py          ← ModelConfig, CacheConfig, SchedulerConfig
│   ├── gpt.py             ← PagedGPT transformer
│   └── paged_attention.py ← Paged attention: write + gather + attend
├── kernels/
│   └── paged_attn_triton.py  ← Triton paged attention kernel
├── sampling/
│   └── sampler.py         ← Greedy, top-k, top-p, rep penalty
├── server/
│   ├── protocol.py        ← OpenAI-compatible Pydantic schemas
│   └── api_server.py      ← FastAPI server (streaming + chat)
├── benchmarks/
│   ├── throughput.py      ← Tokens/sec vs batch size
│   └── latency.py         ← TTFT, TPOT, E2E latency
├── quantization/
│   └── int8_quant.py      ← INT8/NF4 weight + KV cache quantization
├── tours/
│   ├── kv_cache_tour.py   ← Why paging? Fragmentation → PagedAttention
│   ├── scheduler_tour.py  ← Static → continuous batching, preemption
│   └── inference_tour.py  ← Prefill vs decode, roofline, full demo
└── generate.py            ← CLI: generate text
```

---

## Week-by-Week Learning Plan

### Week 1 — Core Data Structures
```bash
python tours/kv_cache_tour.py       # KV cache concepts
python -c "
import sys; sys.path.insert(0, '.')
from engine.sequence import Sequence
from model.config import SamplingParams
from engine.kv_cache import BlockManager
from model.config import CacheConfig

bm = BlockManager(CacheConfig(block_size=4, num_gpu_blocks=20))
seq = Sequence(0, [1,2,3,4,5], SamplingParams())
bm.allocate(seq)
print('blocks:', seq.block_table)
bm.append_slot(seq)
print('after append:', seq.block_table)
"
```

**Study:** `engine/sequence.py`, `engine/kv_cache.py`, `model/config.py`

### Week 2 — Scheduler and Continuous Batching
```bash
python tours/scheduler_tour.py
```

**Study:** `engine/scheduler.py` — especially `schedule()`, preemption logic

### Week 3 — The Model and Paged Attention
```bash
cd Capstone5
# Smoke test the model
python -m model.gpt

# Verify paged attention is numerically correct
python -c "
import sys; sys.path.insert(0, '.')
import torch, math
from model.paged_attention import paged_attention_decode
...
"
```

**Study:** `model/gpt.py`, `model/paged_attention.py`

### Week 4 — Full Engine, Benchmarks, Quantization
```bash
# Run the engine end to end
python generate.py --prompt "Hello world" --max-tokens 50 --stream

# Throughput benchmark
python -m benchmarks.throughput --batch-sizes 1 4 8 16

# Latency benchmark
python -m benchmarks.latency --prompt-lens 16 64 256

# Quantization experiments
python -m quantization.int8_quant
```

### Week 5 — Triton Kernel, API Server, Extensions
```bash
# Triton paged attention (requires CUDA)
python kernels/paged_attn_triton.py

# API server
pip install fastapi uvicorn
python server/api_server.py --model nano

# Test the API
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "nano", "prompt": "Hello", "max_tokens": 50}'
```

---

## Key Design Decisions

### 1. Block Size = 16 tokens
Each physical block stores 16 consecutive tokens of KV for one layer.
Larger blocks → less fragmentation overhead, but coarser allocation.
vLLM default: 16. Llama.cpp: 32.

### 2. Separate Prefill and Decode Passes
Our ModelRunner runs prefill sequences first, then decode sequences,
in separate forward passes. vLLM mixes them in one batched pass using
a prefix sum (cumulative position) trick — an exercise for the reader.

### 3. PyTorch Reference + Triton Kernel
`model/paged_attention.py` has a correct pure-PyTorch implementation.
`kernels/paged_attn_triton.py` has the fused Triton version.
Use PyTorch for debugging, Triton for production.

### 4. Preemption = Recompute by Default
Recompute mode requires no CPU memory and has simpler code.
Swap mode requires CPU cache tensors but avoids re-running prefill.
Toggle with `SchedulerConfig(preemption_mode="swap")`.

---

## Performance Expectations (RTX PRO 2000, fp16)

| Config | Throughput |
|--------|-----------|
| nano model, batch=1  | ~200 tokens/sec |
| nano model, batch=8  | ~1,200 tokens/sec |
| nano model, batch=16 | ~2,000 tokens/sec |

**Bottleneck at batch=1:** memory-bound (KV cache reads dominate)
**Bottleneck at large batch:** compute-bound (attention FLOPs)

---

## Concepts Covered

| Topic | Where |
|-------|-------|
| KV cache derivation | `tours/kv_cache_tour.py` Lesson 1 |
| PagedAttention memory savings | `tours/kv_cache_tour.py` Lesson 3–4 |
| Block table mechanics | `engine/kv_cache.py` |
| Continuous batching | `tours/scheduler_tour.py` Lesson 2–3 |
| Preemption (swap vs recompute) | `tours/scheduler_tour.py` Lesson 4 |
| Prefill vs decode | `tours/inference_tour.py` Lesson 1 |
| Roofline model | `tours/inference_tour.py` Lesson 2 |
| Online softmax (Triton) | `kernels/paged_attn_triton.py` |
| INT8 weight quantization | `quantization/int8_quant.py` |
| NF4 quantization (QLoRA) | `quantization/int8_quant.py` |
| Streaming generation | `engine/llm_engine.py` `stream()` |
| OpenAI-compatible API | `server/api_server.py` |
| TTFT / TPOT metrics | `benchmarks/latency.py` |

---

## Further Extensions

- **Chunked prefill**: split long prompts into `max_chunk_size` pieces per step
- **Prefix caching**: hash prompt prefixes, reuse shared blocks (CoW)
- **Speculative decoding**: draft model proposes tokens, target model verifies in parallel
- **Tensor parallelism**: shard attention heads across multiple GPUs
- **FlashAttention v2**: replace `F.scaled_dot_product_attention` with custom Triton
- **Beam search**: use `SequenceGroup.fork()` to branch from one prompt
- **LoRA serving**: hot-swap adapter weights per request

---

## References

- vLLM paper: [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- Orca: [A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu)
- Flash Attention: [Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- QLoRA / NF4: [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- Sarathi (chunked prefill): [Sarathi: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills](https://arxiv.org/abs/2308.16369)
