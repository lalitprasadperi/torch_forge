# 🔥 torch_forge

> *Build things. Measure things. Understand why they work.*

A hands-on PyTorch curriculum built as a series of **capstone projects** — each one a self-contained deep-dive into a core area of GPU computing and deep learning engineering.

The philosophy: **don't just read about PyTorch, build tools with it.** Every capstone produces something real that you can run, measure, and extend.

---

## 🗂️ Capstones

| # | Title | Topics | Status |
|---|-------|--------|--------|
| [**1**](Capstone1/) | 🔬 **PyTorch Performance Lab** | Tensors, Autograd, CUDA Streams, Benchmarking, Roofline Model | ✅ Complete |
| [**2**](Capstone2/) | 🏋️ **Mini Training Framework** | nn.Module, Trainer, DataLoader, AMP, Grad Accum, Checkpointing | ✅ Complete |
| [**3**](Capstone3/) | 🧠 **Rebuild the Transformer** | Attention, MHA, FFN, Residual, RoPE, RMSNorm, SwiGLU, FlashAttention, KV Cache | ✅ Complete |
| [**4**](Capstone4/) | ⚙️ **PyTorch Compiler Deep Dive** | torch.compile, TorchDynamo, FX Graph, AOTAutograd, Inductor, Triton Kernels, CUDA Graphs, Kernel Fusion | ✅ Complete |
| [**5**](Capstone5/) | 🚀 **Build a Mini vLLM** | PagedAttention, Continuous Batching, KV Block Manager, Prefill/Decode, Triton Decode Kernel, INT8/NF4 Quantization, Streaming, OpenAI API | ✅ Complete |

---

## 🔬 Capstone 1 — PyTorch Performance Lab

> **Goal:** Become completely comfortable with PyTorch tensors while building a GPU benchmarking framework that resembles an internal performance lab.

### What you build

A benchmarking library (`perf_lab/`) that measures every major neural network operation on your GPU and reports latency, TFLOPS, memory bandwidth, and arithmetic intensity — with a roofline model to tell you whether each op is compute-bound or memory-bound.

### Learning path

```
tensor_tour.py  →  autograd_demo.py  →  streams_demo.py  →  run_bench.py
     🧱                  🔁                   🌊                  📊
  Tensors &           Gradients &          CUDA Streams        Full Perf
  Memory Model        Backprop             & Timing            Benchmark
```

### Operations benchmarked

| Op | Shape | Bound |
|----|-------|-------|
| MatMul | `(M,K) @ (K,N)` | 🔥 Compute (large) |
| Conv2D | `(N,C,H,W) * kernel` | ⚡ Mixed |
| LayerNorm | `(B,T,D) → norm` | 💧 Memory |
| Softmax | `(B,H,T,T) → probs` | 💧 Memory |
| GELU | `(B,T,D) → act` | 💧 Memory |
| RMSNorm | `(B,T,D) → norm` | 💧 Memory |

### Quick start

```bash
cd Capstone1
source /home/jmd/venvs/rtx2000/bin/activate

python tensor_tour.py       # 🧱 Tensor internals — strides, views, broadcasting, CUDA
python autograd_demo.py     # 🔁 Gradients — computation graph, backward, no_grad
python streams_demo.py      # 🌊 CUDA streams — async model, events, prefetch

python run_bench.py                              # full sweep, all 6 ops
python run_bench.py --ops matmul softmax         # specific ops
python run_bench.py --peak-tflops 50 --peak-bw 288  # roofline annotation
python run_bench.py --profile layernorm          # kernel timeline → chrome://tracing
```

### Sample output

```
GPU    : NVIDIA RTX PRO 2000 Blackwell  (15.5 GiB)
Timing : 20 warmup, 200 repeats  (CUDA events)

Op           Config                   ms(mean)  ±(std)  TFLOPS    GB/s    AI
──────────── ──────────────────────── ──────── ──────── ──────── ──────── ────
matmul       4096x4096x4096              3.145   0.096   43.695    32.0  1365
matmul       llama7b_ffn_up              0.516   0.001    0.260   260.4     1
layernorm    llama_7b                    0.067   0.002    0.995   746.4     1
gelu         llama7b_ffn                 0.354   0.003    0.510   255.0     2
rmsnorm      llama70b_b4                 1.012   0.004    0.521   260.8     2
```

---

## 🏋️ Capstone 2 — Mini Training Framework

> **Goal:** Build a complete training framework from scratch — every component hand-rolled so you understand exactly what happens when you call `trainer.fit()`.

### What you build

A modular training library (`framework/`) that handles the full pipeline from raw datasets to saved checkpoints, with support for mixed precision, gradient accumulation, and learning rate scheduling. Trains three different architectures on three datasets.

### Learning path

```
module_tour.py  →  hooks_demo.py  →  amp_demo.py  →  train.py
     🔩                 🪝                ⚡               🚀
  nn.Module          Forward &         Mixed            Full
  Internals          Backward          Precision        Training
                     Hooks             (AMP)            Loop
```

### Models & datasets

| Model | Dataset | Input | Classes | Target Acc |
|-------|---------|-------|---------|------------|
| MLP (236k params) | MNIST | 28×28 gray | 10 | ~99% |
| CNN (95k params) | CIFAR-10 | 32×32 RGB | 10 | ~87% |
| MiniResNet-18 (11.3M) | Tiny ImageNet | 64×64 RGB | 200 | ~55% |

### Framework components

```
framework/
├── models/      MLP · CNN · MiniResNet-18 · BaseModel (hooks API)
├── data/        Dataset wrappers · transforms · DataLoader factory
├── optimizers/  SGD · Adam · AdamW
├── schedulers/  Step · Cosine · WarmupCosine
├── utils/       AverageMeter · TopKAccuracy · set_seed
├── logger.py    Console table + CSV export
├── checkpoint.py  Full state save/load (model+opt+sched+scaler)
├── evaluator.py   Val loop with no_grad + eval mode
└── trainer.py   AMP + gradient accumulation + grad clipping
```

### Quick start

```bash
cd Capstone2
source /home/jmd/venvs/rtx2000/bin/activate

# Educational tours first (beginner-friendly narration)
python tours/module_tour.py      # 🔩 7 lessons: nn.Module, Parameters, Buffers, state_dict
python tours/hooks_demo.py       # 🪝 6 lessons: forward/backward hooks, dead ReLU detection
python tours/amp_demo.py         # ⚡ 6 lessons: FP16, GradScaler, autocast, speedup

# Then train
python train.py --config mnist_mlp          # MNIST  — downloads automatically (~170 MB)
python train.py --config cifar10_cnn        # CIFAR-10 — downloads automatically (~163 MB)
python train.py --config tiny_imagenet      # Tiny ImageNet — manual download required

# Useful flags
python train.py --config cifar10_cnn --resume               # resume from latest checkpoint
python train.py --config mnist_mlp --eval-only --resume     # evaluate only, no training
python train.py --config mnist_mlp --lr 5e-4 --epochs 10   # override config values
```

### Sample training output

```
════════════════════════════════════════════════════════════
  Capstone 2 — Training Framework
  Config  : mnist_mlp    Device  : cuda    AMP : True
════════════════════════════════════════════════════════════

  Dataset : mnist    Train: 60,000    Val: 10,000
  Model   : MLP      Trainable parameters: 235,914
  Optimizer : AdamW  lr=0.001  wd=0.0001
  Scheduler : cosine  epochs=20

┌──────┬────────┬───────────┬──────────┬──────────┬────────────┐
│ epoch│  phase │   loss    │  acc@1   │  acc@5   │     lr     │
├──────┼────────┼───────────┼──────────┼──────────┼────────────┤
│    1 │  train │   0.2841  │  91.45%  │  99.98%  │  1.00e-03  │
│    1 │    val │   0.1123  │  96.68%  │  99.97%  │  1.00e-03  │
│   20 │    val │   0.0301  │  99.14%  │ 100.00%  │  1.00e-07  │
```

### Concepts mastered

| Area | What you learn |
|------|---------------|
| **nn.Module** | Parameters vs Buffers, state_dict, module tree, apply() |
| **Hooks** | Activation capture, gradient monitoring, dead ReLU detection |
| **AMP** | FP16 dynamic range, GradScaler algorithm, autocast dtype routing |
| **Training loop** | Grad accumulation (`loss/N`), grad clipping, best-model tracking |
| **DataLoader** | num_workers, pin_memory, prefetch_factor, worker seeding |
| **Checkpointing** | Full state (model+opt+sched+scaler+epoch) save and resume |

---

## 🧠 Capstone 3 — Rebuild the Transformer

> **Goal:** Implement every component of a GPT from scratch and understand why each design choice exists. Add the modern variants (RoPE, RMSNorm, SwiGLU, FlashAttention) used in LLaMA and Mistral.

### What you build

A complete transformer library (`transformer/`) implementing all components from first principles, plus a character-level GPT that trains on Shakespeare in ~15 minutes.

### Learning path

```
attention_tour.py  →  rope_tour.py  →  flash_tour.py  →  train_gpt.py
      🔍                  🌀               ⚡                 🚀
  Attention from       Rotary PE       FlashAttention      Train GPT
  First Principles     Explained       Memory Analysis     on Shakespeare
```

### Components implemented

| Component | Classic | Modern (LLaMA-style) |
|-----------|---------|---------------------|
| Position  | Sinusoidal / Learned PE | RoPE (relative, in Q/K) |
| Norm      | LayerNorm | RMSNorm (10-30% faster) |
| FFN       | GELU | SwiGLU (gated, better perplexity) |
| Attention | Naive O(T²) | FlashAttention O(T) memory |

### Quick start

```bash
cd Capstone3
source /home/jmd/venvs/rtx2000/bin/activate

python tours/attention_tour.py   # 7 lessons: attention from first principles
python tours/rope_tour.py        # 7 lessons: rotary embeddings
python tours/flash_tour.py       # 6 lessons: FlashAttention memory analysis

python train_gpt.py --config gpt_nano           # train ~10M param GPT (~15 min)
python train_gpt.py --config gpt_modern         # same size, LLaMA architecture
python train_gpt.py --config gpt_nano --generate-only --prompt "HAMLET:"
```

---

## ⚙️ Capstone 4 — PyTorch Compiler Deep Dive

> **Goal:** Treat PyTorch as a compiler stack. Trace a single tensor operation from Python bytecode all the way to GPU hardware, understanding every transformation stage.

### The Compilation Pipeline

```
Python Code
  ↓  TorchDynamo — bytecode interception, FX graph capture, guards
FX Graph (DAG of ops, no Python control flow)
  ↓  AOTAutograd — trace the backward pass ahead of time
Joint fwd+bwd FX Graph
  ↓  TorchInductor — loop fusion, memory planning, layout optimisation
Triton (GPU) or C++ (CPU)
  ↓  Triton Compiler → PTX → SASS
GPU Hardware (Tensor Cores, HBM, SRAM)
```

### What you build

Custom Triton kernels for the key transformer operations, a graph inspection toolkit, and benchmarks that quantify exactly how much each compiler stage contributes.

### Learning path

```
compiler_pipeline_tour.py  →  triton_tour.py  →  fusion_tour.py
           🗺️                       🔧                  🔥
     Full Pipeline             GPU Architecture       Memory Wall &
     End-to-End               Triton DSL             Fusion Patterns
```

### Topics covered

| Stage | Files | What you learn |
|---|---|---|
| **TorchDynamo** | `compiler/dynamo/` | Bytecode interception, graph breaks, guards, fullgraph mode |
| **FX Graph** | `compiler/fx/` | Node types, symbolic trace, custom passes, graph rewriting |
| **AOTAutograd** | `compiler/aot/` | Ahead-of-time backward tracing, functorch `grad`/`vmap` |
| **TorchInductor** | `compiler/inductor/` | Loop fusion, IR, Triton codegen, kernel cache |
| **CUDA Graphs** | `compiler/cuda_graphs/` | Capture + replay, launch overhead elimination |
| **Triton Kernels** | `kernels/` | Vector ops, softmax, tiled matmul, RMSNorm, Flash Attention |
| **Benchmarks** | `benchmarks/` | torch.compile speedup, fusion benefit, CUDA Graph overhead |

### Quick start

```bash
cd Capstone4
source /home/jmd/venvs/rtx2000/bin/activate

# Start with the tours
python tours/compiler_pipeline_tour.py   # full pipeline walk-through
python tours/triton_tour.py              # GPU arch + Triton DSL
python tours/fusion_tour.py             # memory wall + live fusion demo

# Compiler pipeline
python compiler/dynamo/basics.py
python compiler/dynamo/graph_breaks.py
TORCH_LOGS="output_code" python compiler/inductor/ir_inspect.py

# Triton kernels from scratch
python kernels/triton_basics.py
python kernels/flash_attention.py

# Measure the speedups
python benchmarks/compile_speedup.py
python benchmarks/fusion_bench.py
```

---

## 🚀 Capstone 5 — Build a Mini vLLM

> **Goal:** Implement a complete LLM inference engine from scratch, replicating the core innovations of [vLLM](https://arxiv.org/abs/2309.06180) (Kwon et al., 2023): paged KV cache, continuous batching, prefill/decode separation, and a fused Triton attention kernel.

### What you build

A fully functional inference engine (`engine/`) that can serve multiple concurrent requests with paged memory management, streaming output, INT8/NF4 quantization, and an OpenAI-compatible REST API.

### System architecture

```
Request (text)
  ↓
LLMEngine.add_request()
  ↓
Scheduler.schedule()       ← continuous batching, preemption (swap/recompute)
  ↓                ↓
BlockManager            SchedulerOutput
(KV pages)            (which seqs run this step, which blocks to swap)
  ↓
ModelRunner.execute_model()
  ↓              ↓
_run_prefill()   _run_decode()   ← separate forward passes
  ↓
PagedGPT.forward()
  ↓
write_to_cache() + paged_attention_decode()   ← scatter/gather over block tables
  ↓
Sampler (greedy / top-k / top-p / rep penalty)
  ↓
RequestOutput (streaming)
```

### Learning path

```
kv_cache_tour.py  →  scheduler_tour.py  →  inference_tour.py  →  examples/
      📦                    🗓️                     🔬                 🏃
  Why paging?           Continuous          Prefill vs Decode      15 runnable
  Fragmentation         Batching &          Roofline Model         examples
  PagedAttention        Preemption          Full engine demo
```

### Directory structure

```
Capstone5/
├── engine/
│   ├── sequence.py        ← Sequence state machine, SchedulerOutput
│   ├── kv_cache.py        ← BlockManager, PhysicalBlock, BlockAllocator
│   ├── scheduler.py       ← Continuous batching, swap/recompute preemption
│   ├── model_runner.py    ← Batch tensor prep, prefill/decode dispatch
│   └── llm_engine.py      ← add_request → step → stream/generate
├── model/
│   ├── config.py          ← ModelConfig (nano/gpt2_small), CacheConfig, SamplingParams
│   ├── gpt.py             ← PagedGPT: RMSNorm, SwiGLU FFN, paged attention
│   └── paged_attention.py ← write_to_cache, paged_attention_decode/prefill
├── kernels/
│   └── paged_attn_triton.py  ← Fused Triton decode kernel (online softmax)
├── sampling/
│   └── sampler.py         ← Greedy, top-k, top-p, repetition penalty
├── quantization/
│   └── int8_quant.py      ← INT8 symmetric, NF4 (QLoRA-style), KV cache quant
├── server/
│   ├── protocol.py        ← OpenAI-compatible Pydantic schemas
│   └── api_server.py      ← FastAPI: /v1/completions, /v1/chat/completions, SSE
├── benchmarks/
│   ├── throughput.py      ← Tokens/sec vs batch size sweep
│   └── latency.py         ← TTFT, TPOT, E2E latency breakdown
├── tours/
│   ├── kv_cache_tour.py   ← 7 lessons: memory fragmentation → PagedAttention
│   ├── scheduler_tour.py  ← 6 lessons: static → continuous batching, preemption
│   └── inference_tour.py  ← 6 lessons: prefill/decode roofline, full engine demo
├── examples/              ← 15 runnable focused demos (one concept each)
└── generate.py            ← CLI: generate text with streaming
```

### Quick start

```bash
cd Capstone5
source /home/jmd/venvs/rtx2000/bin/activate

# ── Tours (concept-first) ─────────────────────────────────────────────────────
python tours/kv_cache_tour.py        # why paging? fragmentation → PagedAttention
python tours/scheduler_tour.py       # static vs continuous batching, preemption
python tours/inference_tour.py       # prefill vs decode, roofline, full demo

# ── Model smoke tests ─────────────────────────────────────────────────────────
python model/gpt.py                  # PagedGPT prefill + decode smoke test
python model/paged_attention.py      # write_to_cache + paged_attention_decode verify

# ── Generate text ─────────────────────────────────────────────────────────────
python generate.py --prompt "Hello" --max-tokens 50 --stream

# ── Examples (one concept each) ───────────────────────────────────────────────
python examples/01_hello_engine.py         # full engine in 30 lines
python examples/03_multi_request_batching.py  # 3 concurrent requests
python examples/05_block_table_inspector.py   # live block table visualization
python examples/07_preemption_demo.py         # preemption under memory pressure
python examples/11_paged_attention_verify.py  # correctness vs standard attention
python examples/12_triton_paged_attention.py  # Triton kernel: 8.5× speedup at B=16
python examples/13_quantization_compare.py    # fp16 vs INT8 vs NF4 accuracy + size

# ── Benchmarks ────────────────────────────────────────────────────────────────
python -m benchmarks.throughput --batch-sizes 1 4 8 16
python -m benchmarks.latency --prompt-lens 16 64 256

# ── API server ────────────────────────────────────────────────────────────────
pip install fastapi uvicorn
python server/api_server.py
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "nano", "prompt": "Hello", "max_tokens": 50}'
```

### Sample benchmark output

```
Throughput (nano model, fp16, RTX PRO 2000)
  Batch     Tokens/s
  ───────────────────
      1          819
      4         1723
      8         2174
     16         2566   ← 3.1× vs batch=1

Latency
  Prompt    TTFT      TPOT     E2E
  ─────────────────────────────────
      16    1.9ms    1.1ms   71ms
      64    5.0ms    1.1ms   75ms    ← TTFT scales (compute-bound)
     256   17.2ms    1.1ms   88ms    ← TPOT flat (memory-bound)
```

### Core concepts

| Concept | Analogy | Where |
|---------|---------|-------|
| **PagedAttention** | OS virtual memory | `engine/kv_cache.py` |
| **Physical blocks** | Page frames | `engine/kv_cache.py` |
| **Block table** | Page table | `engine/sequence.py` |
| **Continuous batching** | Preemptive scheduling | `engine/scheduler.py` |
| **Prefill** | Cache warm-up (compute-bound) | `engine/model_runner.py` |
| **Decode** | Cache read per token (memory-bound) | `engine/model_runner.py` |
| **Online softmax** | Flash attention over scattered blocks | `kernels/paged_attn_triton.py` |
| **Preemption** | Swap (GPU→CPU) or recompute | `engine/scheduler.py` |
| **TTFT / TPOT** | Serving latency metrics | `benchmarks/latency.py` |
| **INT8 / NF4** | Weight + KV cache quantization | `quantization/int8_quant.py` |

### Triton kernel speedup

```
Config            Std-attn   PyRef    Triton   Speedup
──────────────────────────────────────────────────────
B=1, H=8, seq=128   0.020ms  0.059ms  0.049ms    1.2×
B=4, H=8, seq=256   0.027ms  0.217ms  0.097ms    2.2×
B=16, H=12, seq=256 0.111ms  0.855ms  0.101ms    8.5×
```

The Triton kernel eliminates the Python gather loop and fuses scatter + attention into a single pass over HBM, achieving 8.5× speedup at large batch sizes.

---

## 🧮 Core Concepts Across All Capstones

### The Roofline Model (Capstone 1 & 5)

```
TFLOPS
  │  ══════════════════════════════════ ← GPU compute peak
  │                       COMPUTE BOUND
  │                      /
  │                     / ← ridge point
  │      MEMORY BOUND  /    peak_TF / peak_BW
  │                   /
  └─────────────────────────────────── AI (FLOPs/byte)

Left of ridge  → bandwidth is the bottleneck → optimise memory layout
Right of ridge → compute is the bottleneck   → use tensor cores, quantise

Decode step:  AI ≈ 0.5–1 FLOPs/byte  → deep memory-bound
Prefill step: AI ≈ O(T) FLOPs/byte   → approaches compute-bound at large T
```

### Arithmetic Intensity — The Key Question

| Op Family | AI (float16) | Bound |
|-----------|-------------|-------|
| Large MatMul (N=4096) | ~1365 FLOPs/byte | Compute |
| Small MatMul (N=512)  | ~170 FLOPs/byte  | Compute |
| LLM Decode (1 token)  | ~0.5–1 FLOPs/byte | Memory |
| LLM Prefill (T=256)   | ~128 FLOPs/byte  | Compute |
| LayerNorm / RMSNorm   | ~1.5 FLOPs/byte  | Memory  |
| Softmax / GELU        | ~2–4 FLOPs/byte  | Memory  |

### Mixed Precision (AMP) — How It Works

```
loss (FP32)         grad × S (FP16)        true grad (FP32)
     │                     │                      │
     ▼                     ▼                      ▼
  × scale S  ──→  backward  ──→  unscale (÷S)  ──→  clip  ──→  step
  S = 65536         (FP16)           (FP32)

S halves after overflow, doubles every 2000 clean steps.
```

### PagedAttention — Memory Model (Capstone 5)

```
Traditional (wasteful):
  seq A: [████████████░░░░░░░░░░░░░░░░░]  ← pre-reserved to max_len
  seq B: [████░░░░░░░░░░░░░░░░░░░░░░░░░]  ← internal fragmentation
  seq C: [████████████████████████░░░░░]

PagedAttention (OS-style paging):
  Physical blocks:  [B0][B1][B2][B3][B4][B5][B6][B7]...
  seq A block table: [B0, B3, B7]   ← any 3 non-contiguous blocks
  seq B block table: [B1, B5]
  seq C block table: [B2, B4, B6]   ← waste < 4%
```

---

## 📚 Reading List & Recommended Resources

### Per-Capstone Reading

#### Capstone 1 — Performance Lab

| Topic | Reading | Priority |
|-------|---------|----------|
| Tensor Basics | Deep Learning with PyTorch (Ch. 1–3) | ★★★★☆ |
| Tensor API | PyTorch Tensor Documentation | ★★★★★ |
| Memory Layout | PyTorch Storage & Strides docs | ★★★★★ |
| Benchmarking | `torch.utils.benchmark` docs | ★★★★★ |
| Profiling | PyTorch Profiler Tutorial | ★★★★★ |
| CUDA | CUDA Programming Guide (Memory & Streams chapters) | ★★★★☆ |
| Source | `aten/src/ATen` | ★★★★☆ |

#### Capstone 2 — Mini Training Framework

| Topic | Reading | Priority |
|-------|---------|----------|
| Autograd | PyTorch Autograd docs | ★★★★★ |
| nn.Module | Source: `torch/nn/modules/module.py` | ★★★★★ |
| Parameters | `nn.Parameter` docs | ★★★★★ |
| Optimizers | `torch.optim` source | ★★★★☆ |
| Mixed Precision | AMP docs | ★★★★★ |
| DataLoader | `torch.utils.data` docs | ★★★★★ |
| Reproducibility | PyTorch Reproducibility Guide | ★★★★☆ |
| Checkpointing | Saving & Loading Models docs | ★★★★☆ |

#### Capstone 3 — Mini GPT

| Topic | Reading | Priority |
|-------|---------|----------|
| Transformer | Attention Is All You Need | ★★★★★ |
| GPT | GPT-2 paper | ★★★★★ |
| RoPE | RoFormer paper | ★★★★★ |
| RMSNorm | RMSNorm paper | ★★★★☆ |
| SwiGLU | PaLM paper | ★★★★☆ |
| FlashAttention | FlashAttention v1 & v2 papers | ★★★★★ |
| KV Cache | vLLM paper | ★★★★★ |
| Source | `torch.nn.MultiheadAttention` | ★★★★★ |

#### Capstone 4 — Compiler Stack

| Topic | Reading | Priority |
|-------|---------|----------|
| FX Graph | FX documentation | ★★★★★ |
| TorchDynamo | TorchDynamo docs | ★★★★★ |
| TorchInductor | TorchInductor design docs | ★★★★★ |
| Triton | Triton tutorials | ★★★★★ |
| AOTAutograd | AOTAutograd docs | ★★★★☆ |
| torch.compile | Official tutorial | ★★★★★ |
| Source | `torch/_dynamo` | ★★★★★ |
| Source | `torch/_inductor` | ★★★★★ |

#### Capstone 5 — Mini vLLM

| Topic | Reading | Priority |
|-------|---------|----------|
| vLLM | vLLM paper | ★★★★★ |
| PagedAttention | PagedAttention paper | ★★★★★ |
| TensorRT-LLM | Technical docs | ★★★★☆ |
| llama.cpp | Source code | ★★★★☆ |
| SGLang | Design docs | ★★★★☆ |
| Quantization | AWQ paper | ★★★★☆ |
| Quantization | GPTQ paper | ★★★★☆ |
| FP8 | NVIDIA FP8 paper | ★★★★☆ |

---

### Cross-Cutting: Mathematics

| Topic | Reading | Priority |
|-------|---------|----------|
| Matrix calculus | The Matrix Cookbook | ★★★★★ |
| Linear algebra | Mathematics for Machine Learning | ★★★★☆ |
| Probability | Murphy: Probabilistic Machine Learning (selected chapters) | ★★★☆☆ |
| Optimization | Deep Learning — Goodfellow et al. (Ch. 8) | ★★★★☆ |
| Numerical methods | Numerical Linear Algebra — Trefethen & Bau | ★★★★☆ |

### Cross-Cutting: GPU Programming

| Topic | Reading | Priority |
|-------|---------|----------|
| CUDA Architecture | CUDA Programming Guide | ★★★★★ |
| Occupancy | CUDA Best Practices Guide | ★★★★★ |
| Shared Memory | CUDA Programming Guide | ★★★★★ |
| Tensor Cores | NVIDIA whitepapers | ★★★★★ |
| Memory Hierarchy | CUDA documentation | ★★★★★ |

### Cross-Cutting: Profiling

| Topic | Reading | Priority |
|-------|---------|----------|
| `torch.profiler` | Official docs | ★★★★★ |
| Nsight Systems | User Guide | ★★★★★ |
| Nsight Compute | User Guide | ★★★★★ |
| NVTX | CUDA docs | ★★★★☆ |
| Perfetto | Official documentation | ★★★★☆ |

---

### PyTorch Source Code Reading Order

Rather than reading the repository top-to-bottom, follow this sequence:

| Week | Directory | Why |
|------|-----------|-----|
| 1 | `torch/tensor.py` | Tensor API |
| 2 | `torch/nn/modules/` | Neural network abstractions |
| 3 | `torch/optim/` | Optimization algorithms |
| 4 | `torch/utils/data/` | Data pipeline |
| 5 | `torch/autograd/` | Automatic differentiation |
| 6 | `c10/` | Dispatch and core infrastructure |
| 7 | `aten/` | Native tensor operators |
| 8 | `torch/_dynamo/` | Graph capture |
| 9 | `torch/_inductor/` | Compiler backend |
| 10 | `torch/csrc/` | Python/C++ interface |

---

### Recommended Courses

| Course | Why | Priority |
|--------|-----|----------|
| Official PyTorch Tutorials | Best coverage of the framework | ★★★★★ |
| Stanford CS231n | Vision models and training fundamentals | ★★★★☆ |
| Stanford CS224N | Transformers and language models | ★★★★★ |
| CMU 11-785 (Deep Learning) | Advanced training and systems | ★★★★★ |
| Stanford CS336 (LLMs) | Modern LLM architecture and training | ★★★★★ |
| Hugging Face Course | Transformers in practice | ★★★★☆ |
| NVIDIA Deep Learning Institute | GPU optimization (CUDA/PyTorch) | ★★★★☆ |

---

### Papers Worth Reading End-to-End

These are the papers not to skip:

1. **Attention Is All You Need** — Vaswani et al. 2017
2. **FlashAttention** — Dao et al. 2022
3. **FlashAttention-2** — Dao 2023
4. **RoFormer** — Su et al. 2021 (RoPE)
5. **Root Mean Square Layer Normalization** — Zhang & Sennrich 2019
6. **PagedAttention / vLLM** — Kwon et al. 2023
7. **QLoRA** — Dettmers et al. 2023 (NF4 quantization)
8. **AWQ** — Lin et al. 2023 (activation-aware weight quantization)
9. **GPTQ** — Frantar et al. 2022 (post-training quantization)
10. **TorchDynamo** — Ansel et al. 2023
11. **TorchInductor** — PyTorch compiler design docs
12. **Triton** — Tillet et al. 2019 (tiled neural network compiler)

---

## ⚙️ Hardware

All capstones are developed and tested on:

```
GPU     : NVIDIA RTX PRO 2000 Blackwell
VRAM    : 15.5 GiB
CUDA    : 12.8
PyTorch : 2.11.0+cu128
Python  : 3.12
```

---

## 📦 Setup

```bash
# Clone
git clone https://github.com/lalitprasadperi/torch_forge.git
cd torch_forge

# Activate the venv (has torch + torchvision + numpy + pillow)
source /home/jmd/venvs/rtx2000/bin/activate

# Or add to ~/.bashrc so it activates automatically
echo 'source /home/jmd/venvs/rtx2000/bin/activate' >> ~/.bashrc
```

---

<div align="center">

**🔥 torch_forge** — *forge your understanding one capstone at a time*

</div>
