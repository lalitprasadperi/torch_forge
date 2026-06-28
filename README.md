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

## 🧮 Core Concepts Across All Capstones

### The Roofline Model (Capstone 1)

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
```

### Arithmetic Intensity — The Key Question

| Op Family | AI (float16) | Bound |
|-----------|-------------|-------|
| Large MatMul (N=4096) | ~1365 FLOPs/byte | Compute |
| Small MatMul (N=512)  | ~170 FLOPs/byte  | Compute |
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
