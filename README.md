# 🔥 torch_forge

> *Build things. Measure things. Understand why they work.*

A hands-on PyTorch curriculum built as a series of **capstone projects** — each one a self-contained deep-dive into a core area of GPU computing and deep learning engineering.

The philosophy: **don't just read about PyTorch, build tools with it.** Every capstone produces something real that you can run, measure, and extend.

---

## 🗂️ Capstones

| # | Title | Topics | Status |
|---|-------|--------|--------|
| [**1**](Capstone1/) | 🔬 **PyTorch Performance Lab** | Tensors, Autograd, CUDA Streams, Benchmarking, Roofline Model | ✅ Complete |
| 2 | ⚡ *Coming soon* | Triton kernels — write a fused RMSNorm and Flash-Attention from scratch | 🔜 |
| 3 | 🧵 *Coming soon* | Multi-GPU training — DDP, tensor parallelism, NCCL all-reduce | 🔜 |
| 4 | 📉 *Coming soon* | Quantisation — INT8 matmul, GPTQ, activation-aware scaling | 🔜 |
| 5 | 🔄 *Coming soon* | CUDA Graphs — capture and replay, zero-CPU-overhead inference | 🔜 |

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

## 🧠 Core Concepts Across All Capstones

### The Roofline Model

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

---

## ⚙️ Hardware

All capstones are developed and tested on:

```
GPU  : NVIDIA RTX PRO 2000 Blackwell
VRAM : 15.5 GiB
CUDA : 12.8
PyTorch : 2.11.0+cu128
Python  : 3.12
```

---

## 📦 Setup

```bash
# Clone
git clone https://github.com/lalitprasadperi/torch_forge.git
cd torch_forge

# Each capstone uses the rtx2000 venv (already set up on this machine)
source /home/jmd/venvs/rtx2000/bin/activate

# Or install fresh from requirements.txt
pip install -r Capstone1/requirements.txt
```

---

<div align="center">

**🔥 torch_forge** — *forge your understanding one capstone at a time*

</div>
