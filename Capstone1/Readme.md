# 🔬 Capstone 1 — PyTorch Performance Lab

> *"You can't optimise what you can't measure."*
> Build a GPU benchmarking framework from scratch while learning PyTorch inside-out.

---

## 🎯 What You're Building

A **production-grade performance lab** that measures every major neural network operation on your GPU — the same kind of tool used by ML compiler teams, kernel engineers, and inference optimization shops.

When you're done, you'll be able to answer questions like:

- *"Does my LayerNorm saturate the memory bus, or is there room to improve the kernel?"*
- *"At what matrix size does my GPU flip from memory-bound to compute-bound?"*
- *"How much does batching help for LLaMA-style FFN projections?"*
- *"Which timing method is accurate — `perf_counter` or CUDA Events?"*

---

## 📚 Learning Path

Work through the scripts **in order**. Each one builds on the last.

```
tensor_tour.py  →  autograd_demo.py  →  streams_demo.py  →  run_bench.py
     🧱                  🔁                   🌊                  📊
  Tensors &           Gradients &          CUDA Streams        Full Perf
  Memory Model        Backprop             & Timing            Benchmark
```

---

## 🗂️ Project Layout

```
Capstone1/
│
├── 📖 tensor_tour.py          # LESSON SET 1  — Tensor internals
├── 📖 autograd_demo.py        # LESSON SET 2  — Automatic differentiation
├── 📖 streams_demo.py         # LESSON SET 3  — CUDA streams & timing
├── 🚀 run_bench.py            # CLI entry point — run the full benchmark
│
└── perf_lab/                  # The benchmarking library
    │
    ├── ops/                   # One file per operation
    │   ├── base.py            #   Abstract BenchOp + BenchmarkResult dataclass
    │   ├── matmul.py          #   Matrix multiply
    │   ├── conv2d.py          #   2-D convolution
    │   ├── layernorm.py       #   Layer normalisation
    │   ├── softmax.py         #   Softmax (attention scores)
    │   ├── gelu.py            #   GELU activation
    │   └── rmsnorm.py         #   RMS normalisation (LLaMA-style)
    │
    ├── timing/                # Two timing strategies
    │   ├── cuda_timer.py      #   CUDA Events  — fast, per-kernel GPU timestamps
    │   └── bench_timer.py     #   torch.utils.benchmark — adaptive, statistically robust
    │
    ├── metrics.py             # TFLOPS, GB/s, AI, Roofline model
    ├── profiler_runner.py     # torch.profiler → Chrome trace + kernel table
    └── runner.py              # Orchestrator: sweeps ops × configs, prints table
```

---

## 🧱 Lesson Set 1 — Tensor Tour

**File:** `tensor_tour.py`
**Run:** `python tensor_tour.py`

### What You'll Learn

| Lesson | Topic | Key Insight |
|--------|-------|-------------|
| 1 | **Creating Tensors** | `zeros`, `ones`, `randn`, `arange`, `tensor`, `*_like` |
| 2 | **Tensor Metadata** | `shape`, `dtype`, `device`, `stride`, `numel` |
| 3 | **The Memory Model** | A tensor is a *view* over a flat buffer — not its data |
| 4 | **Reshaping** | `view()` = free reinterpret; `contiguous()` = copy |
| 5 | **Operations** | Pointwise · Reduction · Matmul |
| 6 | **Broadcasting** | Shape rules, transformer patterns, error detection |
| 7 | **CUDA Tensors** | `.cuda()`, memory tracking, async execution |

### 🧠 The Most Important Concept: Strides

```
Logical view (3×4)          Physical memory (flat buffer)
┌───┬───┬───┬───┐           ┌────────────────────────────────────────┐
│ 0 │ 1 │ 2 │ 3 │  row 0 → │  0   1   2   3   4   5   6  ...  11   │
├───┼───┼───┼───┤           └────────────────────────────────────────┘
│ 4 │ 5 │ 6 │ 7 │  row 1        ↑
├───┼───┼───┼───┤           stride = (4, 1)
│ 8 │ 9 │10 │11 │  row 2   element[i,j] = buffer[ i×4 + j×1 ]
└───┴───┴───┴───┘
```

**`a.t()` costs nothing** — it swaps the strides without touching the buffer.
**`.contiguous()`  costs a copy** — forces row-major layout in a new buffer.

### 📡 Broadcasting at a Glance

```python
tokens (8, 512)               # batch of 8 tokens
bias   (   512)               # one bias vector
output (8, 512)  ✓            # bias broadcasts across batch dim

scores (2, 8, 64, 64)         # attention scores: batch, heads, T, T
mask   (1, 1, 64, 64)         # causal mask
output (2, 8, 64, 64)  ✓      # mask broadcasts over batch and heads
```

---

## 🔁 Lesson Set 2 — Autograd Demo

**File:** `autograd_demo.py`
**Run:** `python autograd_demo.py`

### What You'll Learn

| Lesson | Topic | Key Insight |
|--------|-------|-------------|
| 1 | **Why Gradients?** | Gradient descent = the engine of all deep learning |
| 2 | **First Gradient** | `requires_grad=True` → `.backward()` → `.grad` |
| 3 | **Computation Graph** | PyTorch builds a DAG of ops during the forward pass |
| 4 | **Mini Neural Net** | Full W·x + b → MSE loss → backward cycle |
| 5 | **Vector Outputs** | Jacobian-vector products when output isn't scalar |
| 6 | **`no_grad`** | Skip graph construction for 30–50% faster inference |
| 7 | **`autograd.grad`** | Targeted gradients without polluting `.grad` |
| 8 | **Higher-Order Grads** | Gradient of a gradient (2nd derivatives) |
| 9 | **Grad Accumulation** | The silent bug that ruins training — and how to fix it |

### 🧠 The Computation Graph

```
  x ──────┐
           ├──► MulBackward ──┐
  y ──────┘                   ├──► AddBackward ──► z (loss)
  y ──► PowBackward(n=2) ──┘              │
                                          │ .backward()
  x.grad ◄────────────────────────────────┘
  y.grad ◄────────────────────────────────┘
```

### ⚡ The Training Loop Pattern

```python
for batch in dataloader:
    optimizer.zero_grad()          # ❗ always clear old grads first
    out  = model(batch)            # → forward pass, graph builds here
    loss = criterion(out, labels)  # → scalar
    loss.backward()                # → fill .grad on all weights
    optimizer.step()               # → W = W - lr × W.grad

with torch.no_grad():              # → no graph, 30-50% faster
    val_out = model(val_batch)
```

---

## 🌊 Lesson Set 3 — CUDA Streams & Timing

**File:** `streams_demo.py`
**Run:** `python streams_demo.py`

### What You'll Learn

| Lesson | Topic | Key Insight |
|--------|-------|-------------|
| 1 | **Async Execution Model** | CPU launches kernels and moves on — GPU runs in parallel |
| 2 | **❌ Wrong Timing** | `perf_counter()` only measures kernel *launch*, not *execution* |
| 3 | **✅ CUDA Events** | GPU-timestamped events — the only accurate GPU timer |
| 4 | **Cold vs Warm** | First run can be **100–200×** slower; always warm up |
| 5 | **Default Stream** | Stream 0 = serialised queue, operations run in order |
| 6 | **Non-Default Streams** | Multiple queues can run concurrently on different SMs |
| 7 | **Stream Overlap** | Measure actual speedup (depends on SM utilisation per op) |
| 8 | **CUDA Event Semaphore** | `event.record()` + `stream.wait_event()` = GPU-side barrier |
| 9 | **H2D Prefetch** | Overlap CPU→GPU copy with compute → free throughput |

### 🖥️ GPU Execution Model

```
CPU timeline:
────────────────────────────────────────────────────────────────
[ launch kernel A ][ launch kernel B ][ Python continues... ]
         ↓                 ↓
GPU stream 0:
────────────────────────────────────────────────────────────────
         [    kernel A runs    ][    kernel B runs    ]

GPU ≠ CPU: the CPU is already done while the GPU is still working.
Use torch.cuda.synchronize() to make the CPU wait.
```

### ⏱️ Why CUDA Events, Not `perf_counter`

```
❌  t0 = perf_counter()
    c = a @ b              ← CPU returns IMMEDIATELY (async launch)
    t1 = perf_counter()    ← measures 0.004ms (just the launch!)

✅  start.record()
    c = a @ b              ← GPU timestamps inserted in its own timeline
    end.record()
    synchronize()          ← CPU waits
    elapsed = start.elapsed_time(end)   ← actual GPU execution time
```

### 🔀 H2D Prefetch Pattern

```
❌  Sequential:
    [copy batch 0]──[compute batch 0]──[copy batch 1]──[compute batch 1]

✅  Overlapped (prefetch):
    [copy batch 0]──────────────────────────────────────────
                   [compute batch 0]──[copy batch 1]
                                                    [compute batch 1]
    → PCIe transfer hidden behind compute = free throughput
```

---

## 📊 Benchmark — Operations Covered

**Run:** `python run_bench.py`

### The Six Operations

| # | Operation | Shape | Use Case | Bound |
|---|-----------|-------|----------|-------|
| 1 | **MatMul** | `(M,K) @ (K,N)` | Linear layers, attention | 🔥 Compute |
| 2 | **Conv2D** | `(N,C,H,W) * kernel` | CNNs, vision encoders | ⚡ Mixed |
| 3 | **LayerNorm** | `(B,T,D) → norm` | Transformer pre/post norm | 💧 Memory |
| 4 | **Softmax** | `(B,H,T,T) → probs` | Attention weights | 💧 Memory |
| 5 | **GELU** | `(B,T,D) → activation` | FFN block | 💧 Memory |
| 6 | **RMSNorm** | `(B,T,D) → norm` | LLaMA-style norm | 💧 Memory |

### 📐 Understanding the Metrics

#### 🔥 TFLOPS — Compute Throughput

```
TFLOPS = FLOPs ÷ (latency_s × 10¹²)

What it tells you: how hard the ALUs are working
When to watch it: compute-bound ops (large MatMul, Conv2D)

MatMul FLOPs = 2 × M × K × N
  (one multiply + one add per inner-product term)
```

#### 💧 GB/s — Memory Bandwidth

```
GB/s = (bytes_read + bytes_written) ÷ (latency_s × 10⁹)

What it tells you: how fast data moves between VRAM and SM caches
When to watch it: memory-bound ops (LayerNorm, Softmax, GELU, RMSNorm)
```

#### 📏 AI — Arithmetic Intensity

```
AI = FLOPs ÷ bytes   (units: FLOPs/byte)

This is the x-axis of the Roofline Model:
  AI < ridge point  →  MEMORY BOUND   (bandwidth is the bottleneck)
  AI > ridge point  →  COMPUTE BOUND  (ALUs are the bottleneck)

Ridge point = peak_TFLOPS ÷ peak_BW_GB_s
```

### 🏔️ The Roofline Model

```
TFLOPS
  │  ╔═══════════════════════════════════╗ ← Compute Ceiling (GPU peak)
  │  ║                    COMPUTE BOUND ║
  │ /║                                  ║
  │/ ║   MEMORY BOUND                   ║
  │  ╚═════════════════╦════════════════╝
  │                    ↑
  │               Ridge Point
  │            (peak_TF / peak_BW)
  └────────────────────────────────────── AI (FLOPs/byte)

Ops to the LEFT of the ridge:  optimise memory access patterns
Ops to the RIGHT of the ridge: optimise compute (quantisation, tiling)
```

#### Real AI Values for Each Op (float16)

| Operation | FLOPs | Bytes | AI (FLOPs/byte) | Bound |
|-----------|-------|-------|-----------------|-------|
| MatMul 4096³ | 2×4096³ | 3×4096²×2 | **1365** | 🔥 Compute |
| MatMul 512³  | 2×512³  | 3×512²×2  | **170**  | 🔥 Compute |
| Conv2D 3×3   | ~varies | ~varies   | **~20–200** | ⚡ Mixed |
| LayerNorm    | 8×B×T×D | 6×B×T×D×2 | **~1.3** | 💧 Memory |
| Softmax      | 5×B×H×T² | 2×B×H×T²×2 | **~2.5** | 💧 Memory |
| GELU         | 8×elems | 2×elems×2 | **~2.0** | 💧 Memory |
| RMSNorm      | 6×B×T×D | 4×B×T×D×2 | **~1.5** | 💧 Memory |

---

## 🖥️ CLI Reference

### Basic Usage

```bash
# Activate the right venv first
source /home/jmd/venvs/rtx2000/bin/activate
cd /home/jmd/Desktop/TrainHard/LearnTorch/Capstone1
```

### Run the Tutorials (in order)

```bash
python tensor_tour.py              # 🧱 Tensor internals
python autograd_demo.py            # 🔁 Gradients & backprop
python streams_demo.py             # 🌊 CUDA streams & timing
```

### Run the Benchmark

```bash
# Full sweep — all 6 ops, all configs
python run_bench.py

# Specific ops only
python run_bench.py --ops matmul layernorm softmax

# Quick smoke test (few reps)
python run_bench.py --warmup 5 --repeat 30

# With Roofline annotation (add your GPU's spec sheet numbers)
python run_bench.py --peak-tflops 50 --peak-bw 288

# Use torch.utils.benchmark timer (adaptive, more robust for variable ops)
python run_bench.py --timer bench --ops gelu rmsnorm

# Profile a single op → opens Chrome trace at chrome://tracing
python run_bench.py --profile softmax
python run_bench.py --profile matmul --trace traces/matmul.json
```

### Output Columns

```
Op           Config                   ms(mean)  ±(std)  TFLOPS    GB/s    AI   Bound
──────────── ──────────────────────── ──────── ──────── ──────── ──────── ──── ─────
matmul       4096x4096x4096              3.145   0.096   43.695    32.0  1365  COM
layernorm    llama_7b                    0.067   0.002    0.995   746.4     1   MEM
gelu         llama7b_ffn                 0.354   0.003    0.510   255.0     2   MEM
```

| Column | Meaning | Good when... |
|--------|---------|--------------|
| `ms(mean)` | Average latency, CUDA events | Lower is better |
| `±(std)` | Run-to-run stability | < 5% of mean |
| `TFLOPS` | Compute throughput | High for MatMul/Conv |
| `GB/s` | Memory bandwidth | High for Norm/GELU |
| `AI` | FLOPs per byte | Tells you which metric matters |
| `Bound` | COM = compute, MEM = memory | Matches expected behaviour |

---

## 🔍 Profiling — Kernel Timeline

The profiler captures every CUDA kernel launch and produces a Chrome trace you can inspect visually.

```bash
python run_bench.py --profile matmul --trace traces/matmul.json
```

Then open **`chrome://tracing`** in Chrome, click **Load**, and select `traces/matmul.json`.

```
chrome://tracing view:

GPU  ┃▓▓▓▓▓▓matmul_kernel▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓┃
CPU  ┃▒▒▒▒launch▒▒▒ ... Python overhead ...   ┃
      ├───────── kernel time ─────────────────┤
```

You can see:
- Exact kernel names (e.g. `ampere_fp16_s1688gemm_...`)
- Kernel start/end times and gaps between kernels
- Memory allocation/free events
- CPU↔GPU sync points

---

## ⏱️ Two Timing Strategies — When to Use Each

### CUDA Events (`CudaTimer`) — default

```python
start = torch.cuda.Event(enable_timing=True)
end   = torch.cuda.Event(enable_timing=True)
start.record()
fn()
end.record()
torch.cuda.synchronize()
ms = start.elapsed_time(end)
```

✅ Per-iteration GPU timestamps
✅ Fast — minimal overhead per rep
✅ Works even during multi-stream execution
✅ Best for: tight benchmarks, comparing specific kernels

### `torch.utils.benchmark` (`BenchTimer`) — `--timer bench`

```python
timer = benchmark.Timer(stmt="fn()", globals={"fn": fn})
result = timer.blocked_autorange(min_run_time=1.0)
```

✅ Adapts iteration count until measurement is stable
✅ Handles CPU overhead subtraction automatically
✅ Returns IQR (robust to outliers) instead of std
✅ Best for: one-off measurements, ops with variable runtime

---

## 🧩 Concepts Mastered

After working through this capstone you will have hands-on experience with:

### PyTorch Fundamentals
- [x] 🧱 Tensor creation, metadata, dtype, device
- [x] 📐 Storage, strides, views, contiguous layout
- [x] 📡 Broadcasting rules and shape inference
- [x] ➗ Pointwise, reduction, and matrix operations
- [x] 🔁 `requires_grad`, `.backward()`, `.grad`
- [x] 🌐 The computation graph (DAG of `grad_fn` nodes)
- [x] 🚫 `torch.no_grad()` and `torch.inference_mode()`

### GPU Programming
- [x] 🌊 CUDA streams and the async execution model
- [x] ⏱️ CUDA Events vs `perf_counter` for timing
- [x] 🧊 Cold vs warm runs — why warmup is mandatory
- [x] 🔀 Stream parallelism and overlap measurement
- [x] 🔗 Cross-stream sync with CUDA events as semaphores
- [x] 📦 H2D prefetch pattern (copy hidden behind compute)

### Performance Analysis
- [x] 🔥 TFLOPS — compute throughput measurement
- [x] 💧 GB/s — memory bandwidth measurement
- [x] 📏 Arithmetic Intensity and the Roofline Model
- [x] 🏔️ Compute-bound vs memory-bound identification
- [x] 🔬 `torch.profiler` and Chrome trace analysis

### Ops Benchmarked
- [x] ⚡ MatMul — the workhorse of neural networks
- [x] 🖼️ Conv2D — CNNs, vision, and cuDNN algorithm selection
- [x] 📊 LayerNorm — two-pass normalisation, bandwidth analysis
- [x] 🎯 Softmax — attention weight computation
- [x] ⚙️ GELU — transformer FFN activation
- [x] 🦙 RMSNorm — LLaMA-style single-pass normalisation

---

## 💡 Interesting Results to Look For

### 🔍 MatMul — Compute vs Memory Transition

```
Size 512³  → AI ~170  → compute bound but GPU not fully saturated
Size 4096³ → AI ~1365 → deeply compute bound, should hit near-peak TFLOPS

LLaMA decode (M=1):  extremely memory bound (AI ≈ 1)
LLaMA prefill:       compute bound (M=2048, many tokens)
```

> **Question to explore:** At what matrix size does your GPU transition
> from memory-bound to compute-bound? That's your practical ridge point.

### 🔍 LayerNorm — Memory Wall

```
AI ≈ 1.3 FLOPs/byte for all shapes
GB/s should be ~constant regardless of hidden dim

If GB/s is MUCH lower than GPU peak bandwidth → kernel isn't fusing the passes
A Triton kernel can be 2× faster here
```

### 🔍 Cold vs Warm (streams_demo.py)

```
Lesson 4 output on your machine:
  Cold run #1: ~100 ms
  Warm runs avg: ~0.6 ms
  Ratio: ~150x

This is why every production benchmark has a warmup loop.
```

---

## 📦 Dependencies

```bash
# Already in the rtx2000 venv — no install needed
torch >= 2.1.0    # (this machine has 2.11.0+cu128)
```

---

## 🚀 Next Steps (Capstone 2 Ideas)

Once you're comfortable with measuring performance, the natural next step is **improving** it:

| Idea | What You'd Learn |
|------|-----------------|
| 🔺 Write a Triton kernel for RMSNorm | Fused single-pass, custom CUDA memory tiling |
| ⚡ Flash-Attention from scratch | Online softmax, tiled SRAM computation |
| 📉 INT8 quantised MatMul | Post-training quantisation, CUTLASS |
| 🧵 Multi-GPU matmul with NCCL | All-reduce, tensor parallelism |
| 🔄 CUDA graph capture | Replay a benchmark with zero CPU overhead |
| 🧠 Build a custom `autograd.Function` | Forward/backward in C++/CUDA extension |

---

<div align="center">

**🔬 Built as part of the TrainHard → LearnTorch curriculum**

```
tensor_tour.py → autograd_demo.py → streams_demo.py → run_bench.py
```

*Run each script. Read every line of output. Then read the source.*

</div>
