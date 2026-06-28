# Capstone 4 — PyTorch Systems & Compiler Deep Dive

Treat PyTorch as a compiler stack. Trace a tensor operation from Python all the way to GPU hardware.

```
Python
  ↓  TorchDynamo (bytecode interception, FX graph capture)
FX Graph
  ↓  AOTAutograd (trace forward + backward ahead of time)
Joint fwd/bwd FX Graph
  ↓  TorchInductor (loop fusion, memory planning, codegen)
Triton / C++
  ↓  Triton Compiler → PTX
CUDA / GPU Hardware
```

---

## Structure

```
Capstone4/
├── compiler/
│   ├── dynamo/
│   │   ├── basics.py          # torch.compile, backends, fullgraph, explain()
│   │   └── graph_breaks.py    # what causes breaks and how to fix them
│   ├── fx/
│   │   └── graph_inspect.py   # symbolic_trace, export, node types, custom passes
│   ├── aot/
│   │   └── autograd.py        # AOTAutograd, functorch transforms, vmap+grad
│   ├── inductor/
│   │   └── ir_inspect.py      # Inductor IR, fusion, kernel cache, codegen
│   └── cuda_graphs/
│       └── basics.py          # manual graph, make_graphed_callables, speedup bench
├── kernels/
│   ├── triton_basics.py       # vector add, row softmax, fused bias+relu (+autotune)
│   ├── matmul.py              # tiled GEMM with Tensor Cores, autotune, vs cuBLAS
│   ├── fused_ops.py           # fused RMSNorm, SwiGLU, residual+RMSNorm
│   └── flash_attention.py     # tiled attention without O(T²) memory + SDPA comparison
├── benchmarks/
│   ├── compile_speedup.py     # MLP, TransformerBlock, full training step
│   ├── fusion_bench.py        # per-pattern fusion speedup, HBM traffic saved
│   └── cuda_graph_bench.py    # kernel count, launch overhead, graph speedup
└── tours/
    ├── compiler_pipeline_tour.py  # end-to-end narrated walk: Python → GPU
    ├── triton_tour.py             # GPU architecture, memory hierarchy, Triton DSL
    └── fusion_tour.py             # memory wall, fusion patterns, live measurement
```

---

## Quick Start

```bash
# Activate the venv
source /home/jmd/venvs/rtx2000/bin/activate
cd /home/jmd/Desktop/TrainHard/LearnTorch/Capstone4

# Tours (start here)
python tours/compiler_pipeline_tour.py
python tours/triton_tour.py
python tours/fusion_tour.py

# Compiler pipeline
python compiler/dynamo/basics.py
python compiler/dynamo/graph_breaks.py
python compiler/fx/graph_inspect.py
python compiler/aot/autograd.py
TORCH_LOGS="output_code" python compiler/inductor/ir_inspect.py
python compiler/cuda_graphs/basics.py

# Triton kernels
python kernels/triton_basics.py
python kernels/matmul.py
python kernels/fused_ops.py
python kernels/flash_attention.py

# Benchmarks
python benchmarks/compile_speedup.py
python benchmarks/fusion_bench.py
python benchmarks/cuda_graph_bench.py
```

---

## What Each Stage Does

### TorchDynamo

Python's bytecode evaluator is replaced with Dynamo's evaluator at runtime.
Dynamo watches every instruction. Tensor ops are recorded into an FX graph.
Python control flow that depends on tensor *values* (`.item()`, `if tensor > 0`) causes a **graph break** — Dynamo compiles what it has, runs the Python eagerly, then resumes tracing.

Every compiled graph is guarded: shape, dtype, device. Guards fail → recompile.

```python
# See what Dynamo produces
explanation = torch._dynamo.explain(fn)(x)
print(f"graphs: {explanation.graph_count}, breaks: {explanation.graph_break_count}")
```

### FX Graph

A DAG of PyTorch operations. Nodes are ops; edges are tensors. No Python control flow.

```python
import torch.fx as fx
gm = fx.symbolic_trace(model)
gm.graph.print_tabular()   # see every node
print(gm.code)             # see the reconstructed Python
```

### AOTAutograd

Traces the **backward** pass too, as a static FX graph. Both forward and backward are then compiled by Inductor. Enables cross-fwd/bwd kernel fusion.

```python
from functorch.compile import aot_function
compiled = aot_function(fn, fw_compiler, bw_compiler)
```

### TorchInductor

Converts FX ops to loops; fuses loops with the same iteration space; generates Triton (GPU) or C++/OpenMP (CPU). Results cached at `~/.cache/torch_extensions/`.

```bash
TORCH_LOGS="output_code" python script.py   # print generated Triton
```

### Triton

Python DSL that compiles to PTX. You write one program per tile; Triton handles threading. Key primitives: `tl.load`, `tl.store`, `tl.dot`, `tl.sum`.

```python
@triton.jit
def kernel(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid  = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x    = tl.load(x_ptr + offs, mask=offs < N)
    tl.store(out_ptr + offs, tl.maximum(x, 0), mask=offs < N)
```

### CUDA Graphs

Capture the entire sequence of CUDA kernel launches once; replay with one CPU call.

```python
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_out = model(static_x)   # record

static_x.copy_(new_input)
g.replay()   # one CPU call → all GPU ops
```

---

## Key Concepts

| Concept | What it Means | Where to Learn |
|---|---|---|
| Graph break | Point where Dynamo can't trace — falls back to Python | `compiler/dynamo/graph_breaks.py` |
| Guard | Assertion that a compiled graph is still valid | `compiler/dynamo/basics.py` |
| FX node op types | placeholder, get_attr, call_function, call_module, output | `compiler/fx/graph_inspect.py` |
| AOTAutograd | Traces backward statically, enables cross-fwd/bwd fusion | `compiler/aot/autograd.py` |
| Operator fusion | Multiple ops → one kernel, fewer HBM round-trips | `tours/fusion_tour.py` |
| Arithmetic intensity | FLOPs/byte — determines if op is memory or compute-bound | `tours/triton_tour.py` |
| Tensor Cores | Hardware matrix units inside SMs, used by `tl.dot` | `kernels/matmul.py` |
| CUDA Graph | Record + replay sequence, eliminates per-kernel CPU launch | `compiler/cuda_graphs/basics.py` |
| Flash Attention | Tiled attention with O(T) memory via online softmax | `kernels/flash_attention.py` |

---

## Why This Matters for Performance Engineering

Every millisecond saved in model inference or training comes from one of:

1. **Fewer operations** — algorithmic improvement (KV cache, quantisation)
2. **Faster operations** — better kernels (Tensor Cores, Flash Attention)
3. **Fewer memory accesses** — kernel fusion (torch.compile)
4. **Less CPU overhead** — CUDA Graphs

After this capstone you can:
- Profile *where* a model's time goes (compute vs memory vs CPU overhead)
- Identify graph breaks and fix them
- Inspect what torch.compile generates and understand why
- Write custom Triton kernels for ops that Inductor can't fuse
- Apply CUDA Graphs for fixed-shape inference
- Reason about the Roofline model for your specific GPU

---

## Verified Results (RTX PRO 2000, PyTorch 2.11, CUDA 12.8)

### Triton Kernels

| Kernel | Triton | Torch (eager) | Max diff |
|---|---|---|---|
| Vector add (1M fp32) | 0.007 ms | 0.006 ms | 0.00e+00 |
| Row softmax (512×32K fp16) | 0.584 ms | 0.532 ms | 4.7e-10 |
| Fused bias+ReLU (2K×4K fp16) | 0.264 ms | 0.531 ms — **2.0×** | 0.00e+00 |
| Fused RMSNorm (2K×4K fp16) | 0.031 ms | 1.182 ms — **38×** | 1.9e-03 |
| Fused SwiGLU (2K×2*d fp16) | 1.242 ms | 0.884 ms | 7.8e-03 |
| Fused Residual+RMSNorm (2K×4K fp16) | 0.266 ms | 1.357 ms — **5.1×** | 3.9e-03 |

### Kernel Fusion (torch.compile vs eager)

| Pattern | Eager | Compiled | Speedup |
|---|---|---|---|
| GELU chain `x*2+0.5 → GELU` | 0.258 ms | 0.045 ms | **5.7×** |
| RMSNorm manual | 1.181 ms | 0.030 ms | **39×** |
| Residual + RMSNorm | 1.331 ms | 0.155 ms | **8.6×** |
| HBM saved over 32 layers | — | — | **~74 ms/fwd** |

### CUDA Graph Overhead

| Model | Eager | CUDA Graph | Speedup |
|---|---|---|---|
| SmallNet 32×Linear[64] | 0.326 ms | 0.147 ms | **2.2×** |
| MediumNet 16×Linear[512] | 0.165 ms | 0.107 ms | **1.5×** |
| LargeNet 4×Linear[4096] | 1.228 ms | 1.279 ms | **1.0×** *(compute-bound)* |

SmallNet launches **96 CUDA kernels** per forward pass (~1.4 ms launch overhead at 15 μs/kernel). CUDA Graphs reduce that to ~5 μs total.

---

*Capstone 4 of the LearnTorch series. Prerequisites: Capstone 1 (benchmarking), Capstone 2 (training framework), Capstone 3 (transformer from scratch).*
