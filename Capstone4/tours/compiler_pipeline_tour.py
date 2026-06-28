"""
Tour: The PyTorch Compiler Pipeline

Walk every stage of the compilation stack for a single operation.
Python → Dynamo → FX → AOTAutograd → Inductor → Triton → CUDA → GPU

Run:
  python tours/compiler_pipeline_tour.py
"""

import torch
import torch.nn as nn
import torch.fx as fx
import os

COLS = 68

def box(title):
    line = "─" * (COLS - 2)
    print(f"\n┌{line}┐")
    pad = (COLS - 2 - len(title)) // 2
    print(f"│{' '*pad}{title}{' '*(COLS-2-pad-len(title))}│")
    print(f"└{line}┘")

def lesson(n, title):
    print(f"\n{'═'*COLS}")
    print(f"  Lesson {n}: {title}")
    print(f"{'═'*COLS}")

def explain(text):
    for line in text.strip().split("\n"):
        print(f"  {line}")
    print()

def show(label, value=""):
    print(f"  ► {label}")
    if value:
        for line in str(value).split("\n"):
            print(f"      {line}")


# ─────────────────────────────────────────────────────────────────────────────

lesson(1, "The Compilation Stack Overview")

explain("""
When you call torch.compile(model), PyTorch wraps it in a 4-stage pipeline:

  ┌─────────────┐
  │  Your Code  │  ← Python function, nn.Module, etc.
  └──────┬──────┘
         │  Python bytecode interception
  ┌──────▼──────┐
  │ TorchDynamo │  ← Traces Python, builds FX graph, installs guards
  └──────┬──────┘
         │  FX Graph (symbolic operations)
  ┌──────▼──────┐
  │ AOTAutograd │  ← Traces the BACKWARD pass too, ahead of time
  └──────┬──────┘
         │  Joint forward+backward FX graph
  ┌──────▼──────┐
  │  Inductor   │  ← Optimises graph: fusion, memory planning, layout
  └──────┬──────┘
         │  Triton (GPU) or C++ (CPU)
  ┌──────▼──────┐
  │  GPU/CPU    │  ← Native hardware execution
  └─────────────┘

Each stage is a compiler pass. Each can be inspected, replaced, or extended.
""")

# ─────────────────────────────────────────────────────────────────────────────

lesson(2, "TorchDynamo: Bytecode Interception")

explain("""
TorchDynamo does NOT parse Python source code.
It intercepts Python BYTECODE at runtime, instruction by instruction.

When Python runs a function, it executes bytecode like:
  LOAD_FAST   x         # push x onto stack
  LOAD_ATTR   shape     # push x.shape
  LOAD_METHOD relu      # find torch.relu
  CALL_METHOD 1         # call relu(x)

Dynamo replaces the Python frame evaluator with its own evaluator.
For tensor operations: records them into an FX graph.
For Python control flow: evaluates the condition (Python-level), marks a break.

KEY CONCEPT — Guards:
  Every compiled graph is protected by assertions:
    GUARD: x.shape == (64, 512)
    GUARD: x.dtype == torch.float32
    GUARD: x.device.type == 'cuda'
  If any guard fails on the next call → Dynamo recompiles.
  This is called "specialisation" — fast for one shape, recompile for new ones.
""")

def show_dynamo_guards():
    """Actually inspect guard generation."""
    import torch._dynamo

    def fn(x):
        return torch.relu(x @ x.T)

    x = torch.randn(8, 8, device="cuda" if torch.cuda.is_available() else "cpu")
    explanation = torch._dynamo.explain(fn)(x)

    show("Dynamo explanation for: relu(x @ x.T)")
    print(f"      graph_count: {explanation.graph_count}")
    print(f"      break_count: {explanation.break_count}")
    if hasattr(explanation, 'guards') and explanation.guards:
        print(f"      # guards:    {len(explanation.guards)}")
        for g in explanation.guards[:3]:
            print(f"      guard: {str(g)[:60]}")

show_dynamo_guards()

# ─────────────────────────────────────────────────────────────────────────────

lesson(3, "FX Graph: The Intermediate Representation")

explain("""
After Dynamo traces your code, it builds an FX Graph.
Think of it as the "abstract syntax tree" of PyTorch operations.

Node types:
  placeholder   → function inputs (tensors you pass in)
  get_attr      → model parameters (weights, biases)
  call_function → pure functions like torch.relu, operator.add
  call_method   → methods like .view(), .transpose()
  call_module   → sub-modules like nn.Linear, nn.LayerNorm
  output        → the return value

The GRAPH is a DAG: each node's inputs are edges to other nodes.
No Python control flow — it's a pure dataflow graph.
""")

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 16)
    def forward(self, x):
        return torch.relu(self.fc(x))

model = SimpleModel()
gm    = fx.symbolic_trace(model)

show("FX Graph for relu(linear(x)):")
for node in gm.graph.nodes:
    print(f"      {node.op:<16}  name={str(node.name):<20}  "
          f"target={str(node.target)[:30]}")

show("Generated Python code (gm.code):")
for line in gm.code.strip().split("\n"):
    print(f"      {line}")

# ─────────────────────────────────────────────────────────────────────────────

lesson(4, "AOTAutograd: Tracing the Backward")

explain("""
Standard autograd builds the backward graph LAZILY during the forward pass.
Each time you call forward, a new backward graph is created.
The compiler can't see this — it's different every call.

AOTAutograd solves this with 'functional' tracing:
  1. Express the model as a pure function: f(params, inputs) → output
  2. Use functorch's grad transform to trace the backward symbolically
  3. This gives TWO static FX graphs:
       • Forward:  (inputs, params) → (output, saved_activations)
       • Backward: (grad_out, saved_activations) → (grad_inputs, grad_params)

Both graphs are static — compiled ONCE, reused EVERY step.

This enables cross-forward-backward optimisations:
  • Fuse the loss computation into the backward graph
  • Plan memory layout for activation buffers
  • Schedule backward kernels to overlap with forward
""")

show("Example: AOTAutograd decomposes relu backward")
explain("""
  Forward:  out = relu(x)     saves: (x, out_mask = x > 0)
  Backward: grad_in = grad_out * out_mask

  AOTAutograd traces this as TWO separate FX graph nodes:
    fwd: relu → (output, mask)
    bwd: mul(grad_output, mask) → grad_input

  Inductor can then fuse the mul with other elementwise grad ops.
""")

# ─────────────────────────────────────────────────────────────────────────────

lesson(5, "TorchInductor: Loop Fusion and Codegen")

explain("""
Inductor takes the FX graph and lowers it to a LOOP-BASED IR.
Each FX node becomes a loop:

  FX: add(x, y)
  Inductor IR:
    for i in range(N):
        out[i] = x[i] + y[i]

  FX: relu(z)
  Inductor IR:
    for i in range(N):
        out2[i] = max(z[i], 0)

Inductor checks: do these loops have the SAME iteration space?
If yes → FUSE them:
    for i in range(N):
        tmp     = x[i] + y[i]
        out2[i] = max(tmp, 0)

This fused loop becomes ONE Triton kernel: one pass over memory.
Unfused: 2 passes (x,y → tmp, tmp → out2)
Fused:   1 pass  (x,y → out2 directly)

After fusion, Inductor generates Triton code and hands it to the Triton
compiler which produces PTX (CUDA assembly) for the GPU.
""")

show("See generated Triton code with:")
print("""
      TORCH_LOGS="output_code" python your_script.py

      # Or inspect the cache:
      ls ~/.cache/torch_extensions/inductor_*/

      # The file will look like:
      @triton.jit
      def triton_fused_add_relu(in_ptr0, in_ptr1, out_ptr0, xnumel, ...):
          xoffset = tl.program_id(0) * XBLOCK
          xindex  = xoffset + tl.arange(0, XBLOCK)
          xmask   = xindex < xnumel
          tmp0    = tl.load(in_ptr0 + xindex, xmask)
          tmp1    = tl.load(in_ptr1 + xindex, xmask)
          tmp2    = tmp0 + tmp1
          tmp3    = tl.maximum(tmp2, 0)
          tl.store(out_ptr0 + xindex, tmp3, xmask)
""")

# ─────────────────────────────────────────────────────────────────────────────

lesson(6, "CUDA Graphs: Zero-Overhead Replay")

explain("""
Even with Triton kernels, launching them from Python has overhead:
  Each kernel launch: ~15 μs (Python → CUDA driver → kernel scheduler)
  100 kernels: 1.5 ms of PURE overhead before any compute happens.

CUDA Graphs solve this by recording ALL launches once and replaying them
as a single GPU-side 'program' with ONE CPU call.

The flow:
  1. CAPTURE:  Tell CUDA to watch all GPU ops (but don't execute)
  2. RECORD:   Run the model normally — every kernel is recorded
  3. REPLAY:   One CPU call → GPU executes the entire recorded sequence

Constraints:
  • Input tensor ADDRESSES must be the same (same buffer, new data)
  • No CPU↔GPU sync inside the captured region (.item(), .cpu())
  • Shapes can't change (different shape → new graph)

torch.compile automatically uses CUDA Graphs when:
  options={"triton.cudagraphs": True}
  OR: model with fixed shapes + torch.compile
""")

show("The full pipeline in one call:")
print("""
      model = MyModel().cuda()
      x     = torch.randn(B, D, device='cuda')

      # torch.compile wraps ALL 4 stages:
      compiled = torch.compile(model,
                               mode='max-autotune',
                               options={'triton.cudagraphs': True})

      # First call: Dynamo traces → AOT traces → Inductor compiles → Graph captured
      # Subsequent calls: Guards check → CUDA Graph replay (5μs overhead)
      out = compiled(x)
""")

# ─────────────────────────────────────────────────────────────────────────────

lesson(7, "Putting It Together: A Complete Example")

explain("""
Let's trace a single model through the full pipeline.

Model: one_layer(x) = relu(x @ W.T + b)

  Stage 1 — Python bytecode (what you write):
    out = torch.relu(x @ W.T + b)

  Stage 2 — FX Graph (Dynamo output):
    %x        = placeholder
    %W        = get_attr(weight)
    %b        = get_attr(bias)
    %wt       = call_method(W, 'T')
    %xw       = call_function(torch.matmul, [x, wt])
    %xwb      = call_function(torch.add, [xw, b])
    %out      = call_function(torch.relu, [xwb])
    return %out

  Stage 3 — AOTAutograd (adds backward graph):
    fwd: saves x, W, relu_mask
    bwd: grad_b = grad_out.sum(0)
         grad_x = grad_out * mask @ W
         grad_W = (grad_out * mask).T @ x

  Stage 4 — Inductor IR (fuses add+relu, plans memory):
    for i in N:                    # fused add+relu kernel
        out[i] = max(xw[i]+b[i%D], 0)
    matmul(x, W.T, ...)            # separate GEMM kernel (can't fuse with matmul)

  Stage 5 — Triton code (one kernel for add+relu, cuBLAS for matmul):
    @triton.jit
    def fused_add_relu(xw_ptr, b_ptr, out_ptr, N, D, BLOCK: tl.constexpr):
        i   = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        xw  = tl.load(xw_ptr + i, i < N)
        b   = tl.load(b_ptr + i % D, i < N)
        tl.store(out_ptr + i, tl.maximum(xw + b, 0), i < N)

  Stage 6 — CUDA Graph (optional):
    Record: matmul kernel → fused_add_relu kernel
    Replay: 1 CPU call → 2 GPU kernels in sequence

  Stage 7 — GPU hardware:
    Matrix Tensor Cores execute the GEMM at peak throughput.
    All CUDA cores execute the fused elementwise at memory bandwidth.
""")

box("Tour Complete")
print("""
  Files to explore next:
    compiler/dynamo/basics.py       → torch.compile in practice
    compiler/dynamo/graph_breaks.py → what breaks compilation
    compiler/fx/graph_inspect.py    → inspect the FX graph
    compiler/aot/autograd.py        → trace the backward
    compiler/inductor/ir_inspect.py → see generated Triton code
    compiler/cuda_graphs/basics.py  → CUDA Graph APIs
    kernels/triton_basics.py        → write your first Triton kernel
""")
