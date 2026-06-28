"""
TorchDynamo — PyTorch's Python Bytecode Compiler

THE BIG PICTURE
───────────────
torch.compile() is NOT magic. It works by:

  1. TorchDynamo  intercepts Python execution at the bytecode level,
                  traces your code into an FX graph, stops at "breaks"
  2. AOTAutograd  traces the backward pass too (ahead of time)
  3. TorchInductor generates Triton (GPU) or C++ (CPU) from the graph
  4. The compiled kernel is cached and reused every future call

WHY BYTECODE INTERCEPTION?
  Python is dynamic — you can't know at definition time what shapes/dtypes
  tensors will have, whether Python control flow depends on tensor values, etc.
  Dynamo runs your code in a special tracing mode where every Python bytecode
  instruction is watched. When it sees PyTorch ops, it records them. When it
  sees Python-level control flow (if/for/while), it stops and "breaks" the graph.

GUARDS:
  Every compiled graph is protected by guards — assertions that the compiled
  version is still valid. Common guards:
    • tensor shape:  x.shape == (32, 512)
    • tensor dtype:  x.dtype == torch.float16
    • Python values: batch_size == 32

  If guards fail (e.g. you call with a different batch size), Dynamo recompiles.
  This is the "specialisation" trade-off: faster execution for one input shape,
  but recompile cost if shapes change.

Run this file to see torch.compile in action:
  python compiler/dynamo/basics.py
"""

import torch
import torch.nn as nn


# ── A simple model to compile ─────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 2048)
        self.fc2 = nn.Linear(2048, 512)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def demo_basic_compile():
    """Show that torch.compile produces the same output, faster."""
    print("\n── Basic torch.compile ──────────────────────────────────────────")
    model = MLP().cuda()
    x     = torch.randn(64, 512, device="cuda")

    eager_out = model(x)

    # compile() returns a drop-in replacement — same API, same output
    compiled_model = torch.compile(model)
    compiled_out   = compiled_model(x)

    print(f"  Max diff eager vs compiled: {(eager_out - compiled_out).abs().max():.2e}")
    print("  Output is numerically identical (just faster after warmup).")


def demo_backends():
    """
    torch.compile supports multiple backends. The default is 'inductor'.

    Backends:
      inductor      — default, generates Triton (GPU) / C++ (CPU), best perf
      eager         — fallback, runs ops one by one in Python (no speedup)
      aot_eager     — AOTAutograd only, no kernel codegen, good for debugging
      cudagraphs    — wrap in CUDA Graphs, no fusion, just capture overhead
      onnxrt        — export to ONNX and run in ONNXRuntime
      openxla       — Google XLA backend (for TPUs / experimental)
    """
    print("\n── Compiler backends ────────────────────────────────────────────")
    model = MLP().cuda()
    x     = torch.randn(64, 512, device="cuda")

    for backend in ["eager", "aot_eager", "inductor"]:
        compiled = torch.compile(model, backend=backend)
        out = compiled(x)
        print(f"  backend={backend:<12}  output shape={out.shape}  ✓")


def demo_fullgraph():
    """
    fullgraph=True tells Dynamo: fail loudly if the graph can't be compiled
    as a single unit (i.e., any graph break = error).

    Use this to discover and fix graph breaks in production code.
    """
    print("\n── fullgraph=True (strict mode) ─────────────────────────────────")

    def clean_fn(x):
        # Pure tensor ops — compiles to a single graph
        return torch.relu(x @ x.T)

    compiled_clean = torch.compile(clean_fn, fullgraph=True)
    x = torch.randn(32, 32, device="cuda")
    out = compiled_clean(x)
    print(f"  clean function: compiled OK, output shape={out.shape}")

    def broken_fn(x):
        # .item() extracts a Python scalar — breaks the graph
        if x.sum().item() > 0:   # data-dependent Python branch → graph break
            return x * 2
        return x * -1

    compiled_broken = torch.compile(broken_fn, fullgraph=False)  # OK with breaks
    out2 = compiled_broken(x)
    print(f"  broken function with fullgraph=False: ran OK (compiled in pieces)")

    try:
        compiled_strict = torch.compile(broken_fn, fullgraph=True)
        compiled_strict(x)
    except Exception as e:
        print(f"  broken function with fullgraph=True: ERROR as expected")
        print(f"    {type(e).__name__}: {str(e)[:80]}...")


def demo_explain():
    """
    torch._dynamo.explain() tells you HOW Dynamo compiled your function:
    number of graphs, where breaks occurred, what guards were generated.
    """
    print("\n── torch._dynamo.explain() ──────────────────────────────────────")

    def fn_with_breaks(x):
        x = x * 2
        if x.sum().item() > 0:   # break here
            x = x + 1
        return x * x

    x = torch.randn(8, 8, device="cuda")
    explanation = torch._dynamo.explain(fn_with_breaks)(x)
    print(f"  Number of graphs (sub-graphs): {explanation.graph_count}")
    print(f"  Number of graph breaks:        {explanation.graph_break_count}")
    for i, reason in enumerate(explanation.break_reasons):
        print(f"  Break {i}: {reason.reason[:80]}")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available — switch device to 'cpu' and remove .cuda() calls")
    else:
        demo_basic_compile()
        demo_backends()
        demo_fullgraph()
        demo_explain()
        print("\nDone. Next: compiler/dynamo/graph_breaks.py")
