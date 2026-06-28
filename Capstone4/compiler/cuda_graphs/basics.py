"""
CUDA Graphs — Eliminate CPU Launch Overhead

THE PROBLEM
────────────
Every CUDA kernel launch has CPU overhead:
  - Python → C++ dispatch: ~5–10 μs
  - CUDA driver setup:     ~2–5 μs
  - Kernel scheduling:     ~1–3 μs

For a large model like GPT-3, one forward pass launches ~10,000 CUDA kernels.
Total CPU launch overhead: 10,000 × 15 μs = 150 ms per step.
On an A100 that does the actual compute in 50 ms — you're 3× overhead-bound!

THE SOLUTION: CUDA GRAPHS
──────────────────────────
CUDA Graphs capture a SEQUENCE of CUDA operations once, then replay it
with a single CPU call. All the kernel launches happen on the GPU itself.

  Capture phase (once):
    • Record all CUDA operations to a 'graph'
    • No actual GPU work happens

  Replay phase (every iteration):
    • One CPU call: cudaGraphLaunch(graph)
    • GPU executes all operations in order
    • CPU overhead: ~1–5 μs TOTAL (instead of 150 ms)

CONSTRAINTS:
  • Input/output memory addresses must be FIXED (same buffers every call)
  • No dynamic control flow inside the captured region
  • No CPU↔GPU synchronisation inside capture
  • Shapes must not change (new shape = recapture)
  • PyTorch random ops need special handling (separate RNG state)

PYTORCH APIs:
  Low-level:  torch.cuda.CUDAGraph()   + g.capture_begin() / g.capture_end()
  High-level: torch.cuda.make_graphed_callables(model, sample_inputs)
  With compile: torch.compile(model, options={"triton.cudagraphs": True})

Run this file:
  python compiler/cuda_graphs/basics.py
"""

import torch
import torch.nn as nn
import time


class BenchModel(nn.Module):
    def __init__(self, d=512, n_layers=8):
        super().__init__()
        self.layers = nn.Sequential(*[
            nn.Sequential(nn.Linear(d, d), nn.GELU())
            for _ in range(n_layers)
        ])

    def forward(self, x):
        return self.layers(x)


def benchmark(fn, x, n_warmup=10, n_iter=100, label=""):
    for _ in range(n_warmup):
        fn(x)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn(x)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / n_iter * 1000
    print(f"  {label:<35}  {ms:.3f} ms/iter")
    return ms


def demo_manual_cuda_graph():
    """
    Manual CUDA Graph API — shows exactly what's happening.
    """
    print("\n── Manual CUDA Graph (low-level API) ────────────────────────────")
    model = BenchModel().cuda().eval()
    x     = torch.randn(32, 512, device="cuda")

    # ── Warmup (outside capture — initialises lazy ops like cuBLAS) ───────────
    with torch.no_grad():
        for _ in range(3):
            _ = model(x)

    # ── Capture ───────────────────────────────────────────────────────────────
    # IMPORTANT: capture uses static buffers. We must use the SAME x every call.
    static_x = x.clone()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        static_out = model(static_x)   # recorded, not executed

    # ── Replay ────────────────────────────────────────────────────────────────
    # Copy new data into static_x, then replay
    static_x.copy_(x)      # update the input buffer
    g.replay()             # ONE CPU call → ALL ops execute on GPU

    print(f"  Graph captured and replayed. Output shape: {static_out.shape}")
    print(f"  Max diff vs eager: {(model(x) - static_out).abs().max():.2e}")


def demo_make_graphed_callables():
    """
    torch.cuda.make_graphed_callables() — high-level API.
    Automatically handles the capture + replay pattern.
    """
    print("\n── make_graphed_callables (high-level) ──────────────────────────")
    model = BenchModel().cuda().eval()
    x     = torch.randn(32, 512, device="cuda")

    with torch.no_grad():
        # Wrap the model's forward pass in a CUDA Graph
        graphed_model = torch.cuda.make_graphed_callables(model, (x,))
        out = graphed_model(x)

    print(f"  Graphed model output shape: {out.shape}")
    eager_out = model(x)
    print(f"  Max diff vs eager: {(out - eager_out).abs().max():.2e}")


def demo_speedup():
    """
    Measure the actual launch overhead saved by CUDA Graphs.
    """
    print("\n── CUDA Graph speedup benchmark ─────────────────────────────────")
    model = BenchModel(d=512, n_layers=4).cuda().eval()
    x     = torch.randn(32, 512, device="cuda")

    # Eager
    with torch.no_grad():
        t_eager = benchmark(model, x, label="Eager (CPU launches per kernel)")

    # CUDA Graph
    static_x = x.clone()
    with torch.no_grad():
        for _ in range(3):
            model(static_x)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            static_out = model(static_x)

    def graphed_call(inp):
        static_x.copy_(inp)
        g.replay()
        return static_out

    with torch.no_grad():
        t_graph = benchmark(graphed_call, x, label="CUDA Graph (single launch)")

    # torch.compile with cudagraphs
    compiled = torch.compile(model, options={"triton.cudagraphs": True})
    with torch.no_grad():
        for _ in range(3):
            compiled(x)
        t_compile = benchmark(compiled, x, label="torch.compile + cudagraphs")

    print()
    print(f"  CUDA Graph speedup:    {t_eager/t_graph:.2f}×")
    print(f"  compile+graphs speedup:{t_eager/t_compile:.2f}×")
    print()
    print("  Note: speedup is largest when the model is compute-light")
    print("  (CPU launch overhead dominates). For large models, compute dominates.")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA required for CUDA Graphs")
    else:
        demo_manual_cuda_graph()
        demo_make_graphed_callables()
        demo_speedup()
        print("\nNext: kernels/triton_basics.py")
