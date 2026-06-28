"""
CUDA Graph Overhead Benchmark

Measures CPU launch overhead and how CUDA Graphs eliminate it.

The key insight: CPU launch overhead is FIXED per kernel, regardless of
kernel size. So small-but-frequent kernels suffer most.

OVERHEAD PROFILE (RTX PRO 2000, rough estimates):
  CUDA kernel launch via Python:  ~10–20 μs per kernel
  CUDA Graph replay:              ~3–5 μs TOTAL (for any number of kernels)

  A GPT-2 medium forward pass: ~1,000 CUDA kernels
  Without graphs: 1000 * 15μs = 15 ms pure overhead
  With graphs:    5 μs total

  For a model that should run in 5 ms, you're spending 3× on overhead!

WHEN CUDA GRAPHS HELP MOST:
  ✓ Small-to-medium models (overhead is proportionally large)
  ✓ Inference with fixed batch size / sequence length
  ✓ Models with many small kernels (RNNs, residual blocks)

WHEN CUDA GRAPHS DON'T HELP:
  ✗ Models with dynamic shapes (recapture cost)
  ✗ Models with CPU logic (graphs can't capture)
  ✗ Very large kernels (compute-bound — overhead amortised)

Run:
  python benchmarks/cuda_graph_bench.py
"""

import torch
import torch.nn as nn
import time


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def bmark(fn, *args, n_warmup=20, n_iter=200, sync=True):
    for _ in range(n_warmup):
        fn(*args)
    if sync and DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn(*args)
    if sync and DEVICE == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iter * 1000


# ─────────────────────────────────────────────────────────────────────────────
# Count kernels: use PyTorch profiler to count kernel launches
# ─────────────────────────────────────────────────────────────────────────────

def count_kernels(model, x):
    """Run one forward pass under profiler and count GPU kernels launched."""
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        with torch.no_grad():
            model(x)

    events = prof.key_averages()
    # Count unique CUDA kernel events
    cuda_events = [e for e in events if e.self_device_time_total > 0]
    total_calls = sum(e.count for e in cuda_events)
    return total_calls, cuda_events[:5]


# ─────────────────────────────────────────────────────────────────────────────
# Model classes: small (overhead-dominated) vs large (compute-dominated)
# ─────────────────────────────────────────────────────────────────────────────

class SmallNet(nn.Module):
    """Many tiny ops — overhead-dominated."""
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(64, 64) for _ in range(32)])

    def forward(self, x):
        for l in self.layers:
            x = torch.relu(l(x))
        return x


class MediumNet(nn.Module):
    """Balanced compute and overhead."""
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(512, 512) for _ in range(16)])

    def forward(self, x):
        for l in self.layers:
            x = torch.gelu(l(x))
        return x


class LargeNet(nn.Module):
    """Large matmuls — compute-dominated."""
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(4096, 4096) for _ in range(4)])

    def forward(self, x):
        for l in self.layers:
            x = torch.gelu(l(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark: eager vs CUDA Graph vs compile+cudagraph
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_model(name, model, x_shape, n_iter=200):
    model = model.cuda().eval()
    x = torch.randn(*x_shape, device="cuda")

    results = {}

    # Eager
    with torch.no_grad():
        results["eager"] = bmark(model, x, n_iter=n_iter)

    # CUDA Graph (manual)
    static_x = x.clone()
    with torch.no_grad():
        for _ in range(3):
            model(static_x)   # warmup
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            static_out = model(static_x)

    def graphed(inp):
        static_x.copy_(inp)
        g.replay()
        return static_out

    with torch.no_grad():
        results["cuda_graph"] = bmark(graphed, x, n_iter=n_iter)

    # torch.compile (default: inductor)
    compiled = torch.compile(model)
    with torch.no_grad():
        results["compile"] = bmark(compiled, x, n_iter=n_iter)

    # torch.compile + cudagraphs
    compiled_cg = torch.compile(model, options={"triton.cudagraphs": True})
    with torch.no_grad():
        results["compile+cg"] = bmark(compiled_cg, x, n_iter=n_iter)

    return results


def main():
    if DEVICE != "cuda":
        print("CUDA required for CUDA Graphs")
        return

    print(f"\n{'═'*70}")
    print("  CUDA Graph Overhead Benchmark")
    print(f"{'═'*70}")

    models = [
        ("SmallNet  (32×Linear[64→64])",   SmallNet(),  (32, 64)),
        ("MediumNet (16×Linear[512→512])",  MediumNet(), (32, 512)),
        ("LargeNet  (4×Linear[4096→4096])", LargeNet(),  (32, 4096)),
    ]

    header = (f"  {'Model':<30}  {'Eager':>8}  {'Graph':>8}  "
              f"{'Compile':>9}  {'C+Graph':>9}  {'Graph↑':>7}  {'C+G↑':>7}")
    print(f"\n{header}")
    print("  " + "─" * 80)

    for name, model, shape in models:
        r = benchmark_model(name, model, shape)
        baseline = r["eager"]
        graph_speedup  = baseline / r["cuda_graph"]
        cg_speedup     = baseline / r["compile+cg"]
        print(f"  {name:<30}  "
              f"{r['eager']:>6.3f}ms  "
              f"{r['cuda_graph']:>6.3f}ms  "
              f"{r['compile']:>7.3f}ms  "
              f"{r['compile+cg']:>7.3f}ms  "
              f"{graph_speedup:>6.2f}×  "
              f"{cg_speedup:>6.2f}×")

    print(f"\n{'═'*70}")
    print("  READING THE RESULTS:")
    print("  • SmallNet:  high graph speedup — overhead-dominated")
    print("  • LargeNet:  low graph speedup — compute-dominated")
    print("  • compile+cg gets BOTH operator fusion AND zero launch overhead")
    print()
    print("  KERNEL COUNT ANALYSIS:")
    net = SmallNet().cuda().eval()
    x   = torch.randn(32, 64, device="cuda")
    with torch.no_grad():
        n_kernels, top5 = count_kernels(net, x)
    print(f"  SmallNet launches {n_kernels} CUDA kernels per forward pass")
    print(f"  At 15μs per launch: {n_kernels*15/1000:.1f} ms pure launch overhead")
    print(f"  CUDA Graph reduces that to ~5μs total")
    print(f"{'═'*70}")


if __name__ == "__main__":
    main()
