"""
Graph Breaks — What Stops the Compiler and How to Fix Them

A graph break happens when TorchDynamo encounters Python code it can't
symbolically trace. It compiles what it has, runs the problematic Python
eagerly, then resumes compilation after.

GRAPH BREAK COST:
  Each break means:
    • A round-trip from GPU → CPU (synchronise, run Python, re-dispatch)
    • The optimizer can't fuse ops across the break boundary
    • Dynamo recompiles surrounding graphs more often

  One graph break in a hot loop can kill all compilation gains.

COMMON CAUSES:
  1. .item() / .tolist()      — extract scalar/list from tensor (data-dependent)
  2. Python print on tensors  — calls repr(), needs real value
  3. if tensor_val > 0        — data-dependent Python branch
  4. for i in range(tensor):  — dynamic loop bound
  5. external C extensions    — not torch, can't trace
  6. unsupported ops          — some ops aren't traceable yet
  7. global/closured non-tensor state mutations

HOW TO DETECT BREAKS:
  torch._dynamo.explain(fn)(inputs)   → structured report
  TORCH_LOGS="graph_breaks" python... → verbose log every break
  torch.compile(fn, fullgraph=True)   → error on first break

HOW TO FIX BREAKS:
  • Replace .item() comparisons with torch ops: use torch.where() instead of if
  • Replace Python loops on tensors with tensor ops
  • Mark helper functions with @torch.compiler.disable to skip compilation
  • Use torch._dynamo.mark_dynamic(tensor, dim) for dynamic shapes

Run this file:
  python compiler/dynamo/graph_breaks.py
"""

import torch
import torch.nn as nn
import torch._dynamo


def count_graphs(fn, x):
    """Count how many sub-graphs Dynamo creates for fn."""
    explanation = torch._dynamo.explain(fn)(x)
    return explanation.graph_count, explanation.graph_break_count


# ── Break type 1: .item() ────────────────────────────────────────────────────

def fn_with_item(x):
    """Common pattern: check tensor value with .item() → breaks graph."""
    if x.mean().item() > 0:   # ← GRAPH BREAK: calls CPU, exits graph
        return x * 2
    return x * 0.5

def fn_without_item(x):
    """Fix: replace if/.item() with torch.where() — stays in graph."""
    cond = (x.mean() > 0).float()   # scalar tensor, no Python branch
    return torch.where(cond.bool(), x * 2, x * 0.5)


# ── Break type 2: data-dependent loop ────────────────────────────────────────

def fn_dynamic_loop(x):
    """Loop bound from tensor → Dynamo can't unroll → break."""
    n = x.size(0)
    out = x
    for _ in range(n):          # ← n is a Python int (OK only if static)
        out = out + 1
    return out
    # This actually works IF n is always the same (Dynamo specialises on n).
    # It breaks if n varies between calls (dynamic shapes).

def fn_static_equiv(x):
    """Fix: express as tensor op instead of Python loop."""
    return x + x.size(0)       # no loop at all


# ── Break type 3: non-torch function call ────────────────────────────────────

import numpy as np

def fn_numpy_in_middle(x):
    """Mixing numpy into a tensor computation → graph break."""
    x_np = x.cpu().numpy()     # ← BREAK: exits PyTorch graph
    x_np = np.clip(x_np, 0, 1)
    return torch.from_numpy(x_np).to(x.device)

def fn_torch_only(x):
    """Fix: use torch.clamp — stays in graph, compilable."""
    return torch.clamp(x, 0, 1)


# ── Break type 4: print statement ────────────────────────────────────────────

def fn_with_print(x):
    """print() forces .item()/.repr() internally → break."""
    print(f"  shape: {x.shape}, mean: {x.mean():.3f}")  # ← BREAK
    return x * 2

def fn_without_print(x):
    """Fix: remove debug prints from hot path, or use hooks."""
    return x * 2


# ── Demonstration ─────────────────────────────────────────────────────────────

def demo():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(32, 32, device=device)

    cases = [
        ("fn_with_item    ", fn_with_item,      "data-dependent .item() branch"),
        ("fn_without_item ", fn_without_item,   "torch.where() fix"),
        ("fn_dynamic_loop ", fn_dynamic_loop,   "Python loop (specialised on n=32)"),
        ("fn_torch_only   ", fn_static_equiv,   "tensor op replacement"),
    ]

    print("\n── Graph break analysis ─────────────────────────────────────────")
    print(f"  {'Function':<22}  {'Graphs':>6}  {'Breaks':>6}  Note")
    print("  " + "─" * 65)

    for name, fn, note in cases:
        try:
            graphs, breaks = count_graphs(fn, x)
            print(f"  {name}  {graphs:>6}  {breaks:>6}  {note}")
        except Exception as e:
            print(f"  {name}  ERROR: {str(e)[:50]}")

    print()
    print("  Goal: 1 graph, 0 breaks → entire function compiled as one unit.")
    print("  More graphs = more CPU↔GPU sync points = less speedup.")

    # Verify functional equivalence of fixed versions
    print("\n── Verify fixes produce same output ──────────────────────────────")
    x_cpu = torch.randn(32, 32)
    print(f"  item vs where:  max_diff={( fn_with_item(x_cpu) - fn_without_item(x_cpu) ).abs().max():.2e}")
    print(f"  numpy vs clamp: max_diff={( fn_numpy_in_middle(x_cpu) - fn_torch_only(x_cpu) ).abs().max():.2e}")


if __name__ == "__main__":
    demo()
    print("\nNext: compiler/fx/graph_inspect.py")
