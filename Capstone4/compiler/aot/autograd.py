"""
AOTAutograd — Trace the Backward Pass Ahead of Time

THE PROBLEM AOTAutograd SOLVES
───────────────────────────────
torch.compile() needs to optimise the ENTIRE training step, including backward.
But PyTorch's autograd engine computes gradients lazily, using a dynamic
computation graph built during the forward pass. That graph is recreated every
iteration — the compiler can't see it ahead of time.

SOLUTION: AOTAutograd (Ahead-Of-Time Autograd)
  Uses torch.autograd.functional.vjp (vector-Jacobian product) to trace the
  backward pass as a STATIC FX graph before the first real execution.

  Result: two separate compiled graphs:
    1. Forward graph  — takes (inputs, params) → (outputs, saved_for_backward)
    2. Backward graph — takes (grad_output, saved_for_backward) → (grad_inputs, grad_params)

  Both graphs are then handed to Inductor for kernel codegen.

WHAT THIS ENABLES:
  • Operator fusion across forward AND backward (e.g. fused gradient update)
  • Memory planning: decide which activations to save vs recompute
  • Activation checkpointing at the compiler level
  • Kernel scheduling to overlap compute and memory ops

HOW TO USE IT DIRECTLY:
  from functorch.compile import aot_function
  compiled_fn = aot_function(fn, fw_compiler, bw_compiler)

  In practice, torch.compile() calls this for you automatically.

Run this file:
  python compiler/aot/autograd.py
"""

import torch
import torch.nn as nn
from functorch.compile import aot_function, make_boxed_compiler


def print_graph(name, gm, _):
    """Compiler that just prints the graph, doesn't compile it."""
    print(f"\n  ── {name} graph ──────────────────────────────────────────────")
    for node in gm.graph.nodes:
        print(f"    {str(node.name):<30}  {node.op:<16}  {str(node.target)[:40]}")
    return gm.forward   # return eager forward


def demo_aot_trace():
    """Capture forward and backward graphs for a simple function."""
    print("\n── AOTAutograd: trace forward + backward ─────────────────────────")

    def fn(x, w):
        # Simple: linear + relu
        return torch.relu(x @ w.T)

    captured_graphs = []

    def fw_compiler(gm, _):
        captured_graphs.append(("forward", list(gm.graph.nodes)))
        return gm.forward

    def bw_compiler(gm, _):
        captured_graphs.append(("backward", list(gm.graph.nodes)))
        return gm.forward

    compiled = aot_function(fn, fw_compiler=fw_compiler, bw_compiler=bw_compiler)

    x = torch.randn(4, 8,  requires_grad=True)
    w = torch.randn(16, 8, requires_grad=True)
    out = compiled(x, w)
    out.sum().backward()

    for graph_name, nodes in captured_graphs:
        print(f"\n  {graph_name.upper()} GRAPH  ({len(nodes)} nodes):")
        for node in nodes:
            if node.op != "placeholder" and node.op != "output":
                print(f"    {str(node.name):<30}  {str(node.target)[:40]}")


def demo_saved_tensors():
    """
    AOTAutograd decides which activations to save for backward.
    For relu: save input x (to compute grad: grad * (x > 0))
    For matmul(a, b): save both a and b
    This is the 'remat strategy' — recompute vs save trade-off.
    """
    print("\n── What AOTAutograd saves for backward ───────────────────────────")
    print("""
  Forward:   relu(x @ w.T)

  To compute backward we need:
    ∂L/∂w = x.T @ grad_output @ relu_mask(x @ w.T)
    ∂L/∂x = grad_output @ relu_mask(x @ w.T) @ w

  AOTAutograd saves:
    • The input to relu  (x @ w.T, shape (B, out))  — to compute relu mask
    • x                  (shape (B, in))             — to compute ∂L/∂w
    • w                  (shape (out, in))            — to compute ∂L/∂x

  Total saved: 3 tensors.

  With 'activation checkpointing' (gradient checkpointing):
    Don't save relu input — RECOMPUTE it during backward.
    Trades memory for compute: O(1) saved tensors instead of O(1).

  torch.compile() integrates with checkpointing:
    torch.compile(use_full_graph=True) + torch.utils.checkpoint
""")


def demo_functorch():
    """
    AOTAutograd is built on functorch (now in PyTorch as torch.func).
    functorch provides function transforms that compose:
      • vmap      — vectorise over a batch dimension
      • grad      — compute gradients of a function
      • jacfwd    — forward-mode Jacobian
      • jacrev    — reverse-mode Jacobian

    These make the backward graph explicit rather than implicit.
    """
    print("\n── torch.func (functorch) transforms ────────────────────────────")

    def loss_fn(params, x):
        w, b = params
        return ((x @ w + b) ** 2).mean()

    w = torch.randn(4, 4, requires_grad=False)
    b = torch.randn(4, requires_grad=False)
    x = torch.randn(8, 4)

    # Compute gradient with respect to params using grad transform
    grad_fn = torch.func.grad(loss_fn)
    grads   = grad_fn((w, b), x)

    print(f"  grad w.r.t w: shape={grads[0].shape}  norm={grads[0].norm():.4f}")
    print(f"  grad w.r.t b: shape={grads[1].shape}  norm={grads[1].norm():.4f}")

    # vmap: vectorise grad over a batch of different xs
    batched_grad = torch.func.vmap(grad_fn, in_dims=(None, 0))
    x_batch = torch.randn(5, 8, 4)   # 5 different xs
    batch_grads = batched_grad((w, b), x_batch)
    print(f"\n  Batched grad (vmap over 5 xs):")
    print(f"    grad_w shape: {batch_grads[0].shape}  (5 independent gradients)")


if __name__ == "__main__":
    demo_aot_trace()
    demo_saved_tensors()
    demo_functorch()
    print("\nNext: compiler/inductor/ir_inspect.py")
