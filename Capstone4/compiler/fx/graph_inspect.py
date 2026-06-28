"""
FX Graphs — The Intermediate Representation

After TorchDynamo traces your code, it produces an FX Graph.
An FX Graph is the backbone of the entire torch.compile pipeline.

WHAT IS AN FX GRAPH?
  A directed acyclic graph (DAG) where:
    • Each NODE is one operation (a PyTorch op, a Python call, or a constant)
    • Each EDGE is a tensor (flows from producer node to consumer node)
    • No Python control flow — all branches have been resolved

NODE TYPES:
  placeholder  — function input (a tensor passed in)
  get_attr     — read a model parameter or buffer from self
  call_function — a standalone function like torch.relu, operator.add
  call_method  — a method call like .view(), .transpose()
  call_module  — call a sub-module (nn.Linear, nn.LayerNorm, etc.)
  output       — the function's return value

WHY FX?
  Once you have a graph, you can:
    • Inspect it (print it, visualise it)
    • Rewrite it (graph transforms / passes)
    • Optimise it (fuse ops, eliminate redundancy)
    • Compile it to Triton / C++ / ONNX / anything

  FX is PyTorch's IR — everything above it is Python, everything below is native code.

TWO WAYS TO GET AN FX GRAPH:
  1. torch.fx.symbolic_trace(model)         — symbolic, works on simple models
  2. torch._dynamo.export(fn)(inputs)       — Dynamo capture (handles more cases)

Run this file:
  python compiler/fx/graph_inspect.py
"""

import torch
import torch.nn as nn
import torch.fx as fx


# ── Method 1: symbolic_trace ──────────────────────────────────────────────────

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 8)
        self.bn     = nn.BatchNorm1d(8)

    def forward(self, x):
        return self.bn(torch.relu(self.linear(x)))


def demo_symbolic_trace():
    print("\n── torch.fx.symbolic_trace ──────────────────────────────────────")
    model = SimpleNet()
    graph_module = fx.symbolic_trace(model)

    print(f"  Type: {type(graph_module)}")
    print()
    print("  FX Graph nodes:")
    print(f"  {'#':>3}  {'op':>14}  {'name':>18}  {'target':>25}  args")
    print("  " + "─" * 85)

    for i, node in enumerate(graph_module.graph.nodes):
        args_str = str(node.args)[:40]
        print(f"  {i:>3}  {node.op:>14}  {str(node.name):>18}  "
              f"{str(node.target):>25}  {args_str}")

    print()
    print("  Printed graph (code form):")
    graph_module.graph.print_tabular()


def demo_graph_to_code():
    """
    FX can convert a graph back to Python code — invaluable for debugging.
    The generated code is the 'canonical form' of what the model computes.
    """
    print("\n── Graph → Python code ──────────────────────────────────────────")
    model = SimpleNet()
    gm    = fx.symbolic_trace(model)

    print("  Generated Python (gm.code):")
    for line in gm.code.strip().split("\n"):
        print(f"    {line}")


def demo_node_properties():
    """
    Each node carries rich metadata: shape, dtype, stride of its output tensor.
    This is called 'shape propagation' and is essential for kernel codegen.
    """
    print("\n── Node metadata (shapes / dtypes) ──────────────────────────────")

    class ShapeNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8, 16)
        def forward(self, x):
            return torch.relu(self.fc(x))

    model    = ShapeNet()
    gm       = fx.symbolic_trace(model)

    # Run shape propagation: gives every node an 'example_value' with meta
    ex_input = torch.randn(4, 8)
    interp   = fx.Interpreter(gm)
    interp.run(ex_input)   # populates node values in the interpreter

    print(f"  {'Node':<20}  op")
    for node in gm.graph.nodes:
        print(f"  {str(node.name):<20}  {node.op}")


def demo_dynamo_export():
    """
    torch._dynamo.export() is more powerful than symbolic_trace:
    handles models with data-dependent code, guards, etc.
    This is what torch.compile uses internally.
    """
    print("\n── torch._dynamo.export() ───────────────────────────────────────")

    def fn(x, y):
        return torch.relu(x + y) * 2

    x = torch.randn(4, 4)
    y = torch.randn(4, 4)

    exported = torch.export.export(fn, (x, y))
    print(f"  Exported program type: {type(exported)}")
    print()
    print("  Graph:")
    exported.graph_module.graph.print_tabular()


def demo_custom_pass():
    """
    FX passes are graph transformations: walk nodes, rewrite, delete, insert.
    This is how Inductor implements operator fusion.
    """
    print("\n── Custom FX pass: eliminate double-negation ─────────────────────")

    def fn(x):
        return torch.neg(torch.neg(x))   # -(-x) == x

    gm, guards = torch._dynamo.export(fn)(torch.randn(4))
    print("  Before pass:")
    gm.graph.print_tabular()

    # Walk the graph and replace neg(neg(x)) → x
    for node in list(gm.graph.nodes):
        if (node.op == "call_function" and
                node.target == torch.neg and
                len(node.args) == 1):
            inner = node.args[0]
            if (inner.op == "call_function" and inner.target == torch.neg):
                # Replace node with inner's input (skip double neg)
                node.replace_all_uses_with(inner.args[0])
                gm.graph.erase_node(node)
                gm.graph.erase_node(inner)
                break

    gm.graph.lint()   # validate graph is still legal
    gm.recompile()    # regenerate Python code

    print("\n  After pass (double neg removed):")
    gm.graph.print_tabular()

    x = torch.randn(4)
    original_out = fn(x)
    optimised_out = gm(x)
    print(f"\n  Max diff: {(original_out - optimised_out).abs().max():.2e}  (should be 0)")


if __name__ == "__main__":
    demo_symbolic_trace()
    demo_graph_to_code()
    demo_node_properties()
    demo_dynamo_export()
    demo_custom_pass()
    print("\nNext: compiler/inductor/ir_inspect.py")
