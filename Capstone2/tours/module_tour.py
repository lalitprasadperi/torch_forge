"""
module_tour.py — nn.Module Internals: A Deep Dive

Run with:
  python tours/module_tour.py

What you'll learn:
  Lesson 1 — What is nn.Module?
  Lesson 2 — Parameters vs Buffers
  Lesson 3 — state_dict() and load_state_dict()
  Lesson 4 — Named submodules and the module tree
  Lesson 5 — Training vs Eval mode
  Lesson 6 — apply() — recursive weight initialisation
  Lesson 7 — Custom forward() — how control flows
"""

import sys
import torch
import torch.nn as nn
from collections import OrderedDict

COLS = 68


# ── Narration helpers ─────────────────────────────────────────────────────────

def lesson(n, title):
    print()
    print("╔" + "═" * (COLS - 2) + "╗")
    print(f"║  LESSON {n}: {title:<{COLS - 14}}║")
    print("╚" + "═" * (COLS - 2) + "╝")

def explain(*lines):
    for line in lines:
        print(f"  {line}")
    print()

def show(label, value):
    print(f"  {'▶ ' + label:<30} {value}")

def divider():
    print("  " + "─" * (COLS - 4))


# ── Lessons ───────────────────────────────────────────────────────────────────

def lesson_1_what_is_module():
    lesson(1, "What is nn.Module?")
    explain(
        "nn.Module is the base class for EVERY neural network in PyTorch.",
        "It provides automatic parameter tracking, device movement, and serialisation.",
        "",
        "You subclass it and implement forward() — PyTorch handles everything else.",
    )

    class TinyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 2)   # weight (2×3) + bias (2,)
            self.relu   = nn.ReLU()

        def forward(self, x):
            return self.relu(self.linear(x))

    net = TinyNet()
    x   = torch.tensor([[1.0, 2.0, 3.0]])
    out = net(x)     # calls net.__call__() which calls net.forward()

    explain("  Code:")
    print("    class TinyNet(nn.Module):")
    print("        def __init__(self):")
    print("            super().__init__()")
    print("            self.linear = nn.Linear(3, 2)")
    print("            self.relu   = nn.ReLU()")
    print("        def forward(self, x):")
    print("            return self.relu(self.linear(x))")
    print()
    show("Input  shape :", x.shape)
    show("Output shape :", out.shape)
    show("Output       :", out)
    show("net.training :", net.training)  # True by default


def lesson_2_parameters_vs_buffers():
    lesson(2, "Parameters vs Buffers")
    explain(
        "Parameters: tensors that NEED gradients and are UPDATED by the optimizer.",
        "  • Registered automatically when you assign nn.Parameter to a module attribute.",
        "  • Or implicitly when a sub-module (like nn.Linear) stores them.",
        "  • Saved in state_dict(). Returned by model.parameters().",
        "",
        "Buffers: tensors that are SAVED (in state_dict) but NOT updated by optimizer.",
        "  • Examples: BatchNorm running_mean, running_var.",
        "  • Register with: self.register_buffer('name', tensor)",
        "  • Moved to GPU with model.to(device) — just like parameters.",
    )

    class ModelWithBuffer(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(3, 3))
            self.register_buffer("running_sum", torch.zeros(3))

        def forward(self, x):
            self.running_sum += x.sum(0).detach()
            return x @ self.weight.T

    m = ModelWithBuffer()

    explain("  Inspecting parameters:")
    for name, param in m.named_parameters():
        print(f"    {name:<25}  shape={list(param.shape)}  "
              f"requires_grad={param.requires_grad}")

    print()
    explain("  Inspecting buffers:")
    for name, buf in m.named_buffers():
        print(f"    {name:<25}  shape={list(buf.shape)}  "
              f"requires_grad={buf.requires_grad}")

    print()
    explain(
        "  Key difference:",
        "  • parameter.requires_grad = True  → gradient flows, optimizer updates it",
        "  • buffer.requires_grad    = False → no gradient, no optimizer update",
        "                                      but SAVED in state_dict for inference",
    )


def lesson_3_state_dict():
    lesson(3, "state_dict() and load_state_dict()")
    explain(
        "state_dict() returns an OrderedDict of {name: tensor} for all",
        "parameters AND buffers. This is what you save to disk as a .pt file.",
        "",
        "load_state_dict() copies tensors from a state dict INTO the model in-place.",
        "It does NOT create a new model — just populates an existing one.",
    )

    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )

    sd = model.state_dict()
    explain("  state_dict() keys and shapes:")
    for k, v in sd.items():
        print(f"    {k:<30}  shape={list(v.shape)}")

    explain("",
            "  Saving and loading:",
            "    torch.save(model.state_dict(), 'model.pt')   # save",
            "    state = torch.load('model.pt')                # load",
            "    model.load_state_dict(state)                  # apply",
            "",
            "  strict=True (default): every key must match exactly.",
            "  strict=False: allows partial loading (e.g. pretrained backbone",
            "  without the classification head).",
            )

    # Demonstrate: save, corrupt one key, try strict vs non-strict
    import copy
    sd_copy = copy.deepcopy(sd)
    del sd_copy["2.weight"]   # remove the last linear's weight
    try:
        model.load_state_dict(sd_copy, strict=True)
    except RuntimeError as e:
        print(f"  strict=True error (expected): {str(e)[:80]}...")

    model.load_state_dict(sd_copy, strict=False)
    print(f"  strict=False: loaded OK (missing key silently skipped)")


def lesson_4_module_tree():
    lesson(4, "Named Submodules — The Module Tree")
    explain(
        "When you assign an nn.Module as an attribute, PyTorch registers it as",
        "a child module. The full tree is traversable with named_modules().",
        "",
        "This is how .to(device) and .parameters() work recursively — they",
        "walk the entire tree and apply the operation to every node.",
    )

    class ConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.bn   = nn.BatchNorm2d(out_ch)
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.relu(self.bn(self.conv(x)))

    class TinyCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.block1 = ConvBlock(3, 16)
            self.block2 = ConvBlock(16, 32)
            self.head   = nn.Linear(32, 10)

        def forward(self, x):
            x = self.block2(self.block1(x))
            return self.head(x.flatten(1))

    net = TinyCNN()

    explain("  named_modules() — the full tree:")
    for name, module in net.named_modules():
        indent = "    " + "  " * name.count(".")
        print(f"{indent}{name or '(root)':<30}  [{module.__class__.__name__}]")

    print()
    explain("  named_children() — direct children only:")
    for name, child in net.named_children():
        print(f"    {name:<20}  [{child.__class__.__name__}]")

    total = sum(p.numel() for p in net.parameters())
    show("\n  Total parameters", f"{total:,}")


def lesson_5_train_eval_mode():
    lesson(5, "Training vs Eval Mode")
    explain(
        "model.train()  — sets model.training = True",
        "model.eval()   — sets model.training = False",
        "",
        "Two layers behave DIFFERENTLY in train vs eval:",
        "",
        "  BatchNorm:",
        "    train: normalises using CURRENT BATCH mean/var",
        "           updates running_mean, running_var with EMA",
        "    eval : normalises using STORED running_mean, running_var",
        "           (accumulated during training)",
        "",
        "  Dropout:",
        "    train: randomly zeroes activations with probability p",
        "           scales remaining by 1/(1-p) to preserve expected value",
        "    eval : passes all activations unchanged",
    )

    model = nn.Sequential(
        nn.Linear(4, 4),
        nn.BatchNorm1d(4),
        nn.ReLU(),
        nn.Dropout(p=0.5),
    )

    x = torch.ones(8, 4)   # 8 samples, 4 features

    model.train()
    out_train_a = model(x)
    out_train_b = model(x)  # different due to Dropout

    model.eval()
    with torch.no_grad():
        out_eval_a = model(x)
        out_eval_b = model(x)  # same every time

    show("train mode — run 1, row 0 :", out_train_a[0].detach().round(decimals=3))
    show("train mode — run 2, row 0 :", out_train_b[0].detach().round(decimals=3))
    show("eval  mode — run 1, row 0 :", out_eval_a[0].detach().round(decimals=3))
    show("eval  mode — run 2, row 0 :", out_eval_b[0].detach().round(decimals=3))
    explain("",
            "  Train outputs differ (Dropout). Eval outputs are identical.",
            "  ALWAYS call model.eval() before inference or validation!")


def lesson_6_apply():
    lesson(6, "model.apply() — Recursive Weight Init")
    explain(
        "apply(fn) calls fn(module) on every node in the module tree,",
        "depth-first (leaves first). Used for weight initialisation.",
        "",
        "Pattern:",
        "  def init_weights(m):",
        "      if isinstance(m, nn.Linear):",
        "          nn.init.kaiming_normal_(m.weight)",
        "  model.apply(init_weights)",
    )

    net = nn.Sequential(
        nn.Linear(4, 8), nn.ReLU(),
        nn.Linear(8, 4), nn.ReLU(),
        nn.Linear(4, 2),
    )

    def print_stats(label):
        for name, p in net.named_parameters():
            if "weight" in name:
                print(f"    {label} {name:<30}  "
                      f"mean={p.data.mean():.3f}  std={p.data.std():.3f}")

    explain("  BEFORE init:")
    print_stats("")

    def kaiming_init(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            nn.init.zeros_(m.bias)

    net.apply(kaiming_init)

    print()
    explain("  AFTER Kaiming He init:")
    print_stats("")
    explain("",
            "  After init, std ≈ sqrt(2/fan_in). This keeps the activation",
            "  variance stable across layers with ReLU, preventing vanishing grads.")


def lesson_7_custom_forward():
    lesson(7, "Custom forward() — Branching, Skip Connections, Inspection")
    explain(
        "forward() is not just 'apply layers in sequence'. You can:",
        "  • Use if/else for dynamic architecture",
        "  • Return multiple tensors (logits + auxiliary loss)",
        "  • Add residual connections",
        "  • Store intermediate activations for inspection",
    )

    class InspectableResBlock(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.fc1      = nn.Linear(dim, dim)
            self.fc2      = nn.Linear(dim, dim)
            self.relu     = nn.ReLU()
            self.last_residual: torch.Tensor | None = None

        def forward(self, x):
            residual = x                           # skip connection
            out      = self.relu(self.fc1(x))
            out      = self.fc2(out)
            self.last_residual = residual.detach() # store for inspection
            return self.relu(out + residual)       # residual add

    block = InspectableResBlock(dim=4)
    x     = torch.randn(2, 4)
    out   = block(x)

    show("Input  :", x)
    show("Output :", out)
    show("Stored residual :", block.last_residual)

    explain("",
            "  The output depends on BOTH the transformed path AND the skip path.",
            "  During backprop, gradients flow through the residual path unchanged,",
            "  giving early layers a direct gradient signal regardless of depth.",
            )


# ── Run all lessons ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("╔" + "═" * (COLS - 2) + "╗")
    print(f"║  nn.Module Internals — A Deep Dive{' ' * (COLS - 38)}║")
    print(f"║  7 lessons on Parameters, Buffers, state_dict, and hooks{' ' * (COLS - 59)}║")
    print("╚" + "═" * (COLS - 2) + "╝")

    lesson_1_what_is_module()
    lesson_2_parameters_vs_buffers()
    lesson_3_state_dict()
    lesson_4_module_tree()
    lesson_5_train_eval_mode()
    lesson_6_apply()
    lesson_7_custom_forward()

    print()
    print("╔" + "═" * (COLS - 2) + "╗")
    print(f"║  ALL LESSONS COMPLETE {'─' * 10} Next: tours/hooks_demo.py{' ' * (COLS - 53)}║")
    print("╚" + "═" * (COLS - 2) + "╝")
    print()
