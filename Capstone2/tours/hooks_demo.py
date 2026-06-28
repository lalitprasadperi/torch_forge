"""
hooks_demo.py — Forward and Backward Hooks

Run with:
  python tours/hooks_demo.py

What you'll learn:
  Lesson 1 — What are hooks?
  Lesson 2 — Forward hooks: capture activations
  Lesson 3 — Backward hooks: capture gradients
  Lesson 4 — Visualise dead ReLUs with hooks
  Lesson 5 — Gradient flow — which layers learn?
  Lesson 6 — Remove hooks and avoid memory leaks
"""

import sys
import torch
import torch.nn as nn
import math

COLS = 68

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
    print(f"  {'▶ ' + label:<32} {value}")

def bar(value, max_val, width=30):
    """ASCII progress bar."""
    filled = int(width * value / max_val) if max_val > 0 else 0
    return "█" * filled + "░" * (width - filled)


# ── Shared model ──────────────────────────────────────────────────────────────

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1  = nn.Linear(16, 32)
        self.relu = nn.ReLU()
        self.fc2  = nn.Linear(32, 16)
        self.relu2 = nn.ReLU()
        self.fc3  = nn.Linear(16, 2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)


# ── Lessons ───────────────────────────────────────────────────────────────────

def lesson_1_what_are_hooks():
    lesson(1, "What Are Hooks?")
    explain(
        "A hook is a callback function you attach to a module. PyTorch calls",
        "it automatically at specific points in the forward/backward pass.",
        "",
        "  Forward hook   : called AFTER module.forward() returns",
        "                   receives (module, input, output)",
        "",
        "  Backward hook  : called during loss.backward() when gradients",
        "                   flow through this module",
        "                   receives (module, grad_input, grad_output)",
        "",
        "Why are hooks useful?",
        "  • Capture intermediate activations without modifying the model",
        "  • Debug dead ReLUs, exploding/vanishing gradients",
        "  • Build feature extractors (e.g. linear probing on frozen backbone)",
        "  • Gradient penalty, spectral normalisation",
        "",
        "Registration:",
        "  handle = layer.register_forward_hook(fn)",
        "  handle = layer.register_full_backward_hook(fn)",
        "",
        "Remove when done to avoid memory leaks:",
        "  handle.remove()",
    )


def lesson_2_forward_hooks():
    lesson(2, "Forward Hooks: Capture Activations")
    explain(
        "We'll attach a forward hook to each layer and print the activation",
        "shape and statistics as a tensor flows through the network.",
    )

    model = FeedForward()

    captured = {}

    def make_hook(name):
        def hook(module, inp, out):
            captured[name] = {
                "shape": tuple(out.shape),
                "mean":  out.detach().mean().item(),
                "std":   out.detach().std().item(),
                "frac_zero": (out.detach() == 0).float().mean().item(),
            }
        return hook

    handles = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.ReLU)):
            handles.append(module.register_forward_hook(make_hook(name)))

    x = torch.randn(8, 16)   # batch of 8, input dim 16
    with torch.no_grad():
        _ = model(x)

    explain("  Activation statistics after forward pass:")
    print(f"  {'Layer':<12} {'Shape':<14} {'Mean':>8} {'Std':>8} {'Frac=0':>8}")
    print("  " + "─" * 56)
    for name in ["fc1", "relu", "fc2", "relu2", "fc3"]:
        if name in captured:
            c = captured[name]
            print(f"  {name:<12} {str(c['shape']):<14} "
                  f"{c['mean']:>8.3f} {c['std']:>8.3f} "
                  f"{c['frac_zero']:>7.1%}")

    for h in handles:
        h.remove()

    explain("", "  Notice: ReLU layers have frac_zero ≈ 50% (half neurons zero).",
            "  This is normal — it's what makes ReLU non-linear.",
            "  If frac_zero approaches 100%, those neurons are 'dead'.")


def lesson_3_backward_hooks():
    lesson(3, "Backward Hooks: Capture Gradients")
    explain(
        "Backward hooks fire during loss.backward(). They receive the gradient",
        "that is FLOWING INTO the layer (grad_output — from upstream / loss side).",
        "",
        "This lets you inspect how strong the gradient signal is at each layer.",
        "Vanishing gradient: grad norm → 0 in early layers → those layers stop learning.",
    )

    model = FeedForward()
    criterion = nn.CrossEntropyLoss()

    grad_norms = {}

    def make_grad_hook(name):
        def hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                grad_norms[name] = grad_output[0].norm().item()
        return hook

    handles = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            handles.append(module.register_full_backward_hook(make_grad_hook(name)))

    x      = torch.randn(8, 16)
    labels = torch.randint(0, 2, (8,))
    loss   = criterion(model(x), labels)
    loss.backward()

    max_norm = max(grad_norms.values()) if grad_norms else 1.0
    explain("  Gradient norms at each Linear layer (higher = stronger signal):")
    for name in ["fc1", "fc2", "fc3"]:
        if name in grad_norms:
            n = grad_norms[name]
            print(f"  {name:<8}  norm={n:>8.4f}  {bar(n, max_norm, 30)}")

    for h in handles:
        h.remove()

    explain("",
            "  fc3 (closest to loss) typically has the largest gradient.",
            "  fc1 (furthest from loss) has the smallest.",
            "  This is the vanishing gradient problem in action.",
            "  ResNet skip connections fix this by providing a shorter gradient path.")


def lesson_4_dead_relus():
    lesson(4, "Detecting Dead ReLUs")
    explain(
        "A ReLU neuron is 'dead' if it outputs 0 for ALL inputs in your dataset.",
        "This happens when the pre-activation (input to ReLU) is always negative.",
        "Dead neurons never fire, never produce a gradient, and never recover.",
        "",
        "Causes:",
        "  • Too-high learning rate → weights driven very negative in one step",
        "  • Poor weight initialisation",
        "  • Exploding gradients",
        "",
        "We'll deliberately create dead neurons and detect them with hooks.",
    )

    class BadInit(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1  = nn.Linear(8, 16)
            self.relu = nn.ReLU()
            self.fc2  = nn.Linear(16, 4)

        def forward(self, x):
            return self.fc2(self.relu(self.fc1(x)))

    model = BadInit()
    # Artificially make all fc1 biases very negative → all neurons dead
    with torch.no_grad():
        model.fc1.bias.fill_(-100.0)

    dead_counts = {}

    def relu_hook(module, inp, out):
        # out shape: (B, neurons)
        # A neuron is dead if it's 0 for ALL samples in this batch
        always_zero = (out.detach() == 0).all(dim=0)  # (neurons,) bool
        dead_counts["relu"] = always_zero.sum().item()
        dead_counts["total"] = out.shape[1]

    handle = model.relu.register_forward_hook(relu_hook)

    x = torch.randn(64, 8)   # 64 samples
    with torch.no_grad():
        _ = model(x)

    handle.remove()

    dead  = dead_counts.get("relu", 0)
    total = dead_counts.get("total", 1)
    show("Dead neurons  :", f"{dead} / {total}  ({dead/total:.0%})")
    show("Bias values   :", "all set to -100 → all pre-activations negative")

    explain("",
            "  100% dead neurons: the ReLU layer outputs all zeros.",
            "  Fix: use proper initialisation (Kaiming He) and moderate learning rates.",
            "  Alternative: use GELU or SiLU which are smooth and never fully zero.")


def lesson_5_gradient_flow():
    lesson(5, "Gradient Flow Visualisation")
    explain(
        "Let's train a small network for 10 steps and track how gradient norms",
        "evolve for each layer. This is exactly the kind of diagnostic you'd run",
        "when a model isn't learning — before blaming the data or architecture.",
    )

    torch.manual_seed(0)
    model = FeedForward()
    opt   = torch.optim.SGD(model.parameters(), lr=0.01)
    crit  = nn.CrossEntropyLoss()

    history = {"fc1": [], "fc2": [], "fc3": []}

    def make_hook(name):
        def hook(module, gi, go):
            if go[0] is not None:
                history[name].append(go[0].norm().item())
        return hook

    handles = [
        model.fc1.register_full_backward_hook(make_hook("fc1")),
        model.fc2.register_full_backward_hook(make_hook("fc2")),
        model.fc3.register_full_backward_hook(make_hook("fc3")),
    ]

    for step in range(10):
        x      = torch.randn(32, 16)
        labels = torch.randint(0, 2, (32,))
        loss   = crit(model(x), labels)
        opt.zero_grad()
        loss.backward()
        opt.step()

    for h in handles:
        h.remove()

    explain("  Gradient norm over 10 training steps:")
    print(f"  {'Step':<6}", end="")
    for name in ["fc1", "fc2", "fc3"]:
        print(f"  {name:>10}", end="")
    print()
    print("  " + "─" * 42)
    for i in range(len(history["fc3"])):
        print(f"  {i+1:<6}", end="")
        for name in ["fc1", "fc2", "fc3"]:
            if i < len(history[name]):
                print(f"  {history[name][i]:>10.5f}", end="")
        print()


def lesson_6_remove_hooks():
    lesson(6, "Removing Hooks — Avoid Memory Leaks")
    explain(
        "Every registered hook holds a reference to your closure (and thus to",
        "any tensors captured in it). If you forget to call handle.remove(),",
        "those tensors are never garbage-collected → memory leak.",
        "",
        "Best practice — use a context manager pattern:",
    )

    print("    handles = []")
    print("    try:")
    print("        for name, m in model.named_modules():")
    print("            handles.append(m.register_forward_hook(my_hook))")
    print("        out = model(x)   # hooks fire here")
    print("    finally:")
    print("        for h in handles:")
    print("            h.remove()   # always clean up")
    print()

    explain("  Alternatively, use remove_hooks() from our BaseModel class.")

    model = FeedForward()
    captured = []
    handle = model.fc1.register_forward_hook(lambda m, i, o: captured.append(o.detach()))

    with torch.no_grad():
        model(torch.randn(4, 16))

    show("Captured before remove  :", f"{len(captured)} activation tensors")
    handle.remove()

    with torch.no_grad():
        model(torch.randn(4, 16))

    show("Captured after remove   :", f"{len(captured)} (no new captures)")

    explain("", "  After remove(), the hook no longer fires.")


# ── Run all lessons ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("╔" + "═" * (COLS - 2) + "╗")
    print(f"║  Forward & Backward Hooks — A Deep Dive{' ' * (COLS - 42)}║")
    print(f"║  6 lessons on debugging and inspecting neural networks{' ' * (COLS - 56)}║")
    print("╚" + "═" * (COLS - 2) + "╝")

    lesson_1_what_are_hooks()
    lesson_2_forward_hooks()
    lesson_3_backward_hooks()
    lesson_4_dead_relus()
    lesson_5_gradient_flow()
    lesson_6_remove_hooks()

    print()
    print("╔" + "═" * (COLS - 2) + "╗")
    print(f"║  ALL LESSONS COMPLETE {'─' * 10} Next: tours/amp_demo.py{' ' * (COLS - 52)}║")
    print("╚" + "═" * (COLS - 2) + "╝")
    print()
