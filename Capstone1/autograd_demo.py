#!/usr/bin/env python3
"""
autograd_demo.py — How PyTorch computes gradients automatically.

Run:  python autograd_demo.py
      python autograd_demo.py 2>&1 | less

Lessons
  1. Why do we need gradients?
  2. Your first gradient (scalar case)
  3. The computation graph
  4. Multiple inputs — partial derivatives
  5. Non-scalar outputs — Jacobian-vector product
  6. torch.no_grad — disabling the graph for inference
  7. torch.autograd.grad — targeted gradient computation
  8. Higher-order derivatives (grad of a grad)
  9. Gradient accumulation — the silent bug
  10. Leaf vs intermediate tensors
"""

import torch

COLS = 65


def lesson(num, title):
    print(f"\n{'═' * COLS}")
    print(f"  LESSON {num}: {title}")
    print(f"{'═' * COLS}")


def explain(*lines):
    print()
    for line in lines:
        print(f"  {line}")
    print()


def show(label, value, width=40):
    print(f"  >>> {label:<{width}} {value}")


def code(text):
    print(f"  [ {text} ]")


def divider():
    print(f"  {'─' * (COLS - 2)}")


# ── Introduction ──────────────────────────────────────────────────────────────

print()
print("╔" + "═" * (COLS - 2) + "╗")
print("║" + "  PyTorch Autograd — Automatic Differentiation".center(COLS - 2) + "║")
print("╚" + "═" * (COLS - 2) + "╝")

explain(
    "TRAINING a neural network requires computing gradients:",
    "  'how much does the loss change if I nudge this weight?'",
    "",
    "Doing this by hand for millions of weights is impossible.",
    "PyTorch's AUTOGRAD engine does it automatically.",
    "",
    "The key idea: CHAIN RULE from calculus.",
    "  If  z = f(y)  and  y = g(x),  then:",
    "    dz/dx  =  dz/dy  ×  dy/dx",
    "",
    "  PyTorch builds a graph of operations as your forward pass runs,",
    "  then walks the graph BACKWARDS applying the chain rule at each step.",
    "  This is called BACKPROPAGATION.",
)


# ══════════════════════════════════════════════════════════════════════════════
lesson(1, "Why Gradients? — The Gradient Descent Update")
# ══════════════════════════════════════════════════════════════════════════════

explain(
    "Gradient descent is the optimisation algorithm behind all deep learning.",
    "The update rule is:",
    "",
    "    weight  =  weight  -  learning_rate × gradient",
    "",
    "    gradient = d(loss) / d(weight)   ← 'how much does loss move if I",
    "                                           nudge this weight by ε?'",
    "",
    "Example: suppose loss = weight² and we want to minimise loss.",
    "  d(loss)/d(weight) = 2 × weight",
    "  If weight=3: gradient = 6.  Update: weight = 3 - lr×6",
    "  This moves weight TOWARD zero, where loss is minimised.  ✓",
    "",
    "In a real network: loss depends on millions of weights via",
    "hundreds of layers. Autograd handles ALL of this automatically.",
)


# ══════════════════════════════════════════════════════════════════════════════
lesson(2, "Your First Gradient — Scalar Case")
# ══════════════════════════════════════════════════════════════════════════════

explain(
    "Step 1: Create a tensor with requires_grad=True.",
    "  This tells PyTorch: 'track every operation on this tensor'.",
    "",
    "Step 2: Run a forward pass — build up some expression.",
    "",
    "Step 3: Call .backward() on the final scalar output.",
    "  PyTorch walks the graph backwards and fills .grad on each leaf.",
)

code("x = torch.tensor(3.0, requires_grad=True)")
x = torch.tensor(3.0, requires_grad=True)
show("x.requires_grad", x.requires_grad)
show("x.grad           (before backward)", x.grad)
explain("x.grad is None because we haven't called backward() yet.")

code("y = x ** 2   # y = x²")
y = x ** 2
show("y.item()", y.item())
show("y.grad_fn", y.grad_fn)
explain(
    "y has a grad_fn (PowBackward0) — the backward function for '**'.",
    "This is the node in our computation graph.",
    "y itself does NOT have requires_grad=True set by the user,",
    "but it inherits it because x does.",
)

code("y.backward()   # compute dy/dx and store in x.grad")
y.backward()
show("x.grad", x.grad.item())
explain(
    f"x.grad = {x.grad.item():.1f}  (expected: d(x²)/dx = 2x = 2×3 = 6)  ✓",
    "",
    "After backward(), the computation graph is FREED by default",
    "(to save memory). If you need to call backward() again on the same",
    "graph, pass retain_graph=True.",
)


# ══════════════════════════════════════════════════════════════════════════════
lesson(3, "The Computation Graph")
# ══════════════════════════════════════════════════════════════════════════════

explain(
    "Every operation PyTorch performs builds a node in a Directed Acyclic Graph.",
    "The graph records WHAT function was applied and to WHICH inputs.",
    "backward() traverses this graph from output back to inputs.",
    "",
    "Example:  z = x*y + y²",
    "",
    "  x ──────┐",
    "           ├─ MulBackward0 ──┐",
    "  y ──────┘                  ├─ AddBackward0 ──► z",
    "  y ── PowBackward0(n=2) ──┘",
    "",
    "  backward() computes:",
    "    dz/dx = y           (partial of x*y w.r.t. x)",
    "    dz/dy = x + 2y      (partial of x*y+y² w.r.t. y)",
)

x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

code("x = tensor(2.0, requires_grad=True)")
code("y = tensor(3.0, requires_grad=True)")
code("z = x * y + y ** 2")
z = x * y + y ** 2

show("z.item()          ", z.item())
show("z.grad_fn         ", z.grad_fn)
show("z.grad_fn.next_fns", z.grad_fn.next_functions)

divider()
code("z.backward()   # walk the graph, fill x.grad and y.grad")
z.backward()
show("x.grad", x.grad.item())
show("y.grad", y.grad.item())
explain(
    f"x.grad = {x.grad.item():.1f}  →  dz/dx = y = 3           ✓",
    f"y.grad = {y.grad.item():.1f}  →  dz/dy = x + 2y = 2 + 6 = 8  ✓",
    "",
    "Chain rule at work:",
    "  dz/dx = d(x·y)/dx + d(y²)/dx  =  y  + 0  =  3",
    "  dz/dy = d(x·y)/dy + d(y²)/dy  =  x  + 2y =  2 + 6 = 8",
)


# ══════════════════════════════════════════════════════════════════════════════
lesson(4, "Gradients Through a Mini Neural Network")
# ══════════════════════════════════════════════════════════════════════════════

explain(
    "Let's trace gradients through a single linear layer + MSE loss.",
    "This is exactly what happens in the first layer of any neural network.",
    "",
    "  input x (given) → Linear: y = W·x + b → loss = MSE(y, target)",
    "",
    "  We want: d(loss)/dW and d(loss)/db",
    "  These tell us how to update W and b to reduce the loss.",
)

torch.manual_seed(0)
x = torch.tensor([1.0, 2.0, 3.0])          # input (3 features)
W = torch.randn(2, 3, requires_grad=True)   # weight matrix (2 outputs, 3 inputs)
b = torch.randn(2,    requires_grad=True)   # bias (2 outputs)
t = torch.tensor([1.0, 0.0])               # target

code("y = W @ x + b      # forward: linear layer")
y = W @ x + b
code("loss = ((y - t)**2).mean()  # MSE loss")
loss = ((y - t) ** 2).mean()

show("y (predictions)", [f"{v:.3f}" for v in y.tolist()])
show("loss           ", f"{loss.item():.4f}")

code("loss.backward()   # backprop through the whole graph")
loss.backward()

show("W.grad (shape)", W.grad.shape)
show("W.grad        ", W.grad)
show("b.grad        ", b.grad)
explain(
    "These gradients tell us the 'slope' of the loss landscape w.r.t. each weight.",
    "",
    "Gradient descent update (learning_rate = 0.1):",
    "  W = W - 0.1 * W.grad",
    "  b = b - 0.1 * b.grad",
    "",
    "In practice, an optimiser (Adam, SGD) does this update for you.",
    "You just call: optimizer.zero_grad() → loss.backward() → optimizer.step()",
)


# ══════════════════════════════════════════════════════════════════════════════
lesson(5, "Non-Scalar Outputs — Jacobian-Vector Product")
# ══════════════════════════════════════════════════════════════════════════════

explain(
    "backward() requires the starting point to be a SCALAR (single number).",
    "If your output is a vector/matrix, you need to either:",
    "  a) Reduce it to a scalar (e.g. .sum() or .mean()) — common in training",
    "  b) Pass a gradient tensor to backward() — the Jacobian-vector product",
    "",
    "Option (b) computes:  v^T · J   where J is the Jacobian dy/dx.",
    "",
    "When v = ones, this gives the gradient of sum(y) w.r.t. x,",
    "which is the most common case.",
)

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2   # element-wise square → y is a vector

code("x = [1, 2, 3],  y = x**2  →  y = [1, 4, 9]")
code("y.backward(torch.ones_like(y))")
y.backward(torch.ones_like(y))
show("x.grad", x.grad)
explain(
    "x.grad = [2, 4, 6]  =  2*[1, 2, 3]",
    "",
    "This is equivalent to:  d(y[0]+y[1]+y[2]) / dx  =  d(sum(x²)) / dx  =  2x",
    "",
    "In training, the loss IS a scalar (cross-entropy, MSE, etc.) so you",
    "almost never need to pass a gradient to backward() manually.",
)


# ══════════════════════════════════════════════════════════════════════════════
lesson(6, "torch.no_grad — Fast Inference, No Graph Built")
# ══════════════════════════════════════════════════════════════════════════════

explain(
    "Building the computation graph has TWO costs:",
    "  1. Extra memory to store intermediate values for backward()",
    "  2. CPU overhead for every operation that records a grad_fn",
    "",
    "During INFERENCE you don't need gradients.",
    "torch.no_grad() disables graph construction entirely:",
    "  • ~30–50% faster forward pass",
    "  • No memory held for backward",
    "  • Safe to run without worrying about graph accumulation",
)

x = torch.randn(1000, 1000, requires_grad=True)

code("with torch.no_grad():")
code("    y = x @ x.T   # no grad_fn, no graph")
with torch.no_grad():
    y = x @ x.T

show("y.requires_grad", y.requires_grad)
show("y.grad_fn      ", y.grad_fn)
explain(
    "y has no grad_fn — the graph was never built.",
    "",
    "Even stronger: torch.inference_mode()",
    "  • Even stricter than no_grad — disallows in-place ops that could",
    "    corrupt gradients, and may unlock further kernel optimisations.",
    "  • Recommended for model evaluation and deployment.",
)

code("with torch.inference_mode():")
code("    z = x @ x.T")
with torch.inference_mode():
    z = x @ x.T
show("z.is_inference()", z.is_inference())

divider()
explain(
    "Typical training loop pattern:",
    "",
    "  for batch in dataloader:",
    "      optimizer.zero_grad()           # clear old grads",
    "      out  = model(batch)             # forward (graph builds here)",
    "      loss = criterion(out, labels)   # scalar",
    "      loss.backward()                 # build grads",
    "      optimizer.step()               # update weights",
    "",
    "  with torch.no_grad():              # eval — no graph needed",
    "      val_out = model(val_batch)",
)


# ══════════════════════════════════════════════════════════════════════════════
lesson(7, "torch.autograd.grad — Targeted Gradient Computation")
# ══════════════════════════════════════════════════════════════════════════════

explain(
    "loss.backward() scatters gradients across ALL leaf tensors.",
    "Sometimes you only want the gradient of ONE output w.r.t. ONE input.",
    "",
    "torch.autograd.grad(outputs, inputs) does exactly that:",
    "  • Does NOT write to .grad  (doesn't touch the leaf's grad buffer)",
    "  • Returns a NEW gradient tensor",
    "  • Can target specific input/output pairs",
    "  • Can be composed (higher-order derivatives)",
    "",
    "Used in: gradient penalty (WGAN-GP), meta-learning, second-order optimisers.",
)

x = torch.tensor(4.0, requires_grad=True)
y = x ** 3   # y = x³

code("x = 4.0,  y = x³")
code("(dy_dx,) = torch.autograd.grad(y, x, create_graph=True)")
(dy_dx,) = torch.autograd.grad(y, x, create_graph=True)
show("dy/dx at x=4", f"{dy_dx.item():.1f}  (expected 3x² = 3×16 = 48)  ✓")

explain(
    "create_graph=True keeps the graph for dy_dx alive so we can",
    "differentiate THROUGH it to get second derivatives.",
)

code("(d2y_dx2,) = torch.autograd.grad(dy_dx, x)")
(d2y_dx2,) = torch.autograd.grad(dy_dx, x)
show("d²y/dx² at x=4", f"{d2y_dx2.item():.1f}  (expected 6x = 6×4 = 24)  ✓")
explain(
    "Second derivatives are used in:",
    "  • Hessian-vector products  (second-order optimisers like K-FAC)",
    "  • Gradient penalty terms   (Lipschitz regularisation)",
    "  • Physics-informed NNs     (d²u/dx² appears in PDE residuals)",
)


# ══════════════════════════════════════════════════════════════════════════════
lesson(8, "Gradient Accumulation — A Common Bug")
# ══════════════════════════════════════════════════════════════════════════════

explain(
    "PyTorch ADDS to .grad on every backward() call — it never resets it.",
    "This is a deliberate design choice: it enables gradient accumulation",
    "across micro-batches (useful when GPU memory is too small for a full batch).",
    "",
    "But it's also the #1 beginner bug:",
    "  Forget to call optimizer.zero_grad() → gradients pile up → wrong updates.",
)

x = torch.tensor(2.0, requires_grad=True)
for step in range(3):
    y = x ** 2
    y.backward()
    print(f"  Step {step}: x.grad = {x.grad.item():.1f}   ← {['first run (correct)', 'DOUBLED! (bug)', 'TRIPLED! (bug)'][step]}")

explain(
    "The gradient at x=2 is 2x = 4. But on step 1 it's 8, step 2 it's 12.",
    "That's because .grad was never cleared.",
    "",
    "FIX: zero the gradient before each backward():",
    "  optimizer.zero_grad()   ← in a training loop",
    "  x.grad.zero_()          ← if manually managing gradients",
    "",
    "INTENTIONAL use: accumulate gradients over N micro-batches,",
    "then call optimizer.step(). Equivalent to a batch N times larger.",
)


# ══════════════════════════════════════════════════════════════════════════════
lesson(9, "Leaf vs Intermediate Tensors")
# ══════════════════════════════════════════════════════════════════════════════

explain(
    "LEAF tensor: created directly by user code (e.g. torch.randn with",
    "             requires_grad=True). Model weights are leaf tensors.",
    "             Gradients ACCUMULATE in .grad on leaf tensors.",
    "",
    "INTERMEDIATE tensor: output of an operation on leaf tensors.",
    "                     Its gradients are computed during backward but",
    "                     DISCARDED afterwards to save memory.",
    "                     .grad will be None.",
    "",
    "  x (leaf) ── MulBackward ──► y (intermediate) ── SumBackward ──► z (scalar)",
    "                                                                        │",
    "  x.grad  ◄── backward traversal ─────────────────────────────────────┘",
    "  y.grad = None  (freed after use)",
)

x = torch.randn(3, requires_grad=True)   # leaf
y = x * 2                                # intermediate
z = y.sum()
z.backward()

code("x = randn(3, requires_grad=True)   # leaf")
code("y = x * 2                          # intermediate")
code("z = y.sum()                        # scalar")
code("z.backward()")
show("x.is_leaf", x.is_leaf)
show("x.grad   ", x.grad)
show("y.is_leaf", y.is_leaf)
show("y.grad   ", y.grad)

explain(
    f"x.grad = {x.grad.tolist()}  — filled because x is a leaf  ✓",
    "y.grad = None               — intermediate, grad was freed",
    "",
    "If you need an intermediate gradient (for debugging):",
    "  y.retain_grad()   ← call this BEFORE backward()",
    "  Then y.grad will be populated after backward().",
)

x2 = torch.randn(3, requires_grad=True)
y2 = x2 * 2
y2.retain_grad()   # ask PyTorch to keep y2's grad
y2.sum().backward()
code("y2.retain_grad()   ← before backward()")
show("y2.grad after retain_grad()", y2.grad)
explain("y2.grad = [2,2,2] because d(sum(x*2))/d(y) = [1,1,1] and y=2x so dy→dx chain.")

print()
print("╔" + "═" * (COLS - 2) + "╗")
print("║" + "  Autograd Demo Complete!".center(COLS - 2) + "║")
print("╠" + "═" * (COLS - 2) + "╣")
print("║" + "  Key takeaways:".ljust(COLS - 2) + "║")
print("║" + "    • requires_grad=True  → track ops on this tensor".ljust(COLS - 2) + "║")
print("║" + "    • .backward()         → fill .grad on all leaves".ljust(COLS - 2) + "║")
print("║" + "    • no_grad / inf_mode  → skip graph (inference)".ljust(COLS - 2) + "║")
print("║" + "    • zero_grad()         → ALWAYS before each backward".ljust(COLS - 2) + "║")
print("╚" + "═" * (COLS - 2) + "╝")
print()
