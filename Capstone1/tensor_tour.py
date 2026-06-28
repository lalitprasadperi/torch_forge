#!/usr/bin/env python3
"""
tensor_tour.py — Step-by-step walkthrough of PyTorch tensors.

Run:  python tensor_tour.py
      python tensor_tour.py 2>&1 | less   (paginated)

Lessons
  1. Creating tensors
  2. Reading tensor metadata (shape, dtype, stride)
  3. The memory model: storage, strides, and views
  4. Views vs copies
  5. Tensor operations
  6. Broadcasting
  7. CUDA: moving tensors to the GPU
"""

import torch

W = 65  # column width for formatting


# ── Helpers ───────────────────────────────────────────────────────────────────

def lesson(num, title):
    print(f"\n{'═' * W}")
    print(f"  LESSON {num}: {title}")
    print(f"{'═' * W}")


def explain(*lines):
    """Print narrative text with a left margin."""
    print()
    for line in lines:
        print(f"  {line}")
    print()


def show(label, value, width=38):
    label_str = f"  >>> {label}"
    print(f"{label_str:<{width+5}}  →  {value}")


def code(text):
    """Print a line of code the way a REPL would show it."""
    print(f"  [ {text} ]")


def divider():
    print(f"  {'─' * (W - 2)}")


# ── Introduction ──────────────────────────────────────────────────────────────

print()
print("╔" + "═" * (W - 2) + "╗")
print("║" + "  PyTorch Tensor Tour — A Beginner's Walkthrough".center(W - 2) + "║")
print("╚" + "═" * (W - 2) + "╝")
explain(
    "This script runs code, prints the results, and explains what",
    "each result means. Read the output top-to-bottom as a tutorial.",
    "",
    "A TENSOR is PyTorch's fundamental data structure.",
    "You can think of it as a multi-dimensional array that:",
    "  • lives on CPU or GPU",
    "  • can track operations so gradients can flow backwards",
    "  • is the building block of every neural network layer",
    "",
    f"  1-D tensor  →  a vector     [1, 2, 3]",
    f"  2-D tensor  →  a matrix     [[1, 2], [3, 4]]",
    f"  3-D tensor  →  a cube       shape (batch, rows, cols)",
    f"  4-D tensor  →  images       shape (N, C, H, W)",
)


# ══════════════════════════════════════════════════════════════════════════════
lesson(1, "Creating Tensors")
# ══════════════════════════════════════════════════════════════════════════════

explain(
    "PyTorch gives you several factory functions to create tensors.",
    "Each is suited to a different situation.",
    "We'll create each one, print it, and explain when you'd use it.",
)

# torch.zeros / torch.ones
code("torch.zeros(3, 4)")
t = torch.zeros(3, 4)
show("value", t)
show("shape", t.shape)
explain(
    "torch.zeros(3, 4) creates a 3-row, 4-column tensor filled with 0.0.",
    "Use this to initialise a bias vector or a mask placeholder.",
)

code("torch.ones(2, 3, dtype=torch.float32)")
t = torch.ones(2, 3, dtype=torch.float32)
show("value", t)
show("dtype", t.dtype)
explain(
    "torch.ones creates 1.0s of a specific dtype.",
    "dtype=torch.float32 means each number uses 4 bytes (32 bits).",
    "For GPU inference, float16 (2 bytes) is preferred — half the memory.",
)

# torch.randn
code("torch.randn(2, 3)  # samples from N(0,1)")
t = torch.randn(2, 3)
show("value", t)
explain(
    "torch.randn samples from a standard normal distribution N(0, 1).",
    "This is what you use to randomly initialise model weights.",
    "Each call gives different numbers — tensors are NOT deterministic by default.",
    "",
    "To get reproducible results, set the seed first:",
    "  torch.manual_seed(42)",
    "  torch.randn(2, 3)   ← same numbers every time",
)

# torch.arange
code("torch.arange(0, 10, step=2)")
t = torch.arange(0, 10, step=2, dtype=torch.float32)
show("value", t)
explain(
    "torch.arange is like Python's range() but returns a tensor.",
    "start=0, stop=10, step=2 → [0, 2, 4, 6, 8]  (stop is EXCLUDED)",
    "Great for creating position indices or evenly-spaced grids.",
)

# torch.tensor (from Python data)
code("torch.tensor([[1.0, 2.0], [3.0, 4.0]])")
t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
show("value", t)
show("shape", t.shape)
explain(
    "torch.tensor wraps existing Python lists or numpy arrays.",
    "Use this when you have hand-crafted values or small test cases.",
    "The shape is inferred from the nesting: 2 rows × 2 cols → (2, 2).",
)

# zeros_like / ones_like
x = torch.randn(3, 4)
code("torch.zeros_like(x)  # same shape and dtype as x")
show("shape", torch.zeros_like(x).shape)
show("dtype", torch.zeros_like(x).dtype)
explain(
    "zeros_like / ones_like / empty_like create new tensors that",
    "match the shape and dtype of an existing tensor.",
    "empty_like is fastest — it allocates memory but doesn't initialise it.",
    "Use it when you'll immediately overwrite every element.",
)


# ══════════════════════════════════════════════════════════════════════════════
lesson(2, "Reading Tensor Metadata")
# ══════════════════════════════════════════════════════════════════════════════

explain(
    "Every tensor carries metadata that describes how its data is stored.",
    "You'll read these constantly while debugging and profiling.",
)

x = torch.randn(4, 6)
print("  Creating x = torch.randn(4, 6)\n")

show("x.shape   ", x.shape)
explain(
    "shape is the logical size: 4 rows, 6 columns.",
    "x.shape[0] = 4  (rows),  x.shape[1] = 6  (cols)",
    "Equivalent: x.size()  →  same as x.shape",
)

show("x.dtype   ", x.dtype)
explain(
    "dtype is the data type of each element.",
    "  float32 → 4 bytes per element  (default for CPU tensors)",
    "  float16 → 2 bytes per element  (default for GPU inference)",
    "  bfloat16→ 2 bytes, wider exponent (common in LLM training)",
    "  int64   → 8 bytes              (token IDs, indices)",
)

show("x.device  ", x.device)
explain(
    "device tells you where the data lives.",
    "  'cpu'    → system RAM",
    "  'cuda:0' → GPU #0 VRAM",
    "Operations between tensors on DIFFERENT devices will raise an error.",
)

show("x.ndim    ", x.ndim)
show("x.numel() ", x.numel())
explain(
    "ndim is the number of dimensions (rank): 2 for a matrix.",
    "numel() is the total number of elements: 4 × 6 = 24.",
)

show("x.stride()", x.stride())
explain(
    "stride is the KEY concept for understanding tensor memory.",
    "",
    "  stride() = (6, 1)",
    "",
    "  This means:",
    "    • to move to the NEXT ROW,   jump 6 elements in memory",
    "    • to move to the NEXT COLUMN, jump 1 element in memory",
    "",
    "  To find element x[i, j] in the flat buffer:",
    "    buffer_index = i * stride[0]  +  j * stride[1]",
    "                 = i * 6          +  j * 1",
    "",
    "  Example: x[2, 3] lives at buffer position  2*6 + 3*1 = 15",
    "",
    "  This is ROW-MAJOR order (same as C, NumPy).",
    "  COLUMN-MAJOR (Fortran order) would be stride = (1, 4).",
)

show("x.is_contiguous()", x.is_contiguous())
explain(
    "is_contiguous() = True means the elements are laid out in memory",
    "in the order you'd expect from iterating dimension-by-dimension.",
    "Non-contiguous tensors (e.g. after a transpose) have unusual strides.",
    "Some ops require contiguous input — call .contiguous() to force a copy.",
)


# ══════════════════════════════════════════════════════════════════════════════
lesson(3, "The Memory Model: Storage, Strides, and Views")
# ══════════════════════════════════════════════════════════════════════════════

explain(
    "CRITICAL CONCEPT: a tensor is NOT its data.",
    "",
    "A tensor is a VIEW — a set of metadata (shape, stride, offset)",
    "that describes how to interpret a flat 1-D buffer of numbers.",
    "",
    "Multiple tensors can share the same underlying buffer.",
    "This is how PyTorch avoids copying data unnecessarily.",
    "",
    "Memory layout of a (3, 4) tensor:",
    "",
    "  Logical (what you see)        Physical (what's in RAM)",
    "  ┌───┬───┬───┬───┐             ┌────────────────────────────────────────┐",
    "  │ 0 │ 1 │ 2 │ 3 │  row 0  →  │  0   1   2   3   4   5   6 ... 11     │",
    "  ├───┼───┼───┼───┤             └────────────────────────────────────────┘",
    "  │ 4 │ 5 │ 6 │ 7 │  row 1      stride=(4,1): next row = +4, next col = +1",
    "  ├───┼───┼───┼───┤",
    "  │ 8 │ 9 │10 │11 │  row 2",
    "  └───┴───┴───┴───┘",
)

a = torch.arange(12, dtype=torch.float32).reshape(3, 4)
code("a = torch.arange(12).reshape(3, 4)")
show("a", a)
show("a.stride()        ", a.stride())
divider()

explain(
    "Now transpose it. Transposing SWAPS the strides — no data is copied.",
    "",
    "  a.stride()  = (4, 1)  →  b = a.t()  →  b.stride() = (1, 4)",
    "",
    "  After transpose the physical layout is unchanged:",
    "  ┌────────────────────────────────────────┐",
    "  │  0   1   2   3   4   5   6 ... 11     │  ← same buffer!",
    "  └────────────────────────────────────────┘",
    "  But now b[i,j] = buffer[ i*1 + j*4 ]  (column-major indexing)",
)

b = a.t()
code("b = a.t()  # transpose: zero-cost, just swaps strides")
show("b", b)
show("b.stride()        ", b.stride())
show("b.is_contiguous() ", b.is_contiguous())
show("same buffer?      ", a.data_ptr() == b.data_ptr())
explain(
    "data_ptr() is the memory address of the first element.",
    "Both a and b return the SAME address → same buffer, different view.",
    "",
    "This is why in-place ops on a view affect the original tensor!",
    "  b[0, 0] = 99  →  a[0, 0] is also 99",
)

divider()
explain(
    ".contiguous() materialises the view into a NEW buffer with",
    "normal row-major strides. This IS a memory copy.",
)
c = b.contiguous()
code("c = b.contiguous()  # forces a copy into row-major layout")
show("c.stride()         ", c.stride())
show("c.is_contiguous()  ", c.is_contiguous())
show("c shares buffer?   ", c.data_ptr() == b.data_ptr())

divider()
explain("Slices are also views — same buffer, different offset+stride.")
code("a[1]    # second row — contiguous slice")
show("a[1].stride()      ", a[1].stride())
show("a[1].is_contiguous()", a[1].is_contiguous())
code("a[:, 2] # third column — NON-contiguous")
show("a[:,2].stride()    ", a[:, 2].stride())
show("a[:,2].is_contiguous()", a[:, 2].is_contiguous())
explain(
    "a[:, 2] has stride (4,) because each element of the column",
    "is 4 positions apart in the flat buffer (skipping the other columns).",
    "",
    "Many CUDA kernels require contiguous input. If you see:",
    "  RuntimeError: input must be contiguous",
    "just call .contiguous() before passing to the op.",
)


# ══════════════════════════════════════════════════════════════════════════════
lesson(4, "Reshaping: view() vs reshape() vs contiguous()")
# ══════════════════════════════════════════════════════════════════════════════

explain(
    ".view()     — reinterpret the buffer with a new shape.",
    "              REQUIRES contiguous input. No data copy. Fast.",
    ".reshape()  — same as view() if contiguous, otherwise copies first.",
    "              Always succeeds. Use this unless you need the copy guarantee.",
    ".contiguous() — copy to row-major layout. Needed before .view() on",
    "              non-contiguous tensors.",
)

x = torch.arange(24, dtype=torch.float32)
code("x = torch.arange(24)   # flat 1-D, shape (24,)")
show("x.shape", x.shape)

code("x.view(4, 6)   # reinterpret as (4, 6) — no copy")
y = x.view(4, 6)
show("y.shape", y.shape)
show("y.stride()", y.stride())
show("same buffer?", x.data_ptr() == y.data_ptr())
explain(
    "view(4, 6) just changes the metadata: shape=(4,6), stride=(6,1).",
    "The underlying 24 numbers are unchanged. This costs essentially nothing.",
    "",
    "Rule: numel() must stay the same: 24 = 4 × 6  ✓",
)


# ══════════════════════════════════════════════════════════════════════════════
lesson(5, "Tensor Operations")
# ══════════════════════════════════════════════════════════════════════════════

explain(
    "PyTorch overloads Python operators (+, -, *, /, **) for tensors.",
    "These are element-wise unless the operator is @  (matrix multiply).",
    "",
    "Operations fall into three families:",
    "  POINTWISE  — output[i] = f(a[i])         e.g.  x + y, x.exp()",
    "  REDUCTION  — scalar = f(all elements)     e.g.  x.sum(), x.mean()",
    "  MATMUL     — output = matrix product      e.g.  x @ y, torch.mm(x, y)",
)

x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
y = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
print("  x =", x.tolist(), "   y =", y.tolist(), "\n")

code("x + y   (element-wise add)")
show("result", (x + y))

code("x * y   (Hadamard — element-wise multiply, NOT matmul)")
show("result", (x * y))

code("x @ y   (matrix multiply)")
show("result", (x @ y))
explain(
    "x @ y computes the standard matrix product.",
    "  out[i,j] = sum over k of  x[i,k] * y[k,j]",
    "",
    "For batched matmul:  use torch.bmm(a, b)  for 3-D inputs (B, M, K) @ (B, K, N)",
    "                  or  a @ b directly (PyTorch broadcasts the batch dim).",
)

divider()
explain("REDUCTIONS collapse one or more dimensions into a scalar or smaller tensor.")
code("x.sum()           # sum all elements")
show("x.sum()", x.sum().item())
code("x.sum(dim=0)      # sum along rows → one value per column")
show("x.sum(dim=0)", x.sum(dim=0))
code("x.mean(dim=1)     # mean along columns → one value per row")
show("x.mean(dim=1)", x.mean(dim=1))
code("x.max(dim=1)      # max value + its index along columns")
vals, idxs = x.max(dim=1)
show("x.max(dim=1).values", vals)
show("x.max(dim=1).indices", idxs)
explain(
    "dim=0 means 'collapse along rows' (result has one row).",
    "dim=1 means 'collapse along columns' (result has one column per row).",
    "keepdim=True preserves the collapsed dimension as size 1.",
    "",
    "Example:  x.sum(dim=0, keepdim=True).shape  →  (1, 2) instead of (2,)",
    "  Useful when you need shapes to broadcast correctly in later ops.",
)

divider()
explain("IN-PLACE operations have a trailing underscore (add_, mul_, etc.).")
z = x.clone()
code("z.add_(1.0)   # modifies z in place; no new tensor created")
show("z before", x)
z.add_(1.0)
show("z after ", z)
explain(
    "In-place ops save memory (no temporary tensor) but CANNOT be used",
    "on tensors that are part of an autograd computation graph.",
    "PyTorch will raise an error if you try — this protects gradient computation.",
    "In practice: avoid in-place ops during training, use them freely at inference.",
)


# ══════════════════════════════════════════════════════════════════════════════
lesson(6, "Broadcasting — Automatic Shape Expansion")
# ══════════════════════════════════════════════════════════════════════════════

explain(
    "Broadcasting lets you apply operations between tensors of DIFFERENT shapes",
    "without writing explicit loops or calling .expand()/.tile().",
    "",
    "The rules (applied right-to-left on the dimensions):",
    "  1. Prepend 1s to the shorter shape until both have the same ndim.",
    "  2. For each dimension pair:",
    "       — if they're equal: fine.",
    "       — if one is 1: it is 'stretched' to match the other.",
    "       — otherwise: ERROR.",
    "  3. No data is copied — stretching is virtual during computation.",
    "",
    "Example:  (3, 1) + (1, 4)  →  (3, 4)",
    "",
    "  shape (3, 1)   →   step 2: stretch dim-1 to 4   →   effective (3, 4)",
    "  shape (1, 4)   →   step 2: stretch dim-0 to 3   →   effective (3, 4)",
)

a = torch.tensor([[1.], [2.], [3.]])   # shape (3, 1)
b = torch.tensor([[10., 20., 30., 40.]])  # shape (1, 4)
code("a = [[1],[2],[3]]  shape (3,1)")
code("b = [[10,20,30,40]]  shape (1,4)")
code("a + b  →  broadcasts to (3, 4)")
result = a + b
show("result", result)
explain(
    "Row 0:  1 + [10,20,30,40] = [11, 21, 31, 41]",
    "Row 1:  2 + [10,20,30,40] = [12, 22, 32, 42]",
    "Row 2:  3 + [10,20,30,40] = [13, 23, 33, 43]",
)

divider()
explain(
    "REAL EXAMPLES you'll see in transformer code:",
    "",
    "1. Adding a bias to a batch of tokens:",
    "   tokens shape  (batch=8, hidden=512)",
    "   bias   shape  (hidden=512,)  →  prepend a 1  →  (1, 512)",
    "   output shape  (8, 512)  ✓",
)
tokens = torch.randn(8, 512)
bias   = torch.zeros(512)
code("tokens(8,512) + bias(512,)  →  (8,512)")
show("output shape", (tokens + bias).shape)

explain(
    "2. Adding a causal mask to attention scores:",
    "   scores shape  (batch=2, heads=8, T=64, T=64)",
    "   mask   shape  (1, 1, T=64, T=64)  →  broadcasts over batch and heads",
    "   output shape  (2, 8, 64, 64)  ✓",
)
scores = torch.zeros(2, 8, 64, 64)
mask   = torch.zeros(1, 1, 64, 64)
code("scores(2,8,64,64) + mask(1,1,64,64)  →  (2,8,64,64)")
show("output shape", (scores + mask).shape)

explain(
    "3. Shape mismatch — this raises an error:",
    "   (3, 4) + (3, 5) — neither dim-1 is 1, and 4 ≠ 5",
)
try:
    _ = torch.ones(3, 4) + torch.ones(3, 5)
except RuntimeError as e:
    show("RuntimeError", str(e)[:60] + "...")


# ══════════════════════════════════════════════════════════════════════════════
lesson(7, "CUDA — Moving Tensors to the GPU")
# ══════════════════════════════════════════════════════════════════════════════

explain(
    "GPU tensors live in VRAM (Video RAM) on the GPU card.",
    "All PyTorch tensor operations are the same — they just run faster on GPU.",
    "",
    "Rules:",
    "  • All tensors in an op must be on the SAME device.",
    "  • tensor.cuda() or tensor.to('cuda') copies to the default GPU.",
    "  • tensor.cpu() copies back to system RAM.",
    "  • torch.cuda.synchronize() waits for all GPU work to finish.",
    "    (GPU runs ASYNCHRONOUSLY — the CPU doesn't wait by default.)",
)

if not torch.cuda.is_available():
    explain("CUDA not available on this machine — skipping GPU lessons.")
else:
    explain("CUDA is available. Let's explore the GPU.")

    show("GPU name  ", torch.cuda.get_device_name(0))
    show("GPU count ", torch.cuda.device_count())
    show("VRAM total", f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB")

    divider()
    explain("Moving a tensor to GPU:")
    x_cpu = torch.randn(1000, 1000)
    show("x_cpu.device", x_cpu.device)

    x_gpu = x_cpu.cuda()
    code("x_gpu = x_cpu.cuda()  # copies data to GPU VRAM")
    show("x_gpu.device", x_gpu.device)
    show("x_cpu still exists?", x_cpu.device)
    explain(
        ".cuda() creates a NEW tensor in VRAM — the CPU copy still exists.",
        "To move without keeping a CPU copy: del x_cpu after .cuda()",
        "",
        "Preferred modern syntax:",
        "  x.to('cuda')         — device string",
        "  x.to(torch.device('cuda:0'))  — device object",
        "  x.to(device)        — where device = torch.device('cuda') from config",
    )

    divider()
    explain("Tracking GPU memory usage:")
    torch.cuda.reset_peak_memory_stats()
    y = torch.randn(2000, 2000, device='cuda', dtype=torch.float16)
    bytes_allocated = torch.cuda.memory_allocated()
    bytes_peak      = torch.cuda.max_memory_allocated()
    show("Allocated now  ", f"{bytes_allocated / 1e6:.2f} MB")
    show("Peak since reset", f"{bytes_peak / 1e6:.2f} MB")
    explain(
        "A (2000, 2000) float16 tensor uses:  2000 × 2000 × 2 bytes = 8 MB",
        f"  Measured: {bytes_allocated / 1e6:.2f} MB  ✓",
        "",
        "memory_allocated()  = bytes currently in use",
        "max_memory_allocated() = peak (including tensors already freed)",
        "memory_reserved()   = total bytes cached by PyTorch allocator",
        "                       (higher than allocated, due to caching)",
    )

    divider()
    explain(
        "Running an operation on GPU:",
        "",
        "The GPU executes kernels ASYNCHRONOUSLY.",
        "The CPU launches the kernel and immediately returns.",
        "torch.cuda.synchronize() makes the CPU wait for the GPU to finish.",
        "Without it, you may read stale results or get incorrect timings.",
    )
    a = torch.randn(2048, 2048, device='cuda', dtype=torch.float16)
    b = torch.randn(2048, 2048, device='cuda', dtype=torch.float16)
    code("c = a @ b  # launched on GPU — CPU does NOT wait")
    c = a @ b
    code("torch.cuda.synchronize()  # NOW CPU waits for GPU to finish")
    torch.cuda.synchronize()
    show("c.device", c.device)
    show("c.shape ", c.shape)
    show("c.dtype ", c.dtype)

    divider()
    explain("Moving results back to CPU:")
    c_cpu = c.cpu()
    show("c_cpu.device", c_cpu.device)
    explain(
        "After .cpu(), the tensor is back in system RAM.",
        "You can now convert to numpy: c_cpu.numpy()",
        "Note: you CANNOT call .numpy() on a GPU tensor directly.",
    )

print()
print("╔" + "═" * (W - 2) + "╗")
print("║" + "  Tensor Tour Complete!".center(W - 2) + "║")
print("╠" + "═" * (W - 2) + "╣")
print("║" + "  Next steps:".ljust(W - 2) + "║")
print("║" + "    python autograd_demo.py   — gradients and backprop".ljust(W - 2) + "║")
print("║" + "    python streams_demo.py    — GPU streams and timing".ljust(W - 2) + "║")
print("║" + "    python run_bench.py       — full performance lab".ljust(W - 2) + "║")
print("╚" + "═" * (W - 2) + "╝")
print()
