"""
amp_demo.py — Mixed Precision Training (AMP) Deep Dive

Run with:
  python tours/amp_demo.py

What you'll learn:
  Lesson 1 — Why float16? The case for half precision
  Lesson 2 — The dynamic range problem and GradScaler
  Lesson 3 — autocast — which ops run in FP16 vs FP32
  Lesson 4 — The full AMP training loop
  Lesson 5 — Gradient accumulation with AMP
  Lesson 6 — Measuring the speedup
"""

import sys
import time
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

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
    print(f"  {'▶ ' + label:<34} {value}")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Lessons ───────────────────────────────────────────────────────────────────

def lesson_1_why_fp16():
    lesson(1, "Why float16? The Case for Half Precision")
    explain(
        "float32 (FP32) — the default PyTorch dtype:",
        "  • 32 bits per number  (4 bytes)",
        "  • 1 sign + 8 exponent + 23 mantissa bits",
        "  • Range: ±3.4 × 10^38",
        "  • Precision: ~7 decimal digits",
        "",
        "float16 (FP16) — half precision:",
        "  • 16 bits per number  (2 bytes)",
        "  • 1 sign + 5 exponent + 10 mantissa bits",
        "  • Range: ±65504   ← this is the problem",
        "  • Precision: ~3 decimal digits",
        "",
        "Why bother with FP16 if it has less precision?",
        "  1. 2× less memory  → 2× larger batch size (or model)",
        "  2. GPU Tensor Cores compute FP16 matmuls 4-8× faster than FP32",
        "     On RTX 4090: ~165 TFLOPS FP32 vs ~1320 TFLOPS FP16",
        "  3. 2× less memory bandwidth → 2× faster memory-bound ops",
        "",
        "  For most neural network ops, FP16 precision is sufficient:",
        "  gradients are noisy anyway, and small numerical errors wash out.",
        "",
        "  The ONLY problematic ops are those with large dynamic range:",
        "  softmax, log, exp, LayerNorm. AMP keeps these in FP32 automatically.",
    )

    # Demonstrate the precision difference
    a_fp32 = torch.tensor(1e-8, dtype=torch.float32)
    a_fp16 = a_fp32.half()
    show("1e-8 in FP32 :", a_fp32.item())
    show("1e-8 in FP16 :", a_fp16.item())   # rounds to 0!

    b_fp32 = torch.tensor(65000.0, dtype=torch.float32)
    b_fp16 = b_fp32.half()
    c_fp16 = (b_fp16 * 2).item()            # overflow
    show("65000 × 2 in FP32 :", b_fp32.item() * 2)
    show("65000 × 2 in FP16 :", c_fp16, )   # inf


def lesson_2_grad_scaler():
    lesson(2, "The Dynamic Range Problem and GradScaler")
    explain(
        "Problem: gradients in early training can be very small (1e-5 to 1e-7).",
        "FP16 underflows at ~6e-8, so these gradients round to ZERO.",
        "Zero gradients → no parameter updates → model never learns.",
        "",
        "GradScaler solution:",
        "  1. SCALE:   multiply the loss by a large constant S (e.g. 2^16 = 65536)",
        "              This shifts all gradients up into the FP16 range.",
        "  2. BACKWARD: compute gradients in FP16 (they're all × S, so no underflow)",
        "  3. UNSCALE: divide all gradients by S to recover true gradient values",
        "  4. CHECK:   if any gradient is inf/nan (overflow occurred), SKIP the step",
        "  5. UPDATE:  adjust S up or down based on recent overflow history",
        "",
        "  S starts high (2^16), halves after overflow, doubles after 2000 clean steps.",
        "  In practice S stabilises quickly and overflows become rare.",
    )

    print("  ASCII diagram of loss scale mechanism:")
    print()
    print("   loss (FP32)           grad G (FP16)     true grad G/S")
    print("      │                        │                  │")
    print("      ▼                        ▼                  ▼")
    print("   × scale S  ──→  backward  ──→  unscale(÷S)  ──→  clip  ──→  step")
    print("   S = 65536             (in FP16)        (in FP32)")
    print()
    explain(
        "  Key: scaling happens BEFORE backward so the computation graph",
        "  scales all gradients uniformly. Unscaling restores true values.",
    )

    scaler = GradScaler(enabled=True)
    show("Initial scale :", scaler.get_scale())

    model  = nn.Linear(4, 4).to(DEVICE)
    opt    = torch.optim.SGD(model.parameters(), lr=0.01)
    x      = torch.randn(8, 4, device=DEVICE)

    with autocast(device_type=DEVICE, enabled=True):
        loss = model(x).sum()
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()

    show("Scale after 1 step:", scaler.get_scale())
    explain("  Scale is unchanged if no overflow occurred.")


def lesson_3_autocast_ops():
    lesson(3, "autocast — Which Ops Run in FP16 vs FP32")
    explain(
        "torch.amp.autocast is a context manager. Inside it, PyTorch automatically",
        "casts tensors to the most efficient dtype for each operation.",
        "",
        "  FP16 (fast): matmul, conv2d, linear — compute-intensive",
        "  FP32 (safe): softmax, log, exp, layernorm, loss functions — numerically sensitive",
        "",
        "  You don't need to manually cast anything — PyTorch does it for you.",
        "  The input tensors remain FP32; autocast inserts casts as needed.",
    )

    if DEVICE == "cpu":
        explain("  (Running on CPU — dtype will be bfloat16 instead of float16)")

    model = nn.Sequential(
        nn.Linear(8, 16),
        nn.ReLU(),
        nn.Linear(16, 4),
    ).to(DEVICE)

    x = torch.randn(4, 8, device=DEVICE)

    captured_dtypes = {}

    def dtype_hook(name):
        def hook(mod, inp, out):
            captured_dtypes[name] = {
                "input":  inp[0].dtype if isinstance(inp, tuple) else inp.dtype,
                "output": out.dtype,
            }
        return hook

    handles = [
        model[0].register_forward_hook(dtype_hook("fc1")),
        model[2].register_forward_hook(dtype_hook("fc2")),
    ]

    # WITHOUT autocast
    with torch.no_grad():
        _ = model(x)
    explain("  WITHOUT autocast:")
    for name in ["fc1", "fc2"]:
        d = captured_dtypes.get(name, {})
        print(f"    {name}: input={d.get('input')}  output={d.get('output')}")

    # WITH autocast
    with autocast(device_type=DEVICE, enabled=True):
        with torch.no_grad():
            _ = model(x)
    print()
    explain("  WITH autocast:")
    for name in ["fc1", "fc2"]:
        d = captured_dtypes.get(name, {})
        print(f"    {name}: input={d.get('input')}  output={d.get('output')}")

    for h in handles:
        h.remove()

    explain("",
            "  With autocast, Linear outputs are float16 on CUDA.",
            "  This triggers the FP16 Tensor Core path → significant speedup.",
            )


def lesson_4_full_amp_loop():
    lesson(4, "The Full AMP Training Loop")
    explain(
        "Here is the complete AMP training loop, step by step.",
        "Compare it to the plain FP32 loop — only 4 lines change.",
    )

    print("  FP32 loop (baseline):")
    print("  ┌───────────────────────────────────────────────────┐")
    print("  │  for x, y in loader:                             │")
    print("  │      logits = model(x)                           │")
    print("  │      loss   = criterion(logits, y)               │")
    print("  │      loss.backward()                             │")
    print("  │      optimizer.step()                            │")
    print("  │      optimizer.zero_grad()                       │")
    print("  └───────────────────────────────────────────────────┘")
    print()
    print("  AMP loop (4 changes marked with  ◄):")
    print("  ┌───────────────────────────────────────────────────┐")
    print("  │  scaler = GradScaler()                     ◄ (1)│")
    print("  │  for x, y in loader:                             │")
    print("  │      with autocast(device_type='cuda'):   ◄ (2)│")
    print("  │          logits = model(x)                       │")
    print("  │          loss   = criterion(logits, y)           │")
    print("  │      scaler.scale(loss).backward()        ◄ (3)│")
    print("  │      scaler.unscale_(optimizer)                  │")
    print("  │      clip_grad_norm_(model.parameters(), 1.0)    │")
    print("  │      scaler.step(optimizer)               ◄ (4)│")
    print("  │      scaler.update()                      ◄ (4)│")
    print("  │      optimizer.zero_grad()                       │")
    print("  └───────────────────────────────────────────────────┘")
    print()

    explain(
        "  Change (1): create GradScaler — manages the loss scale",
        "  Change (2): wrap forward + loss in autocast",
        "  Change (3): backward through scaled loss",
        "  Change (4): scaler.step() unscales internally then calls opt.step()",
        "              only if no inf/nan was found in gradients",
        "              scaler.update() adjusts scale for next iteration",
    )

    # Actually run it
    model = nn.Linear(8, 4).to(DEVICE)
    crit  = nn.CrossEntropyLoss()
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = GradScaler(enabled=(DEVICE == "cuda"))

    losses = []
    for step in range(5):
        x = torch.randn(32, 8, device=DEVICE)
        y = torch.randint(0, 4, (32,), device=DEVICE)

        with autocast(device_type=DEVICE, enabled=(DEVICE == "cuda")):
            logits = model(x)
            loss   = crit(logits, y)

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad()
        losses.append(loss.item())

    explain("  5 AMP training steps:")
    for i, l in enumerate(losses):
        print(f"    step {i+1}: loss = {l:.4f}")


def lesson_5_grad_accum_with_amp():
    lesson(5, "Gradient Accumulation with AMP")
    explain(
        "Gradient accumulation: run N mini-batches, accumulate gradients,",
        "then take one optimizer step. Simulates a batch N× larger.",
        "",
        "With AMP, there's a subtle interaction with GradScaler:",
        "  • We must only call scaler.update() at the actual optimizer step",
        "  • Otherwise the scale changes between accumulation steps",
        "",
        "Pattern:",
    )
    print("    accum_steps = 4")
    print("    opt.zero_grad()")
    print()
    print("    for step, (x, y) in enumerate(loader):")
    print("        with autocast(device_type='cuda'):")
    print("            loss = criterion(model(x), y) / accum_steps   ← divide!")
    print()
    print("        scaler.scale(loss).backward()   ← gradients ACCUMULATE in .grad")
    print()
    print("        if (step + 1) % accum_steps == 0:")
    print("            scaler.unscale_(opt)")
    print("            clip_grad_norm_(model.parameters(), 1.0)")
    print("            scaler.step(opt)    ← one real update per accum_steps batches")
    print("            scaler.update()")
    print("            opt.zero_grad()")
    print()
    explain(
        "  Why divide loss by accum_steps?",
        "  Gradient is the average over samples. If you run 4 batches of 64,",
        "  you want the same gradient as one batch of 256:",
        "    sum(loss_i) / N_total = sum(loss_i / accum_steps)",
        "  Without division, your effective lr is accum_steps× too large.",
    )

    # Demonstrate: grad accumulation produces same weights as large batch
    torch.manual_seed(42)
    model_large = nn.Linear(4, 2, bias=False)
    torch.manual_seed(42)
    model_accum = nn.Linear(4, 2, bias=False)

    crit = nn.MSELoss()
    torch.manual_seed(0)
    big_x = torch.randn(8, 4)
    big_y = torch.randn(8, 2)

    # Large batch
    opt_large = torch.optim.SGD(model_large.parameters(), lr=0.1)
    loss = crit(model_large(big_x), big_y)
    loss.backward()
    opt_large.step()

    # 4-step accumulation (batch of 2 each)
    opt_accum = torch.optim.SGD(model_accum.parameters(), lr=0.1)
    for i in range(4):
        x_mini = big_x[i*2:(i+1)*2]
        y_mini = big_y[i*2:(i+1)*2]
        loss_mini = crit(model_accum(x_mini), y_mini) / 4
        loss_mini.backward()
    opt_accum.step()

    diff = (model_large.weight.data - model_accum.weight.data).abs().max().item()
    show("Max weight diff (large vs accum):", f"{diff:.2e}  (should be ~0)")
    explain("  Mathematically identical results — accumulation works correctly.")


def lesson_6_measure_speedup():
    lesson(6, "Measuring the Speedup")
    explain(
        "Let's time FP32 vs AMP on a realistic matmul workload.",
        f"Running on: {DEVICE}",
    )

    if DEVICE == "cpu":
        explain("  CPU detected — AMP on CPU uses bfloat16 (no Tensor Cores).",
                "  For real speedup numbers, run this on a CUDA GPU.",
                "  Continuing with CPU as demonstration...")

    model = nn.Sequential(
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
    ).to(DEVICE)

    crit  = nn.MSELoss()
    opt32 = torch.optim.Adam(model.parameters(), lr=1e-3)

    N_ITERS = 20
    x = torch.randn(256, 512, device=DEVICE)
    y = torch.randn(256, 256, device=DEVICE)

    # Warmup
    for _ in range(3):
        with autocast(device_type=DEVICE, enabled=(DEVICE == "cuda")):
            crit(model(x), y).backward()
        opt32.zero_grad()

    # Time FP32
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        opt32.zero_grad()
        loss = crit(model(x.float()), y.float())
        loss.backward()
        opt32.step()
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t_fp32 = (time.perf_counter() - t0) / N_ITERS * 1000

    # Time AMP
    scaler = GradScaler(enabled=(DEVICE == "cuda"))
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        opt32.zero_grad()
        with autocast(device_type=DEVICE, enabled=(DEVICE == "cuda")):
            loss = crit(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(opt32)
        scaler.update()
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t_amp = (time.perf_counter() - t0) / N_ITERS * 1000

    show("FP32 time per step  :", f"{t_fp32:.2f} ms")
    show("AMP  time per step  :", f"{t_amp:.2f} ms")
    if t_amp > 0:
        show("Speedup             :", f"{t_fp32/t_amp:.2f}×")

    explain("",
            "  On an RTX 3090/4090, AMP typically gives 2-4× speedup for large matmuls.",
            "  Smaller models or CPU show less speedup (overhead dominates).",
            "  AMP pays off most for large batches and large models (transformers, ResNets).",
            )


# ── Run all lessons ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("╔" + "═" * (COLS - 2) + "╗")
    print(f"║  Mixed Precision Training (AMP) — A Deep Dive{' ' * (COLS - 49)}║")
    print(f"║  6 lessons: FP16, GradScaler, autocast, speedup{' ' * (COLS - 51)}║")
    print("╚" + "═" * (COLS - 2) + "╝")

    lesson_1_why_fp16()
    lesson_2_grad_scaler()
    lesson_3_autocast_ops()
    lesson_4_full_amp_loop()
    lesson_5_grad_accum_with_amp()
    lesson_6_measure_speedup()

    print()
    print("╔" + "═" * (COLS - 2) + "╗")
    print(f"║  ALL LESSONS COMPLETE {'─' * 10} Ready to run: python train.py{' ' * (COLS - 56)}║")
    print("╚" + "═" * (COLS - 2) + "╝")
    print()
