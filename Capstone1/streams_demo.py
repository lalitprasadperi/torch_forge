#!/usr/bin/env python3
"""
streams_demo.py — CUDA streams, async execution, and correct GPU timing.

Run:  python streams_demo.py
      python streams_demo.py 2>&1 | less

Lessons
  1. The GPU execution model — why everything is asynchronous
  2. The wrong way to time GPU code (perf_counter)
  3. CUDA Events — the right way to time GPU code
  4. Cold vs warm: why benchmarks need warmup
  5. The default stream — serialised execution
  6. Non-default streams — concurrent execution
  7. Measuring stream overlap benefit
  8. Cross-stream synchronisation with CUDA events
  9. H2D prefetch pattern — hiding memory transfer latency
"""

import torch
import time

W = 65

if not torch.cuda.is_available():
    print("CUDA not available — this demo requires a GPU.")
    raise SystemExit

device = torch.device("cuda")


def lesson(num, title):
    print(f"\n{'═' * W}")
    print(f"  LESSON {num}: {title}")
    print(f"{'═' * W}")


def explain(*lines):
    print()
    for line in lines:
        print(f"  {line}")
    print()


def show(label, value, width=42):
    print(f"  >>> {label:<{width}} {value}")


def code(text):
    print(f"  [ {text} ]")


def divider():
    print(f"  {'─' * (W - 2)}")


# ── Introduction ──────────────────────────────────────────────────────────────

print()
print("╔" + "═" * (W - 2) + "╗")
print("║" + "  CUDA Streams & GPU Timing".center(W - 2) + "║")
print("╚" + "═" * (W - 2) + "╝")

explain(
    f"GPU: {torch.cuda.get_device_name(0)}",
    f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB",
    "",
    "This demo answers two questions every GPU programmer must understand:",
    "",
    "  Q1: How does the GPU actually execute code?",
    "      Answer: ASYNCHRONOUSLY. CPU launches work and moves on.",
    "",
    "  Q2: How do I measure how long a GPU kernel takes?",
    "      Answer: CUDA Events — NOT time.perf_counter().",
)


# ══════════════════════════════════════════════════════════════════════════════
lesson(1, "The GPU Execution Model — Asynchronous by Default")
# ══════════════════════════════════════════════════════════════════════════════

explain(
    "When you write  c = a @ b  on a CUDA tensor, here's what happens:",
    "",
    "  CPU timeline:",
    "  ────────────────────────────────────────────────────────────────",
    "  [  launch matmul kernel  ][ Python continues here immediately ]",
    "                  │",
    "                  ▼  (asynchronous: CPU does not wait)",
    "  GPU timeline:",
    "  ────────────────────────────────────────────────────────────────",
    "                 [  matmul kernel running on GPU SMs  ]",
    "",
    "  The CPU and GPU run IN PARALLEL.",
    "  The CPU puts work items into a QUEUE (called a STREAM).",
    "  The GPU picks work from the queue and executes it.",
    "",
    "  CONSEQUENCE: If you read the result immediately after launching the",
    "  kernel without synchronizing, you might get garbage — the GPU might",
    "  not be done yet!",
    "",
    "  torch.cuda.synchronize() makes the CPU BLOCK until the GPU finishes",
    "  all pending work on all streams.",
)


# ══════════════════════════════════════════════════════════════════════════════
lesson(2, "The Wrong Way to Time GPU Code")
# ══════════════════════════════════════════════════════════════════════════════

explain(
    "time.perf_counter() measures WALL-CLOCK time on the CPU.",
    "Because GPU kernels run asynchronously, perf_counter only measures",
    "how long it took the CPU to LAUNCH the kernel — not to run it.",
    "",
    "Let's prove this:",
)

a = torch.randn(2048, 2048, device=device, dtype=torch.float16)
b = torch.randn(2048, 2048, device=device, dtype=torch.float16)

# Wrong: just measuring kernel launch time
t0 = time.perf_counter()
c = a @ b   # launches asynchronously, CPU returns immediately
t1 = time.perf_counter()
wrong_ms = (t1 - t0) * 1000

# Right: synchronize before stopping the timer
t0 = time.perf_counter()
c = a @ b
torch.cuda.synchronize()   # CPU waits for GPU to finish
t1 = time.perf_counter()
right_ms = (t1 - t0) * 1000

code("# WRONG: no synchronize before t1")
code("t0 = perf_counter(); c = a@b; t1 = perf_counter()")
show("Measured (wrong, launch only)", f"{wrong_ms:.4f} ms")

code("# RIGHT: synchronize before stopping clock")
code("t0 = perf_counter(); c = a@b; synchronize(); t1 = perf_counter()")
show("Measured (right, full kernel) ", f"{right_ms:.3f} ms")

explain(
    f"The wrong measurement ({wrong_ms:.4f} ms) only captured the time",
    f"to LAUNCH the kernel — not to RUN it.",
    f"The correct measurement ({right_ms:.3f} ms) is ~{right_ms/wrong_ms:.0f}x larger.",
    "",
    "But even the 'right' way has a problem: it includes synchronization",
    "OVERHEAD and CPU-side Python latency. For production benchmarks,",
    "use CUDA Events instead (next lesson).",
)


# ══════════════════════════════════════════════════════════════════════════════
lesson(3, "CUDA Events — The Right Way to Time GPU Kernels")
# ══════════════════════════════════════════════════════════════════════════════

explain(
    "A CUDA Event is a timestamp that the GPU itself records in its timeline.",
    "It has NOTHING to do with the CPU clock — it's measured in GPU time.",
    "",
    "Workflow:",
    "  1. Create two events: start and end.",
    "  2. start.record()  → GPU inserts a timestamp when it REACHES this point.",
    "  3. Launch your kernel.",
    "  4. end.record()    → GPU inserts another timestamp AFTER the kernel.",
    "  5. torch.cuda.synchronize()  → CPU waits for all GPU work to finish.",
    "  6. start.elapsed_time(end)   → returns GPU-measured milliseconds.",
    "",
    "This is pure GPU timing: it includes no CPU overhead, no Python latency,",
    "and it accounts for kernels that run while the CPU is doing other things.",
)

# Warmup first (explained in lesson 4)
for _ in range(5):
    _ = a @ b
torch.cuda.synchronize()

start = torch.cuda.Event(enable_timing=True)
end   = torch.cuda.Event(enable_timing=True)

code("start = torch.cuda.Event(enable_timing=True)")
code("end   = torch.cuda.Event(enable_timing=True)")
code("start.record()")
start.record()
code("c = a @ b   # kernel runs asynchronously")
c = a @ b
code("end.record()")
end.record()
code("torch.cuda.synchronize()   # flush GPU pipeline to CPU")
torch.cuda.synchronize()
code("elapsed = start.elapsed_time(end)  # in milliseconds")
elapsed = start.elapsed_time(end)

show("Elapsed time (CUDA event)", f"{elapsed:.3f} ms")

flops = 2 * 2048 * 2048 * 2048
tflops = flops / (elapsed / 1000) / 1e12
show("Achieved TFLOPS", f"{tflops:.2f}")
explain(
    f"2048×2048 fp16 matmul took {elapsed:.3f} ms.",
    f"FLOPs = 2 × 2048³ = {flops/1e9:.1f} GFLOPs",
    f"TFLOPS = {flops/1e9:.1f} GFLOPs / {elapsed:.3f} ms = {tflops:.2f} TFLOPS",
    "",
    "For our benchmark we use 200 repeats and take the mean and std.",
    "This smooths out run-to-run variance from cache effects and scheduling.",
)


# ══════════════════════════════════════════════════════════════════════════════
lesson(4, "Cold vs Warm — Why Benchmarks Must Warm Up")
# ══════════════════════════════════════════════════════════════════════════════

explain(
    "The FIRST time you run a CUDA kernel, several things happen:",
    "  • cuDNN selects the best algorithm (benchmarks heuristics)",
    "  • The CUDA JIT compiler optimises the kernel for your GPU",
    "  • L2 cache is cold — all data must come from VRAM",
    "  • GPU clocks may not be at full boost frequency yet",
    "",
    "All of these make the first run MUCH slower than steady-state.",
    "A good benchmark discards the first N runs (warmup) before measuring.",
)

def time_kernel(fn, n_warmup=0, n_repeat=10):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(n_repeat):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record()
        times.append((s, e))
    torch.cuda.synchronize()
    return [s.elapsed_time(e) for s, e in times]

fn = lambda: a @ b

# Allocate fresh tensors to force a cold run
a_cold = torch.randn(2048, 2048, device=device, dtype=torch.float16)
b_cold = torch.randn(2048, 2048, device=device, dtype=torch.float16)

cold_times = time_kernel(lambda: a_cold @ b_cold, n_warmup=0, n_repeat=10)
warm_times = time_kernel(lambda: a_cold @ b_cold, n_warmup=20, n_repeat=10)

import statistics
show("First 3 runs (cold, ms)      ", [f"{t:.2f}" for t in cold_times[:3]])
show("Warm runs mean  (ms)         ", f"{statistics.mean(warm_times):.3f}")
show("Warm runs stdev (ms)         ", f"{statistics.stdev(warm_times):.3f}")
show("Cold #1 / warm mean ratio    ", f"{cold_times[0] / statistics.mean(warm_times):.1f}x")
explain(
    f"Cold run #{1}: {cold_times[0]:.2f} ms",
    f"Warm runs avg:  {statistics.mean(warm_times):.3f} ms",
    f"Ratio: {cold_times[0]/statistics.mean(warm_times):.0f}x slower when cold.",
    "",
    "Our benchmark runner (CudaTimer) uses n_warmup=20 by default.",
    "After warmup, the stdev should be < 5% of the mean.",
    f"Here: {statistics.stdev(warm_times)/statistics.mean(warm_times)*100:.1f}% — " +
    ("good ✓" if statistics.stdev(warm_times)/statistics.mean(warm_times) < 0.05 else "consider more warmup"),
)


# ══════════════════════════════════════════════════════════════════════════════
lesson(5, "The Default Stream — Serialised Execution")
# ══════════════════════════════════════════════════════════════════════════════

explain(
    "A STREAM is a queue of GPU work items that executes in order.",
    "All GPU ops that don't specify a stream run on STREAM 0 (the default).",
    "",
    "Operations on the SAME stream execute strictly in order:",
    "",
    "  GPU Stream 0:",
    "  ┌──────────────────────────────────────────────────────┐",
    "  │  [  kernel A  ][  kernel B  ][  kernel C  ]         │",
    "  │   (must finish before B starts, B before C, etc.)   │",
    "  └──────────────────────────────────────────────────────┘",
    "",
    "This serialisation is the default and is SAFE — no data races.",
    "It's also a performance bottleneck if you have independent work.",
)

SIZE = 1536
x1 = torch.randn(SIZE, SIZE, device=device, dtype=torch.float16)
x2 = torch.randn(SIZE, SIZE, device=device, dtype=torch.float16)
x3 = torch.randn(SIZE, SIZE, device=device, dtype=torch.float16)

# Warmup
for _ in range(10):
    _ = x1 @ x1
    _ = x2 @ x2
    _ = x3 @ x3
torch.cuda.synchronize()

events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
          for _ in range(3)]

for s, e in events:
    s.record()
    _ = x1 @ x1
    e.record()
torch.cuda.synchronize()

single_ms = statistics.mean(s.elapsed_time(e) for s, e in events)
code("# Three matmuls on default stream (serialised)")
code("c1 = x1@x1; c2 = x2@x2; c3 = x3@x3")
show("Single matmul avg  (ms)", f"{single_ms:.3f}")

# Three sequential on default stream
for _ in range(5):
    _ = x1@x1; _ = x2@x2; _ = x3@x3
torch.cuda.synchronize()

s = torch.cuda.Event(enable_timing=True)
e = torch.cuda.Event(enable_timing=True)
s.record()
_ = x1@x1; _ = x2@x2; _ = x3@x3
e.record()
torch.cuda.synchronize()
seq_ms = s.elapsed_time(e)

show("Three sequential    (ms)", f"{seq_ms:.3f}")
show("Ratio (should ≈ 3x)    ", f"{seq_ms / single_ms:.2f}x")
explain(
    "Three sequential matmuls take approximately 3× one matmul.",
    "The GPU finishes each before starting the next.",
)


# ══════════════════════════════════════════════════════════════════════════════
lesson(6, "Non-Default Streams — Concurrent Execution")
# ══════════════════════════════════════════════════════════════════════════════

explain(
    "You can create additional streams and assign work to them.",
    "Operations on DIFFERENT streams CAN run concurrently (the GPU",
    "schedules them on separate SM groups if capacity allows).",
    "",
    "  GPU Stream 0:  [  kernel A  ]────────────────────────────────────",
    "  GPU Stream 1:              [  kernel B  ]──────────────────────",
    "  GPU Stream 2:                           [  kernel C  ]──────────",
    "",
    "  (Overlap possible if each kernel uses only part of the GPU)",
    "",
    "torch.cuda.Stream() creates a new stream.",
    "with torch.cuda.stream(s): routes all ops inside to stream s.",
)

s0 = torch.cuda.Stream()
s1 = torch.cuda.Stream()
s2 = torch.cuda.Stream()

for _ in range(10):
    with torch.cuda.stream(s0): _ = x1@x1
    with torch.cuda.stream(s1): _ = x2@x2
    with torch.cuda.stream(s2): _ = x3@x3
for s in (s0, s1, s2): s.synchronize()

overall_s = torch.cuda.Event(enable_timing=True)
overall_e = torch.cuda.Event(enable_timing=True)

overall_s.record()
with torch.cuda.stream(s0): _ = x1@x1
with torch.cuda.stream(s1): _ = x2@x2
with torch.cuda.stream(s2): _ = x3@x3
for s in (s0, s1, s2): s.synchronize()
overall_e.record()
torch.cuda.synchronize()
parallel_ms = overall_s.elapsed_time(overall_e)

show("Sequential  (ms)", f"{seq_ms:.3f}")
show("Parallel    (ms)", f"{parallel_ms:.3f}")
show("Speedup         ", f"{seq_ms / parallel_ms:.2f}x")
explain(
    f"Sequential: {seq_ms:.3f} ms    Parallel: {parallel_ms:.3f} ms",
    f"Speedup: {seq_ms/parallel_ms:.2f}x",
    "",
    "HOW MUCH OVERLAP YOU GET depends on GPU utilisation per op:",
    "  • If each op saturates the GPU (uses all SMs), there's no room",
    "    to run another op concurrently → speedup ≈ 1x",
    "  • If each op uses < 50% of SMs, two can run at the same time → 2x",
    "  • Large matmuls on modern GPUs are usually fully GPU-saturating,",
    "    so overlap benefit is small for big ops but significant for small ops.",
    "",
    "Stream overlap is most valuable when mixing small attention ops with",
    "large FFN matmuls, or overlapping memory copies with compute (next lesson).",
)


# ══════════════════════════════════════════════════════════════════════════════
lesson(7, "Cross-Stream Sync — CUDA Events as GPU Semaphores")
# ══════════════════════════════════════════════════════════════════════════════

explain(
    "Problem: Stream B needs data produced by Stream A.",
    "  If we call torch.cuda.synchronize(), we block the CPU until",
    "  ALL GPU work on ALL streams finishes — too aggressive.",
    "",
    "  Better: insert a GPU-side barrier that stalls ONLY Stream B",
    "  until Stream A has passed a specific point.",
    "",
    "Solution: CUDA Events as semaphores.",
    "",
    "  s_producer:  [── produce_data ──][ event.record() ]",
    "                                          │",
    "  s_consumer:  ────────────────[ stream.wait_event(event) ][── consume ──]",
    "                                          ▲",
    "                                GPU-side barrier: no CPU involved",
    "",
    "  stream.wait_event(e) tells the GPU scheduler:",
    "  'do not start work on this stream until event e fires'.",
    "  The CPU can keep running other Python code during all of this.",
)

s_producer = torch.cuda.Stream()
s_consumer = torch.cuda.Stream()
ready = torch.cuda.Event()

code("s_producer = torch.cuda.Stream()")
code("s_consumer = torch.cuda.Stream()")
code("ready = torch.cuda.Event()")
print()
code("with torch.cuda.stream(s_producer):")
code("    data = torch.randn(1024, 1024, device='cuda', dtype=float16)")
code("    produced = data @ data")
code("    ready.record(s_producer)   # GPU: 'I'm done producing'")
print()

with torch.cuda.stream(s_producer):
    data     = torch.randn(1024, 1024, device=device, dtype=torch.float16)
    produced = data @ data
    ready.record(s_producer)

code("s_consumer.wait_event(ready)   # GPU: 'wait until producer fires'")
code("with torch.cuda.stream(s_consumer):")
code("    result = produced.sum()")

s_consumer.wait_event(ready)
with torch.cuda.stream(s_consumer):
    result = produced.sum()

s_consumer.synchronize()
show("Consumer result (via GPU event sync)", f"{result.item():.2f}")
explain(
    "The CPU never blocked — it just submitted work to both queues and",
    "the GPU scheduler handled the ordering dependency internally.",
    "",
    "This technique is used in:",
    "  • vLLM: overlapping attention and FFN across layers",
    "  • NCCL: overlapping all-reduce with compute in DDP training",
    "  • cuDNN: pipelining conv layers in graph mode",
)


# ══════════════════════════════════════════════════════════════════════════════
lesson(8, "H2D Prefetch — Hiding Memory Transfer Latency")
# ══════════════════════════════════════════════════════════════════════════════

explain(
    "Moving data from CPU (host) to GPU (device) over PCIe is slow.",
    "  PCIe 4.0 × 16: ~25–32 GB/s peak",
    "  GPU compute: 40–300 TFLOPS",
    "",
    "Naive pattern — sequential copy then compute:",
    "",
    "  CPU→GPU copy batch 0   ──── compute batch 0 ────",
    "                                                   CPU→GPU copy batch 1   ──── compute batch 1 ────",
    "",
    "Better pattern — PREFETCH: overlap copy of batch N+1 with compute on batch N:",
    "",
    "  CPU→GPU copy batch 0 ──────",
    "                             compute batch 0 ─────────────",
    "                             CPU→GPU copy batch 1 ─────────",
    "                                                           compute batch 1 ────",
    "",
    "  Result: PCIe transfer is 'free' (hidden behind compute time).",
)

ROWS = 2048
cpu_batch_0 = torch.randn(ROWS, ROWS)    # in system RAM
cpu_batch_1 = torch.randn(ROWS, ROWS)

compute_stream = torch.cuda.Stream()
copy_stream    = torch.cuda.Stream()

# ── Baseline: sequential copy → compute × 2 ──
def sequential_baseline():
    gpu_0 = cpu_batch_0.cuda()                    # copy 0
    torch.cuda.synchronize()
    _result0 = gpu_0 @ gpu_0                      # compute 0
    torch.cuda.synchronize()
    gpu_1 = cpu_batch_1.cuda()                    # copy 1
    torch.cuda.synchronize()
    _result1 = gpu_1 @ gpu_1                      # compute 1
    torch.cuda.synchronize()

# ── Overlapped: copy 1 happens while computing 0 ──
def overlapped_prefetch():
    copy_done = torch.cuda.Event()
    with torch.cuda.stream(copy_stream):          # start copy 0
        gpu_0 = cpu_batch_0.cuda()
        copy_done.record(copy_stream)
    compute_stream.wait_event(copy_done)          # compute 0 waits for copy 0
    with torch.cuda.stream(compute_stream):
        _result0 = gpu_0 @ gpu_0                  # compute 0
    copy_done2 = torch.cuda.Event()
    with torch.cuda.stream(copy_stream):          # OVERLAP: copy 1 during compute 0
        gpu_1 = cpu_batch_1.cuda()
        copy_done2.record(copy_stream)
    compute_stream.wait_event(copy_done2)
    with torch.cuda.stream(compute_stream):
        _result1 = gpu_1 @ gpu_1                  # compute 1
    compute_stream.synchronize()

# Warmup
for _ in range(3):
    sequential_baseline()
    overlapped_prefetch()

# Timed runs
N_TRIALS = 5
seq_times = []
for _ in range(N_TRIALS):
    t0 = time.perf_counter()
    sequential_baseline()
    seq_times.append((time.perf_counter() - t0) * 1000)

ovl_times = []
for _ in range(N_TRIALS):
    t0 = time.perf_counter()
    overlapped_prefetch()
    ovl_times.append((time.perf_counter() - t0) * 1000)

seq_ms = statistics.mean(seq_times)
ovl_ms = statistics.mean(ovl_times)

show("Sequential  (copy+compute × 2) ms", f"{seq_ms:.1f}")
show("Overlapped  (prefetch)          ms", f"{ovl_ms:.1f}")
show("Speedup                           ", f"{seq_ms/ovl_ms:.2f}x")
explain(
    f"Sequential: {seq_ms:.1f} ms   Overlapped: {ovl_ms:.1f} ms   Speedup: {seq_ms/ovl_ms:.2f}x",
    "",
    "In real workloads (DataLoader with pin_memory=True + num_workers > 0),",
    "PyTorch handles prefetching automatically. Understanding streams lets you",
    "implement custom pipelines (e.g. inference with chunked KV cache filling).",
)

print()
print("╔" + "═" * (W - 2) + "╗")
print("║" + "  Streams Demo Complete!".center(W - 2) + "║")
print("╠" + "═" * (W - 2) + "╣")
print("║" + "  Key takeaways:".ljust(W - 2) + "║")
print("║" + "    • GPU runs ASYNC — always sync before reading results".ljust(W - 2) + "║")
print("║" + "    • Use CUDA Events, NOT time.perf_counter() for GPU timing".ljust(W - 2) + "║")
print("║" + "    • Cold first run can be 100x slower — always warm up".ljust(W - 2) + "║")
print("║" + "    • Streams enable compute/copy overlap → free throughput".ljust(W - 2) + "║")
print("║" + "    • event.record() + stream.wait_event() = GPU semaphore".ljust(W - 2) + "║")
print("╚" + "═" * (W - 2) + "╝")
print()
