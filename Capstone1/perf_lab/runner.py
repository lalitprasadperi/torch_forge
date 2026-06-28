"""
Benchmark runner: sweeps all ops × configs, times each with CUDA events,
computes TFLOPS / bandwidth / arithmetic intensity, prints an annotated table.
"""

import torch
import statistics

from .ops.base import BenchmarkResult
from .ops.matmul import MatMulOp
from .ops.conv2d import Conv2dOp
from .ops.layernorm import LayerNormOp
from .ops.softmax import SoftmaxOp
from .ops.gelu import GELUOp
from .ops.rmsnorm import RMSNormOp
from .timing.cuda_timer import CudaTimer
from . import metrics


ALL_OPS = [
    MatMulOp(),
    Conv2dOp(),
    LayerNormOp(),
    SoftmaxOp(),
    GELUOp(),
    RMSNormOp(),
]

W = 88   # table width

# Per-op educational headers: what to expect and why
_OP_THEORY = {
    "matmul": (
        "Matrix Multiply  A(M,K) @ B(K,N) → C(M,N)",
        [
            "  FLOPs = 2·M·K·N   (one multiply + one add per inner-product term)",
            "  Arithmetic Intensity  AI = 2·M·K·N / (M·K + K·N + M·N) bytes",
            "    grows linearly with matrix size: AI ≈ N/3 for square float16.",
            "  Small matrices → memory bound (AI < ridge point, data keeps reloading).",
            "  Large matrices → compute bound (data reused many times in L2/SRAM).",
            "  Watch TFLOPS column: should RISE with matrix size until GPU compute peak.",
            "  Watch GB/s column:   should FALL with matrix size (less bandwidth needed).",
        ],
    ),
    "conv2d": (
        "2-D Convolution  (N,C_in,H,W) * (C_out,C_in,kH,kW)",
        [
            "  FLOPs = 2·N·C_out·H_out·W_out·C_in·kH·kW",
            "  cuDNN picks the algorithm: Winograd / FFT / direct-GEMM per shape.",
            "  1×1 convolutions are essentially matrix multiplies — compute bound.",
            "  3×3 + small channels → often bandwidth bound due to low data reuse.",
            "  Expect higher TFLOPS than bandwidth-bound ops, lower than large matmul.",
        ],
    ),
    "layernorm": (
        "LayerNorm  (B,T,D) → normalise last dim, scale, shift",
        [
            "  Two-pass: (1) compute mean & variance over D; (2) normalise + scale.",
            "  AI ≈ 8 FLOPs / 6 bytes ≈ 1.3 FLOPs/byte → strongly MEMORY BOUND.",
            "  Key metric: GB/s (bandwidth), not TFLOPS.",
            "  Fused kernels (Triton, Flash-Norm) reduce to 1 pass and ~2× faster.",
            "  Expect GB/s to be a large fraction of GPU peak bandwidth.",
        ],
    ),
    "softmax": (
        "Softmax  attention scores (B,H,T,T) → row-wise probabilities",
        [
            "  Two-pass: (1) row max + exp; (2) sum + divide.",
            "  AI ≈ 5/2 = 2.5 FLOPs/byte → MEMORY BOUND.",
            "  Flash-Attention fuses softmax into one pass over the K/V matrix,",
            "  avoiding materialising the full (T,T) attention weight matrix.",
            "  T=2048, 32 heads: the attention matrix alone is 2048²×32×2 = 512 MB.",
        ],
    ),
    "gelu": (
        "GELU (tanh approx)  x · 0.5·(1 + tanh(√(2/π)·(x + 0.044715·x³)))",
        [
            "  Pure element-wise: reads once, writes once.",
            "  AI = 8 FLOPs / 2 bytes = 4 FLOPs/byte → MEMORY BOUND.",
            "  Key metric: GB/s. Achieved GB/s ÷ peak BW = kernel efficiency.",
            "  Used in FFN blocks: applied to the first linear's output.",
            "  D is the FFN hidden dim, typically 4× the model hidden dim.",
        ],
    ),
    "rmsnorm": (
        "RMSNorm  x / sqrt(mean(x²) + ε) · weight",
        [
            "  Single-pass computable (no mean subtraction unlike LayerNorm).",
            "  AI ≈ 6/4 = 1.5 FLOPs/byte → MEMORY BOUND.",
            "  LLaMA replaces LayerNorm with RMSNorm: faster, no bias term.",
            "  Our implementation upcasts x to float32 for numerical stability.",
            "  A fused Triton kernel avoids the float32 round-trip overhead.",
        ],
    ),
}


def _print_op_header(op_name: str):
    if op_name not in _OP_THEORY:
        return
    title, lines = _OP_THEORY[op_name]
    border = "─" * W
    print(f"\n  ┌{border}┐")
    print(f"  │  {title:<{W-2}}│")
    print(f"  ├{border}┤")
    for line in lines:
        print(f"  │{line:<{W}}│")
    print(f"  └{border}┘")


def _print_metrics_guide():
    print(f"\n  ┌{'─'*W}┐")
    print(f"  │  {'Column guide':<{W-2}}│")
    print(f"  ├{'─'*W}┤")
    guide = [
        "  ms(mean)  — average latency per call, measured with CUDA events",
        "  ±(std)    — standard deviation; low std = stable result",
        "  TFLOPS    — achieved floating-point throughput (higher = better for compute ops)",
        "  GB/s      — achieved memory bandwidth    (higher = better for memory-bound ops)",
        "  AI        — Arithmetic Intensity = FLOPs / Bytes (determines which metric matters)",
        "              AI >> ridge point → COMPUTE BOUND: watch TFLOPS",
        "              AI << ridge point → MEMORY BOUND:  watch GB/s",
        "  Bound     — COM = compute bound, MEM = memory bound (needs --peak-tflops --peak-bw)",
    ]
    for line in guide:
        print(f"  │{line:<{W}}│")
    print(f"  └{'─'*W}┘")


_COL = (
    f"  {'Op':<12} {'Config':<24} {'ms(mean)':>9} {'±(std)':>7}  "
    f"{'TFLOPS':>7} {'GB/s':>9} {'AI':>6}  {'Bound':<6}"
)
_SEP = f"  {'─'*12} {'─'*24} {'─'*9} {'─'*7}  {'─'*7} {'─'*9} {'─'*6}  {'─'*6}"


def run_all(
    ops=None,
    device: str = "cuda",
    n_warmup: int = 20,
    n_repeat: int = 200,
    peak_tflops: float | None = None,
    peak_bw_gb_s: float | None = None,
) -> list[BenchmarkResult]:
    """
    Benchmark all ops and print an annotated results table.

    peak_tflops / peak_bw_gb_s: your GPU's spec-sheet numbers for roofline
    annotation (memory- vs compute-bound column). Leave None to skip.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required — no GPU found")

    dev = torch.device(device)
    timer = CudaTimer(n_warmup=n_warmup, n_repeat=n_repeat)
    selected_ops = ops or ALL_OPS

    props = torch.cuda.get_device_properties(dev)
    print()
    print("═" * (W + 4))
    print("  PyTorch Performance Lab — Benchmark Results")
    print("═" * (W + 4))
    print(f"  GPU    : {props.name}")
    print(f"  VRAM   : {props.total_memory / 2**30:.1f} GiB")
    print(f"  Timing : {n_warmup} warmup iterations + {n_repeat} timed iterations (CUDA events)")
    if peak_tflops:
        print(f"  Peak   : {peak_tflops} TFLOPS  |  {peak_bw_gb_s} GB/s")
        ridge = peak_tflops / peak_bw_gb_s
        print(f"  Ridge  : {ridge:.1f} FLOPs/byte  "
              f"(ops with AI < {ridge:.0f} are memory-bound, AI > {ridge:.0f} are compute-bound)")

    _print_metrics_guide()

    results: list[BenchmarkResult] = []
    prev_op = None

    for op in selected_ops:
        _print_op_header(op.name)
        print()
        print(_COL)
        print(_SEP)

        for cfg in op.configs():
            inputs = op.make_inputs(cfg, dev)

            def fn(inputs=inputs):
                return op.run(inputs)

            mean_ms, std_ms = timer.measure(fn)

            flops  = op.flop_count(cfg)
            nbytes = op.byte_count(cfg)
            tf     = metrics.tflops(flops, mean_ms)
            bw     = metrics.bandwidth_gb_s(nbytes, mean_ms)
            ai     = metrics.arithmetic_intensity(flops, nbytes)

            bound = ""
            if peak_tflops and peak_bw_gb_s:
                info  = metrics.roofline(tf, bw, ai, peak_tflops, peak_bw_gb_s)
                bound = "COM" if info["bound"] == "compute" else "MEM"

            label = cfg.get("label", "")
            print(
                f"  {op.name:<12} {label:<24} {mean_ms:>9.3f} {std_ms:>7.3f}  "
                f"{tf:>7.3f} {bw:>9.1f} {ai:>6.1f}  {bound:<6}"
            )

            results.append(BenchmarkResult(
                op_name=op.name,
                config=cfg,
                latency_ms=mean_ms,
                latency_std_ms=std_ms,
                tflops=tf,
                bandwidth_gb_s=bw,
                arithmetic_intensity=ai,
            ))

        print()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("═" * (W + 4))
    print("  Summary: How to read these results")
    print("═" * (W + 4))
    print("""
  TFLOPS (Tera Floating-Point Operations per Second)
    Tells you: how fast the GPU's ALUs are working.
    Formula:   TFLOPS = FLOPs / latency_s / 1e12
    Peak FP16 TFLOPS for your GPU is on the spec sheet.
    Achieved / Peak = 'compute efficiency' (higher is better).

  GB/s (Gigabytes per Second memory bandwidth)
    Tells you: how fast data moves between VRAM and the SM caches.
    Formula:   GB/s = (bytes_read + bytes_written) / latency_s / 1e9
    Peak memory bandwidth for your GPU is on the spec sheet.

  Arithmetic Intensity (AI, FLOPs/byte)
    Tells you: how compute-heavy the op is relative to memory traffic.
    Formula:   AI = FLOPs / bytes
    Rule of thumb:
      AI < ridge point → memory-bandwidth bound   → optimise data layout
      AI > ridge point → compute bound             → use tensor cores / quantise

  What's a good result?
    memory-bound ops  (LayerNorm, Softmax, GELU, RMSNorm):
      Target:  achieved GB/s > 60% of GPU peak bandwidth
    compute-bound ops (large MatMul, Conv):
      Target:  achieved TFLOPS > 50% of GPU peak TFLOPS
    """)

    return results
