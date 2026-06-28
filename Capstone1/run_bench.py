#!/usr/bin/env python3
"""
run_bench.py — CLI entry point for the PyTorch Performance Lab.

Examples
--------
# Full sweep (all ops, all configs):
  python run_bench.py

# Only matmul and layernorm:
  python run_bench.py --ops matmul layernorm

# Quick run (fewer reps):
  python run_bench.py --warmup 5 --repeat 50

# Profile first config of one op (writes Chrome trace):
  python run_bench.py --profile softmax
  python run_bench.py --profile matmul --trace traces/matmul.json

# Roofline bound annotation (provide your GPU's peak specs):
  python run_bench.py --peak-tflops 40 --peak-bw 288

# Use torch.utils.benchmark instead of CUDA events:
  python run_bench.py --timer bench --ops gelu
"""

import argparse
import torch
from perf_lab.runner import run_all, ALL_OPS
from perf_lab.profiler_runner import profile_op, print_profile_table
from perf_lab.timing.bench_timer import BenchTimer


def main():
    parser = argparse.ArgumentParser(
        description="PyTorch Performance Lab",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--ops", nargs="*",
        help="ops to benchmark (default: all)\n"
             f"choices: {[o.name for o in ALL_OPS]}",
    )
    parser.add_argument("--warmup",      type=int,   default=20,   help="warmup iterations (default 20)")
    parser.add_argument("--repeat",      type=int,   default=200,  help="timed iterations  (default 200)")
    parser.add_argument("--timer",       choices=["cuda", "bench"], default="cuda",
                        help="timing method: cuda=CUDA events, bench=torch.utils.benchmark")
    parser.add_argument("--profile",     type=str,   default=None, help="profile a single op by name")
    parser.add_argument("--trace",       type=str,   default="traces/trace.json",
                        help="Chrome trace output path (used with --profile)")
    parser.add_argument("--peak-tflops", type=float, default=None,
                        help="GPU peak FP16 TFLOPS for roofline annotation")
    parser.add_argument("--peak-bw",     type=float, default=None,
                        help="GPU peak memory bandwidth in GB/s")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA GPU required. No CUDA device found.")
        return

    # Filter ops by name
    selected_ops = ALL_OPS
    if args.ops:
        selected_ops = [op for op in ALL_OPS if op.name in args.ops]
        unknown = set(args.ops) - {op.name for op in ALL_OPS}
        if unknown:
            print(f"Unknown op names: {unknown}")
            print(f"Available: {[o.name for o in ALL_OPS]}")
            return
        if not selected_ops:
            print("No ops matched.")
            return

    # ── Profile mode ──────────────────────────────────────────────────────────
    if args.profile:
        target = next((op for op in ALL_OPS if op.name == args.profile), None)
        if target is None:
            print(f"Unknown op '{args.profile}'. Available: {[o.name for o in ALL_OPS]}")
            return
        cfg = target.configs()[0]
        inputs = target.make_inputs(cfg, torch.device("cuda"))

        def fn():
            return target.run(inputs)

        print(f"\nProfiling: {target.name}  config: {cfg.get('label', cfg)}")
        events = profile_op(fn, trace_path=args.trace)
        print_profile_table(events)
        return

    # ── Benchmark mode ────────────────────────────────────────────────────────
    if args.timer == "bench":
        # torch.utils.benchmark path (self-contained, prints its own output)
        bench = BenchTimer(min_run_time=1.0)
        props = torch.cuda.get_device_properties(0)
        print(f"\nGPU   : {props.name}")
        print(f"Timer : torch.utils.benchmark (min_run_time=1s)\n")
        for op in selected_ops:
            print(f"── {op.name} ──")
            for cfg in op.configs():
                inputs = op.make_inputs(cfg, torch.device("cuda"))
                label  = cfg.get("label", str(cfg))

                def fn(inputs=inputs):
                    return op.run(inputs)

                mean_ms, iqr_ms = bench.measure(fn, label=f"{op.name}/{label}")
                print(f"  {label:<28} mean={mean_ms:>8.3f} ms  IQR={iqr_ms:>6.3f} ms")
            print()
        return

    run_all(
        ops=selected_ops,
        n_warmup=args.warmup,
        n_repeat=args.repeat,
        peak_tflops=args.peak_tflops,
        peak_bw_gb_s=args.peak_bw,
    )


if __name__ == "__main__":
    main()
