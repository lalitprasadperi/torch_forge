"""
torch.profiler integration.

torch.profiler wraps CUPTI (CUDA Performance Tools Interface) to collect
kernel-level events: GPU start/end time, memory ops, tensor shapes, FLOPs.

The output is:
  1. A table (printed) showing each CUDA kernel's self-time and total-time.
  2. A Chrome trace JSON that you can load at chrome://tracing to see the
     full kernel timeline as colored bars on GPU/CPU lanes.

Schedule semantics:
  wait    : skip (don't record) — lets the system reach steady state
  warmup  : record shapes but discard — fills caches, JITs kernels
  active  : capture these steps fully
  repeat  : how many (wait+warmup+active) cycles before stopping (0=once)
"""

import torch
import torch.profiler
from pathlib import Path
from typing import Callable


def profile_op(
    fn: Callable,
    n_warmup: int = 5,
    n_active: int = 3,
    trace_path: str | None = None,
) -> torch.autograd.profiler_util.EventList:
    """
    Profile fn and return key_averages().
    If trace_path is given, writes a Chrome trace JSON there.
    """
    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    schedule = torch.profiler.schedule(
        wait=0,
        warmup=n_warmup,
        active=n_active,
        repeat=1,
    )

    with torch.profiler.profile(
        activities=activities,
        schedule=schedule,
        record_shapes=True,
        with_flops=True,
        profile_memory=True,
        on_trace_ready=_trace_handler(trace_path),
    ) as prof:
        for _ in range(n_warmup + n_active):
            fn()
            prof.step()

    return prof.key_averages()


def _trace_handler(trace_path: str | None):
    if trace_path is None:
        return None

    def handler(p: torch.profiler.profile):
        path = Path(trace_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        p.export_chrome_trace(str(path))
        print(f"  Chrome trace -> {path}  (open at chrome://tracing)")

    return handler


def print_profile_table(
    events: torch.autograd.profiler_util.EventList,
    sort_by: str = "cuda_time_total",
    row_limit: int = 15,
) -> None:
    print(events.table(sort_by=sort_by, row_limit=row_limit))
