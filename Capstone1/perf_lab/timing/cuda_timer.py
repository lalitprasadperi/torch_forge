import torch
import statistics
from typing import Callable


class CudaTimer:
    """
    Times a callable using CUDA events.

    Why CUDA events and not time.perf_counter()?
    perf_counter() measures wall-clock time on the CPU. GPU kernels launch
    asynchronously — the CPU returns immediately while the GPU is still running.
    CUDA events are timestamps inserted *on the GPU timeline*, so they measure
    actual GPU execution time regardless of CPU/GPU overlap.

    Workflow:
      start.record()   -> GPU records a timestamp when it reaches this point
      kernel(...)      -> GPU starts work
      end.record()     -> GPU records another timestamp after the kernel
      synchronize()    -> CPU blocks until GPU flushes all pending work
      start.elapsed_time(end)  -> returns GPU-measured ms between events
    """

    def __init__(self, n_warmup: int = 20, n_repeat: int = 200):
        self.n_warmup = n_warmup
        self.n_repeat = n_repeat

    def measure(self, fn: Callable, *args, **kwargs) -> tuple[float, float]:
        """
        Run fn, return (mean_ms, std_ms).

        Warmup flushes:
          - CUDA kernel JIT compilation cache
          - L2 cache cold-start effects
          - GPU clock ramp-up on some drivers
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")

        for _ in range(self.n_warmup):
            fn(*args, **kwargs)
        torch.cuda.synchronize()

        starts = [torch.cuda.Event(enable_timing=True) for _ in range(self.n_repeat)]
        ends   = [torch.cuda.Event(enable_timing=True) for _ in range(self.n_repeat)]

        for start, end in zip(starts, ends):
            start.record()
            fn(*args, **kwargs)
            end.record()

        torch.cuda.synchronize()

        times_ms = [s.elapsed_time(e) for s, e in zip(starts, ends)]
        return statistics.mean(times_ms), statistics.stdev(times_ms)
