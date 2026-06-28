import torch
import torch.utils.benchmark as benchmark
from typing import Callable


class BenchTimer:
    """
    Wraps torch.utils.benchmark.Timer.

    Advantages over raw CUDA events:
    - blocked_autorange() adaptively chooses how many iterations to run until
      the measurement is statistically stable (coefficient of variation < 0.1%).
    - Handles CPU overhead subtraction automatically.
    - Returns IQR instead of std, which is more robust to outliers.
    - Works on CPU too (no CUDA needed).

    Use this when you want a single, authoritative number. Use CudaTimer when
    you want per-iteration timing distributions or to control exact rep counts.
    """

    def __init__(self, min_run_time: float = 1.0):
        self.min_run_time = min_run_time

    def measure(self, fn: Callable, label: str = "") -> tuple[float, float]:
        """
        Returns (mean_ms, iqr_ms).

        blocked_autorange collects measurements in blocks so that adaptive
        noise filtering can work. The returned mean is the mean of block medians.
        """
        timer = benchmark.Timer(
            stmt="fn()",
            globals={"fn": fn},
            label=label,
        )
        result = timer.blocked_autorange(min_run_time=self.min_run_time)
        mean_ms = result.mean * 1000
        iqr_ms  = result.iqr  * 1000
        return mean_ms, iqr_ms
