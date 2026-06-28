import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class BenchmarkResult:
    op_name: str
    config: dict
    latency_ms: float
    latency_std_ms: float
    tflops: float
    bandwidth_gb_s: float
    arithmetic_intensity: float


class BenchOp(ABC):
    name: str = "base"

    @abstractmethod
    def make_inputs(self, config: dict, device: torch.device) -> tuple:
        """Allocate and return all input tensors for one forward pass."""

    @abstractmethod
    def run(self, inputs: tuple) -> Any:
        """Execute the operation. Return value may be ignored."""

    @abstractmethod
    def flop_count(self, config: dict) -> int:
        """Total floating-point ops for one call (multiply-adds count as 2)."""

    @abstractmethod
    def byte_count(self, config: dict) -> int:
        """Bytes read + written (denominator for bandwidth calculation)."""

    @abstractmethod
    def configs(self) -> list[dict]:
        """Return the list of config dicts to sweep."""
