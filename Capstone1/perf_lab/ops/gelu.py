import torch
import torch.nn.functional as F
from .base import BenchOp


class GELUOp(BenchOp):
    """
    GELU activation (tanh approximation): x * 0.5 * (1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))

    Used in transformer FFN blocks (typically applied after the first linear layer).
    Pure element-wise: reads once, writes once → bandwidth bound (AI ≈ 8/2 = 4).
    The tanh-approx variant is faster than the exact erf version.
    """

    name = "gelu"

    def make_inputs(self, config: dict, device: torch.device) -> tuple:
        B, T, D = config["B"], config["T"], config["D"]
        dtype = config.get("dtype", torch.float16)
        x = torch.randn(B, T, D, dtype=dtype, device=device)
        return (x,)

    def run(self, inputs: tuple):
        return F.gelu(inputs[0], approximate="tanh")

    def flop_count(self, config: dict) -> int:
        # tanh-approx GELU: ~8 FLOPs/element
        total = config["B"] * config["T"] * config["D"]
        return 8 * total

    def byte_count(self, config: dict) -> int:
        B, T, D = config["B"], config["T"], config["D"]
        eb = torch.tensor(0, dtype=config.get("dtype", torch.float16)).element_size()
        return 2 * B * T * D * eb

    def configs(self) -> list[dict]:
        return [
            {"B": 1, "T": 2048, "D": 3072,  "dtype": torch.float16, "label": "gpt2_ffn"},
            {"B": 1, "T": 2048, "D": 11008, "dtype": torch.float16, "label": "llama7b_ffn"},
            {"B": 4, "T": 2048, "D": 11008, "dtype": torch.float16, "label": "llama7b_ffn_b4"},
            {"B": 1, "T": 2048, "D": 28672, "dtype": torch.float16, "label": "llama70b_ffn"},
        ]
