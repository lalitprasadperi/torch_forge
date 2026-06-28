import torch
import torch.nn.functional as F
from .base import BenchOp


class SoftmaxOp(BenchOp):
    """
    Softmax over the last dimension of an attention score matrix: (B,H,T,T)

    Online softmax (Flash-Attention style) fuses max + exp + sum into one pass.
    The naive two-pass (max scan, then exp/sum) is what PyTorch's built-in does
    unless you use a fused kernel.

    Memory bound: AI ≈ 5/2 = 2.5 FLOPs/byte (well below GPU ridge point).
    """

    name = "softmax"

    def make_inputs(self, config: dict, device: torch.device) -> tuple:
        B, H, T = config["B"], config["H"], config["T"]
        dtype = config.get("dtype", torch.float16)
        x = torch.randn(B, H, T, T, dtype=dtype, device=device)
        return (x,)

    def run(self, inputs: tuple):
        return F.softmax(inputs[0], dim=-1)

    def flop_count(self, config: dict) -> int:
        # max subtract (N), exp (N), sum (N), divide (N), +1 for max reduction ≈ 5N
        B, H, T = config["B"], config["H"], config["T"]
        return 5 * B * H * T * T

    def byte_count(self, config: dict) -> int:
        B, H, T = config["B"], config["H"], config["T"]
        eb = torch.tensor(0, dtype=config.get("dtype", torch.float16)).element_size()
        return 2 * B * H * T * T * eb  # read input, write output

    def configs(self) -> list[dict]:
        return [
            {"B": 1, "H": 12, "T": 512,  "dtype": torch.float16, "label": "gpt2_T512"},
            {"B": 1, "H": 12, "T": 1024, "dtype": torch.float16, "label": "gpt2_T1024"},
            {"B": 1, "H": 32, "T": 2048, "dtype": torch.float16, "label": "llama_T2048"},
            {"B": 4, "H": 32, "T": 2048, "dtype": torch.float16, "label": "llama_b4_T2048"},
        ]
