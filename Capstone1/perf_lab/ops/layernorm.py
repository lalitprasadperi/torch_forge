import torch
import torch.nn.functional as F
from .base import BenchOp


class LayerNormOp(BenchOp):
    """
    LayerNorm over the last dimension: (B, T, D) -> (B, T, D)

    Two-pass algorithm:
      Pass 1: compute mean and variance over D
      Pass 2: normalize, scale by weight, shift by bias

    This makes LayerNorm memory-bandwidth bound: AI ≈ 8/6 ≈ 1.33 FLOPs/byte.
    Fused kernels (Flash-Norm, Triton) reduce this to a single pass.
    """

    name = "layernorm"

    def make_inputs(self, config: dict, device: torch.device) -> tuple:
        B, T, D = config["B"], config["T"], config["D"]
        dtype = config.get("dtype", torch.float16)
        x      = torch.randn(B, T, D, dtype=dtype, device=device)
        weight = torch.ones(D,        dtype=dtype, device=device)
        bias   = torch.zeros(D,       dtype=dtype, device=device)
        return (x, weight, bias)

    def run(self, inputs: tuple):
        x, weight, bias = inputs
        return F.layer_norm(x, [x.shape[-1]], weight, bias)

    def flop_count(self, config: dict) -> int:
        # mean (N), var (2N), normalize (2N), scale+shift (2N+N) ≈ 8N
        total = config["B"] * config["T"] * config["D"]
        return 8 * total

    def byte_count(self, config: dict) -> int:
        B, T, D = config["B"], config["T"], config["D"]
        eb = torch.tensor(0, dtype=config.get("dtype", torch.float16)).element_size()
        # read x (BxTxD) + weight (D) + bias (D), write output (BxTxD)
        # second read of x in normalize pass counts too (conservative)
        return (3 * B * T * D + 2 * D) * eb

    def configs(self) -> list[dict]:
        return [
            {"B": 1, "T": 2048, "D": 768,  "dtype": torch.float16, "label": "gpt2_small"},
            {"B": 1, "T": 2048, "D": 1024, "dtype": torch.float16, "label": "gpt2_med"},
            {"B": 1, "T": 2048, "D": 4096, "dtype": torch.float16, "label": "llama_7b"},
            {"B": 4, "T": 2048, "D": 4096, "dtype": torch.float16, "label": "llama_7b_b4"},
            {"B": 1, "T": 2048, "D": 8192, "dtype": torch.float16, "label": "llama_70b"},
        ]
