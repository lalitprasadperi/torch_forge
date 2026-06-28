import torch
from .base import BenchOp


class RMSNormOp(BenchOp):
    """
    RMSNorm: x / RMS(x) * weight   where RMS(x) = sqrt(mean(x^2) + eps)

    Used in LLaMA instead of LayerNorm: no mean subtraction, no bias.
    Single-pass computable (no mean dependency), but still memory bound.
    AI ≈ 6 / (2*eb_ratio) FLOPs/byte — similar to LayerNorm without bias read.
    """

    name = "rmsnorm"

    def make_inputs(self, config: dict, device: torch.device) -> tuple:
        B, T, D = config["B"], config["T"], config["D"]
        dtype = config.get("dtype", torch.float16)
        x      = torch.randn(B, T, D, dtype=dtype, device=device)
        weight = torch.ones(D,        dtype=dtype, device=device)
        return (x, weight)

    def run(self, inputs: tuple):
        x, weight = inputs
        # Computing in float32 for numerical stability then cast back
        x_f32  = x.float()
        rms    = x_f32.pow(2).mean(dim=-1, keepdim=True).add(1e-6).rsqrt()
        return (x_f32 * rms).to(x.dtype) * weight

    def flop_count(self, config: dict) -> int:
        # pow2 (N) + mean (N) + rsqrt (N) + mul*2 (2N) + cast overhead ≈ 6N
        total = config["B"] * config["T"] * config["D"]
        return 6 * total

    def byte_count(self, config: dict) -> int:
        B, T, D = config["B"], config["T"], config["D"]
        eb = torch.tensor(0, dtype=config.get("dtype", torch.float16)).element_size()
        # read x (BxTxD) + weight (D), write output (BxTxD)
        return (2 * B * T * D + D) * eb

    def configs(self) -> list[dict]:
        return [
            {"B": 1, "T": 2048, "D": 4096, "dtype": torch.float16, "label": "llama7b"},
            {"B": 4, "T": 2048, "D": 4096, "dtype": torch.float16, "label": "llama7b_b4"},
            {"B": 1, "T": 2048, "D": 8192, "dtype": torch.float16, "label": "llama70b"},
            {"B": 4, "T": 2048, "D": 8192, "dtype": torch.float16, "label": "llama70b_b4"},
        ]
