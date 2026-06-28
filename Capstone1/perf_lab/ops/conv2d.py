import torch
import torch.nn.functional as F
from .base import BenchOp


def _out_size(in_size: int, kernel: int, stride: int, padding: int) -> int:
    return (in_size + 2 * padding - kernel) // stride + 1


class Conv2dOp(BenchOp):
    """
    2-D convolution: (N,C_in,H,W) * (C_out,C_in,kH,kW) -> (N,C_out,H_out,W_out)

    FLOPs = 2 * N * C_out * H_out * W_out * C_in * kH * kW
    cuDNN selects the best algorithm (Winograd, FFT, direct GEMM) per shape.
    """

    name = "conv2d"

    def make_inputs(self, config: dict, device: torch.device) -> tuple:
        N, C_in, H, W = config["N"], config["C_in"], config["H"], config["W"]
        C_out = config["C_out"]
        kH, kW = config["kH"], config["kW"]
        stride = config.get("stride", 1)
        padding = config.get("padding", 0)
        dtype = config.get("dtype", torch.float16)
        x = torch.randn(N, C_in, H, W, dtype=dtype, device=device)
        w = torch.randn(C_out, C_in, kH, kW, dtype=dtype, device=device)
        return (x, w, stride, padding)

    def run(self, inputs: tuple):
        x, w, stride, padding = inputs
        return F.conv2d(x, w, stride=stride, padding=padding)

    def flop_count(self, config: dict) -> int:
        N, C_in, H, W = config["N"], config["C_in"], config["H"], config["W"]
        C_out = config["C_out"]
        kH, kW = config["kH"], config["kW"]
        stride = config.get("stride", 1)
        padding = config.get("padding", 0)
        H_out = _out_size(H, kH, stride, padding)
        W_out = _out_size(W, kW, stride, padding)
        return 2 * N * C_out * H_out * W_out * C_in * kH * kW

    def byte_count(self, config: dict) -> int:
        N, C_in, H, W = config["N"], config["C_in"], config["H"], config["W"]
        C_out = config["C_out"]
        kH, kW = config["kH"], config["kW"]
        stride = config.get("stride", 1)
        padding = config.get("padding", 0)
        H_out = _out_size(H, kH, stride, padding)
        W_out = _out_size(W, kW, stride, padding)
        eb = torch.tensor(0, dtype=config.get("dtype", torch.float16)).element_size()
        inp    = N * C_in * H * W
        weight = C_out * C_in * kH * kW
        out    = N * C_out * H_out * W_out
        return (inp + weight + out) * eb

    def configs(self) -> list[dict]:
        base = {"stride": 1, "padding": 0, "dtype": torch.float16}
        return [
            {**base, "N": 1, "C_in": 3,   "H": 224, "W": 224, "C_out": 64, "kH": 7, "kW": 7,
             "stride": 2, "padding": 3, "label": "resnet_stem"},
            {**base, "N": 1, "C_in": 256, "H": 56,  "W": 56,  "C_out": 64, "kH": 1, "kW": 1,
             "label": "resnet_1x1_bottleneck"},
            {**base, "N": 1, "C_in": 64,  "H": 56,  "W": 56,  "C_out": 64, "kH": 3, "kW": 3,
             "padding": 1, "label": "resnet_3x3"},
            {**base, "N": 8, "C_in": 128, "H": 64,  "W": 64,  "C_out": 128, "kH": 3, "kW": 3,
             "padding": 1, "label": "batch8_128ch"},
        ]
