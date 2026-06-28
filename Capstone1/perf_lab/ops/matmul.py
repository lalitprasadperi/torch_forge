import torch
from .base import BenchOp


class MatMulOp(BenchOp):
    """
    Matrix multiply: A(M,K) @ B(K,N) -> C(M,N)

    FLOPs = 2*M*K*N   (one multiply + one add per inner-product element)
    Arithmetic intensity grows with N: AI = 2*M*K*N / ((M*K + K*N + M*N)*bytes)
    -> Large square matrices are strongly compute-bound on modern GPUs.
    """

    name = "matmul"

    def make_inputs(self, config: dict, device: torch.device) -> tuple:
        M, K, N = config["M"], config["K"], config["N"]
        dtype = config.get("dtype", torch.float16)
        a = torch.randn(M, K, dtype=dtype, device=device)
        b = torch.randn(K, N, dtype=dtype, device=device)
        return (a, b)

    def run(self, inputs: tuple):
        return torch.mm(inputs[0], inputs[1])

    def flop_count(self, config: dict) -> int:
        return 2 * config["M"] * config["K"] * config["N"]

    def byte_count(self, config: dict) -> int:
        M, K, N = config["M"], config["K"], config["N"]
        eb = torch.tensor(0, dtype=config.get("dtype", torch.float16)).element_size()
        return (M * K + K * N + M * N) * eb

    def configs(self) -> list[dict]:
        return [
            {"M": s, "K": s, "N": s, "dtype": torch.float16, "label": f"{s}x{s}x{s}"}
            for s in [512, 1024, 2048, 4096]
        ] + [
            # LLM-typical: token projection (seq=1 decode step, hidden→4*hidden)
            {"M": 1,    "K": 4096,  "N": 16384, "dtype": torch.float16, "label": "llama7b_ffn_up"},
            {"M": 2048, "K": 4096,  "N": 16384, "dtype": torch.float16, "label": "llama7b_ffn_prefill"},
        ]
