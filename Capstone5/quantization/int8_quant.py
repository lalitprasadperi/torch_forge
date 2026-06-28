"""
Quantization Experiments — INT8 and INT4 Weight Quantization

WHY QUANTIZE?
─────────────
A GPT-2 small model has ~117M parameters.
At fp16: 117M × 2 bytes = 234 MB
At int8: 117M × 1 byte  = 117 MB  → 2× smaller
At int4: 117M × 0.5 bytes = 58 MB → 4× smaller

For KV cache quantization, going from fp16 to int8:
  - Same number of KV blocks fit in 2× less memory
  - Or: double the maximum context length / batch size

METHODS COVERED:
  1. Dynamic INT8 (RTN): round-to-nearest per-tensor quantization
     Simple, fast, mild quality loss. Works for weights and activations.

  2. Weight-only INT8: quantize weights only, dequantize before matmul
     Good accuracy, compute still in fp16. Saves memory bandwidth.

  3. GPTQ / AWQ sketch: channel-wise quantization with scale factors
     State of the art for LLM quantization.

  4. KV cache quantization: INT8 KV blocks

NOTES:
  - Quantization and dequantization kernels should ideally be fused.
    Here we show the math clearly, not the optimised implementation.
  - AWQ requires calibration data; we show the concept without a full run.
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── INT8 Symmetric Quantization ───────────────────────────────────────────────

def quantize_int8_symmetric(
    x: torch.Tensor,
    dim: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Symmetric per-tensor (dim=None) or per-channel (dim=0) INT8 quantization.

    Returns:
      x_q:   INT8 quantized tensor
      scale: float32 scale factors

    Dequantize with: x_dq = x_q.float() * scale
    """
    if dim is None:
        # Per-tensor: one scale for the whole tensor
        amax = x.abs().max()
        scale = amax / 127.0
        x_q = (x / scale.clamp(min=1e-8)).round().clamp(-128, 127).to(torch.int8)
    else:
        # Per-channel: one scale per output channel (dim=0)
        amax = x.abs().amax(dim=list(range(1, x.ndim)), keepdim=True)
        scale = amax / 127.0
        x_q = (x / scale.clamp(min=1e-8)).round().clamp(-128, 127).to(torch.int8)
        scale = scale.squeeze()

    return x_q, scale


def dequantize_int8(x_q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize INT8 tensor back to float."""
    if scale.ndim == 0:
        return x_q.float() * scale
    # Per-channel: scale shape (out_features,), x_q shape (out_features, in_features)
    return x_q.float() * scale.unsqueeze(-1)


# ── INT8 Linear Layer ─────────────────────────────────────────────────────────

class Int8Linear(nn.Module):
    """
    Linear layer with INT8 weight quantization.

    Weights are stored as int8 + per-channel scale.
    Activations stay in fp16. Dequantize just before the matmul.

    Memory: weight storage is 1 byte/param (vs 2 for fp16)
    Compute: matmul is still fp16 (no int8 accumulation)

    For int8 accumulation (faster on modern GPUs), see:
    torch._C._nn.linear with int8 inputs — not yet stable in PyTorch.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.register_buffer("weight_q", torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer("weight_scale", torch.ones(out_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

    @classmethod
    def from_float(cls, fp_layer: nn.Linear) -> "Int8Linear":
        """Convert an existing fp16/fp32 Linear to Int8Linear."""
        layer = cls(fp_layer.in_features, fp_layer.out_features, fp_layer.bias is not None)
        with torch.no_grad():
            w_q, scale = quantize_int8_symmetric(fp_layer.weight.float().cpu(), dim=0)
            layer.weight_q.copy_(w_q)
            layer.weight_scale.copy_(scale)
            if fp_layer.bias is not None:
                layer.bias = nn.Parameter(fp_layer.bias.clone())
        # Move buffers/params to the same device as the source layer
        return layer.to(fp_layer.weight.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize weights: (out_features, in_features) float
        w = dequantize_int8(self.weight_q, self.weight_scale).to(x.dtype)
        return F.linear(x, w, self.bias)

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, dtype=int8"


# ── INT4 (NF4-style) Quantization ────────────────────────────────────────────

_NF4_LEVELS = torch.tensor([
    -1.0, -0.6961, -0.5251, -0.3949, -0.2844,
    -0.1848, -0.0910,  0.0000,
     0.0796,  0.1609,  0.2461,  0.3379,
     0.4407,  0.5626,  0.7230,  1.0
], dtype=torch.float32)


def quantize_nf4(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    NF4 quantization (bitsandbytes style, QLoRA paper).

    Quantizes to the 16 NF4 levels that have equal frequency under
    a standard normal distribution — optimal for normally-distributed weights.

    Returns:
      x_q:    uint8 packed as 4-bit values (2 per byte) — shape (n//2,)
      absmax: per-block maximum for dequantization
    """
    amax  = x.abs().max()
    x_norm = x / amax.clamp(min=1e-8)   # normalise to [-1, 1]

    # Find nearest NF4 level for each element
    levels = _NF4_LEVELS.to(x.device)
    dists  = (x_norm.unsqueeze(-1) - levels.unsqueeze(0)).abs()
    q_idx  = dists.argmin(dim=-1).to(torch.uint8)   # 0–15 values

    # Pack two 4-bit values per byte
    assert q_idx.numel() % 2 == 0, "need even number of elements for 4-bit packing"
    packed = (q_idx[0::2] << 4) | q_idx[1::2]

    return packed, amax


def dequantize_nf4(packed: torch.Tensor, absmax: torch.Tensor, n: int) -> torch.Tensor:
    """Reverse of quantize_nf4."""
    # Unpack
    high = (packed >> 4) & 0xF
    low  = packed & 0xF
    q_idx = torch.zeros(n, dtype=torch.uint8, device=packed.device)
    q_idx[0::2] = high
    q_idx[1::2] = low

    levels = _NF4_LEVELS.to(packed.device)
    return levels[q_idx.long()] * absmax


# ── Quantize a full model ─────────────────────────────────────────────────────

def quantize_model_int8(model: nn.Module, skip: Tuple[str, ...] = ("lm_head",)) -> nn.Module:
    """
    Walk the model tree and replace nn.Linear with Int8Linear.

    skip: layer name patterns to keep in fp16 (embedding layers,
          output projection — quantising these hurts quality most).
    """
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if any(s in name for s in skip):
            continue

        parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
        parent = model if not parent_name else _get_module(model, parent_name)
        int8_layer = Int8Linear.from_float(module)
        setattr(parent, child_name, int8_layer)

    return model


def _get_module(model: nn.Module, path: str) -> nn.Module:
    parts = path.split(".")
    m = model
    for p in parts:
        m = getattr(m, p)
    return m


# ── KV Cache Quantization ─────────────────────────────────────────────────────

def quantize_kv_cache(
    kv_cache: torch.Tensor,   # (num_blocks, 2, block_size, n_heads, d_head) fp16
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize KV cache to INT8 for memory savings.

    Uses per-block per-channel symmetric quantization.
    Scale shape: (num_blocks, 2, 1, n_heads, 1)

    Memory saving: 2× (fp16 → int8)
    Typical quality loss: <0.5% perplexity increase
    """
    # Scale per (block, key/value, head)
    scale = kv_cache.float().abs().amax(dim=(2, 4), keepdim=True) / 127.0
    kv_q  = (kv_cache.float() / scale.clamp(min=1e-8)).round().clamp(-128, 127).to(torch.int8)
    return kv_q, scale.squeeze((2, 4))


def dequantize_kv_cache(
    kv_q:   torch.Tensor,   # (num_blocks, 2, block_size, n_heads, d_head) int8
    scale:  torch.Tensor,   # (num_blocks, 2, n_heads) float32
) -> torch.Tensor:
    scale_exp = scale.unsqueeze(2).unsqueeze(-1)  # (num_blocks, 2, 1, n_heads, 1)
    return kv_q.float() * scale_exp


# ── Benchmark ─────────────────────────────────────────────────────────────────

def benchmark_quantization():
    """Compare fp16 vs int8 model memory and inference speed."""
    import sys
    sys.path.insert(0, "/home/jmd/Desktop/TrainHard/LearnTorch/Capstone5")
    from model.config import ModelConfig
    from model.gpt import PagedGPT

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("Quantization Benchmark")
    print("=" * 60)

    config = ModelConfig.nano()
    config.vocab_size = 256

    # FP16 model
    model_fp16 = PagedGPT(config).to(device=device, dtype=torch.float16)
    params_fp16 = sum(p.numel() * p.element_size() for p in model_fp16.parameters())
    print(f"\nFP16 model memory: {params_fp16 / 1e6:.1f} MB  ({model_fp16.num_parameters():,} params)")

    # INT8 model
    model_int8 = PagedGPT(config).to(device=device, dtype=torch.float16)
    model_int8 = quantize_model_int8(model_int8)
    params_int8 = sum(
        (p.numel() * p.element_size() if p.dtype != torch.int8 else p.numel())
        for p in model_int8.parameters()
    ) + sum(
        (b.numel() * b.element_size() if b.dtype != torch.int8 else b.numel())
        for b in model_int8.buffers()
    )
    print(f"INT8  model memory: {params_int8 / 1e6:.1f} MB")
    print(f"Compression ratio:  {params_fp16 / params_int8:.2f}×")

    # Speed comparison
    import time
    x = torch.randint(0, config.vocab_size, (1, 32), device=device)
    block_size = 16
    num_blocks = 32
    kv_caches_fp16 = model_fp16.allocate_kv_caches(num_blocks, block_size, device)
    kv_caches_int8 = model_int8.allocate_kv_caches(num_blocks, block_size, device)
    block_tables = torch.zeros(1, num_blocks, dtype=torch.int32, device=device)
    block_tables[0, :2] = torch.tensor([0, 1])
    seq_lens = torch.tensor([32], dtype=torch.int32, device=device)

    warmup = 5
    iters  = 20

    def run(model, kv_caches):
        with torch.no_grad():
            model(x, kv_caches, block_tables, seq_lens, is_prefill=True)
        torch.cuda.synchronize()

    for _ in range(warmup):
        run(model_fp16, kv_caches_fp16)
        run(model_int8, kv_caches_int8)

    start = time.perf_counter()
    for _ in range(iters):
        run(model_fp16, kv_caches_fp16)
    t_fp16 = (time.perf_counter() - start) / iters * 1000

    start = time.perf_counter()
    for _ in range(iters):
        run(model_int8, kv_caches_int8)
    t_int8 = (time.perf_counter() - start) / iters * 1000

    print(f"\nFP16 inference: {t_fp16:.2f} ms/step")
    print(f"INT8 inference: {t_int8:.2f} ms/step")
    print(f"(INT8 may be slower without int8 accumulation kernels)")

    # KV cache quantization
    print("\n--- KV Cache Quantization ---")
    kv = torch.randn(64, 2, 16, 8, 32, device=device, dtype=torch.float16)
    kv_bytes  = kv.numel() * 2
    kv_q, scale = quantize_kv_cache(kv)
    kv_q_bytes = kv_q.numel() + scale.numel() * 4
    kv_dq = dequantize_kv_cache(kv_q, scale).half()
    diff = (kv.float() - kv_dq.float()).abs().max().item()

    print(f"KV cache FP16:   {kv_bytes / 1e6:.2f} MB")
    print(f"KV cache INT8:   {kv_q_bytes / 1e6:.2f} MB")
    print(f"Max quant error: {diff:.4f}")
    print(f"Compression:     {kv_bytes / kv_q_bytes:.2f}×")


if __name__ == "__main__":
    benchmark_quantization()
