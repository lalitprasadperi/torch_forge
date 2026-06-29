"""
Example 13 — Quantization: fp16 vs INT8 vs NF4

Compares three weight precisions on the nano model:
  fp16  — default, full precision
  int8  — symmetric per-channel quantization, 2× smaller
  nf4   — NF4 (QLoRA-style), 4× smaller

Measures:
  - Model weight memory (bytes)
  - Forward pass accuracy (max diff vs fp16 reference)
  - Inference latency per step

Run:  python examples/13_quantization_compare.py
"""
import sys; sys.path.insert(0, ".")
import time
import torch
from model.config import ModelConfig, CacheConfig
from model.gpt import PagedGPT
from quantization.int8_quant import (
    quantize_model_int8,
    quantize_nf4, dequantize_nf4,
    quantize_kv_cache, dequantize_kv_cache,
)


def model_bytes(model):
    p_bytes = sum(
        p.numel() * (1 if p.dtype == torch.int8 else p.element_size())
        for p in model.parameters()
    )
    b_bytes = sum(
        b.numel() * (1 if b.dtype == torch.int8 else b.element_size())
        for b in model.buffers()
    )
    return p_bytes + b_bytes


def run_model(model, token_ids, kv_caches, block_tables, seq_lens):
    with torch.no_grad():
        return model(token_ids, kv_caches, block_tables, seq_lens, is_prefill=True)


def timed(fn, n=20):
    for _ in range(3): fn()
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n): fn()
    if torch.cuda.is_available(): torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000


device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config      = ModelConfig.nano(); config.vocab_size = 256
B, T        = 2, 16
block_size  = 16
num_blocks  = 32

token_ids    = torch.randint(0, config.vocab_size, (B, T), device=device)
block_tables = torch.zeros(B, num_blocks, dtype=torch.int32, device=device)
block_tables[0, :2] = torch.tensor([0, 1]); block_tables[1, :2] = torch.tensor([2, 3])
seq_lens     = torch.tensor([T, T], dtype=torch.int32, device=device)

# ── Build fp16 reference ───────────────────────────────────────────────────────
model_fp16  = PagedGPT(config).to(device=device, dtype=torch.float16)
model_fp16.eval()
kv_fp16     = model_fp16.allocate_kv_caches(num_blocks, block_size, device)
logits_ref  = run_model(model_fp16, token_ids, kv_fp16, block_tables, seq_lens)

# ── Build INT8 quantized ───────────────────────────────────────────────────────
model_int8  = PagedGPT(config).to(device=device, dtype=torch.float16)
# Copy same weights so comparison is fair
model_int8.load_state_dict(model_fp16.state_dict())
model_int8  = quantize_model_int8(model_int8)
model_int8.eval()
kv_int8     = model_int8.allocate_kv_caches(num_blocks, block_size, device)
logits_int8 = run_model(model_int8, token_ids, kv_int8, block_tables, seq_lens)

print("=" * 60)
print("Weight Quantization Comparison")
print("=" * 60)
print(f"  Model:  nano (d={config.d_model}, layers={config.n_layers}, heads={config.n_heads})")
print(f"  Input:  B={B}, T={T}")
print()
print(f"  {'Precision':<12}  {'Memory (MB)':>12}  {'Max diff':>10}  {'Time (ms)':>10}")
print(f"  {'-'*52}")

fp16_bytes = model_bytes(model_fp16)
int8_bytes = model_bytes(model_int8)

diff_int8  = (logits_ref.float() - logits_int8.float()).abs().max().item()

t_fp16 = timed(lambda: run_model(model_fp16, token_ids, kv_fp16, block_tables, seq_lens))
t_int8 = timed(lambda: run_model(model_int8, token_ids, kv_int8, block_tables, seq_lens))

print(f"  {'fp16':<12}  {fp16_bytes/1e6:>12.2f}  {'ref':>10}  {t_fp16:>10.2f}")
print(f"  {'int8':<12}  {int8_bytes/1e6:>12.2f}  {diff_int8:>10.4f}  {t_int8:>10.2f}")
print(f"  {'ratio':<12}  {fp16_bytes/int8_bytes:>11.2f}×                      ")


# ── NF4 single-layer demo ──────────────────────────────────────────────────────
print()
print("=" * 60)
print("NF4 (QLoRA-style) — Single Weight Tensor Demo")
print("=" * 60)
w = model_fp16.blocks[0].attn.qkv_proj.weight.float().cpu()
n = w.numel()
if n % 2 != 0:
    w_pad = torch.cat([w.flatten(), torch.zeros(1)])
else:
    w_pad = w.flatten()

packed, absmax = quantize_nf4(w_pad)
w_dq           = dequantize_nf4(packed, absmax, w_pad.numel())
w_dq           = w_dq[:n].reshape(w.shape)
max_diff_nf4   = (w.flatten() - w_dq.flatten()).abs().max().item()

orig_bytes  = n * 4            # float32
nf4_bytes   = (n + 1) // 2    # 4-bit packed
scale_bytes = 4                # one float32 absmax

print(f"  Weight shape:     {list(w.shape)}")
print(f"  Original (fp32):  {orig_bytes/1024:.1f} KB")
print(f"  NF4 packed:       {nf4_bytes/1024:.1f} KB  ({orig_bytes/nf4_bytes:.1f}× smaller)")
print(f"  Max weight diff:  {max_diff_nf4:.4f}")
print()
print("  NF4 uses 16 levels optimised for normally-distributed weights.")
print("  At large scale, NF4 gives near-fp16 perplexity at 4× compression.")


# ── KV Cache quantization ──────────────────────────────────────────────────────
print()
print("=" * 60)
print("KV Cache INT8 Quantization")
print("=" * 60)
kv = torch.randn(num_blocks, 2, block_size, config.n_heads, config.d_head,
                  device=device, dtype=torch.float16)
kv_q, scale = quantize_kv_cache(kv)
kv_dq       = dequantize_kv_cache(kv_q, scale).half()

kv_fp16_mb  = kv.numel() * 2 / 1e6
kv_int8_mb  = (kv_q.numel() + scale.numel() * 4) / 1e6
kv_diff     = (kv.float() - kv_dq.float()).abs().max().item()

print(f"  KV cache FP16:   {kv_fp16_mb:.3f} MB")
print(f"  KV cache INT8:   {kv_int8_mb:.3f} MB")
print(f"  Compression:     {kv_fp16_mb/kv_int8_mb:.2f}×")
print(f"  Max quant error: {kv_diff:.4f}")
print()
print("  INT8 KV cache doubles the number of concurrent sequences")
print("  at no quality loss for most prompts (error << 0.1).")
