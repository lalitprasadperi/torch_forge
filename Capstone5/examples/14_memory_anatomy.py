"""
Example 14 — GPU Memory Anatomy

Shows exactly where every byte goes on the GPU during inference:
  1. Model weights (static, loaded once)
  2. KV cache (static pool, pre-allocated at startup)
  3. Activations (dynamic, allocated/freed per step)
  4. Overhead (CUDA context, PyTorch allocator bookkeeping)

Also shows how changing configs shifts the memory budget.

Run:  python examples/14_memory_anatomy.py
"""
import sys; sys.path.insert(0, ".")
import torch
from model.config import ModelConfig, CacheConfig


def analyze(model_config, cache_config, batch_size, seq_len, label):
    H  = model_config.n_heads
    D  = model_config.d_head
    L  = model_config.n_layers
    dM = model_config.d_model
    dF = model_config.d_ff
    V  = model_config.vocab_size
    bs = cache_config.block_size
    NB = cache_config.num_gpu_blocks
    fp = 2  # bytes per fp16 element

    # ── Model weights ──────────────────────────────────────────────────────────
    w_embed   = V * dM * fp
    w_attn    = L * 4 * dM * dM * fp          # Q,K,V,O projections
    w_ffn     = L * 2 * dM * dF * fp          # gate+up, down
    w_norm    = L * 2 * dM * fp               # RMSNorm per layer
    w_total   = w_embed + w_attn + w_ffn + w_norm

    # ── KV cache (pre-allocated pool) ─────────────────────────────────────────
    # (num_blocks, 2, block_size, n_heads, d_head) × n_layers
    kv_total  = NB * 2 * bs * H * D * fp * L

    # ── Activations (peak, during one forward pass) ───────────────────────────
    # Rough: each layer materialises ~4 tensors of shape (B, T, dM)
    # plus attention scores (B, H, T, T) for prefill
    act_ff    = batch_size * seq_len * dM * fp * 4 * L      # FFN activations
    act_attn  = batch_size * H * seq_len * seq_len * fp * L  # attention scores
    act_total = act_ff + act_attn

    # ── Total ──────────────────────────────────────────────────────────────────
    total = w_total + kv_total + act_total

    print(f"\n  ── {label} ──")
    print(f"  Config: d={dM}, layers={L}, heads={H}, vocab={V}")
    print(f"          blocks={NB}×{bs}tok, batch={batch_size}, seq={seq_len}")
    print()
    print(f"  Weights:     {w_total/1e6:8.1f} MB   ({w_total/total*100:.0f}%)")
    print(f"    embed:     {w_embed/1e6:8.1f} MB")
    print(f"    attention: {w_attn/1e6:8.1f} MB")
    print(f"    FFN:       {w_ffn/1e6:8.1f} MB")
    print(f"  KV cache:    {kv_total/1e6:8.1f} MB   ({kv_total/total*100:.0f}%)")
    print(f"  Activations: {act_total/1e6:8.1f} MB   ({act_total/total*100:.0f}%)")
    print(f"  ─────────────────────────────────────")
    print(f"  TOTAL:       {total/1e6:8.1f} MB")

    max_context = NB * bs
    print(f"\n  Max concurrent tokens: {max_context:,}")
    return total, w_total, kv_total, act_total


print("=" * 60)
print("GPU Memory Anatomy — Where Every Byte Goes")
print("=" * 60)

# Nano model
nano = ModelConfig.nano(); nano.vocab_size = 256
analyze(nano, CacheConfig(block_size=16, num_gpu_blocks=256), batch_size=8, seq_len=64,
        label="Nano model (d=128, 4 layers)")

# GPT-2 small
gpt2s = ModelConfig.gpt2_small()
analyze(gpt2s, CacheConfig(block_size=16, num_gpu_blocks=2000), batch_size=16, seq_len=512,
        label="GPT-2 small (d=768, 12 layers)")

# GPT-2 small with INT8 KV
analyze(gpt2s, CacheConfig(block_size=16, num_gpu_blocks=4000), batch_size=16, seq_len=512,
        label="GPT-2 small + INT8 KV (2× more blocks)")

# ── Live measurement on actual GPU ────────────────────────────────────────────
if torch.cuda.is_available():
    from model.gpt import PagedGPT
    import torch

    print()
    print("=" * 60)
    print("Live GPU Memory (actual allocation)")
    print("=" * 60)

    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()

    config      = ModelConfig.nano(); config.vocab_size = 256
    model       = PagedGPT(config).cuda().half()
    after_model = torch.cuda.memory_allocated()

    kv_caches   = model.allocate_kv_caches(256, 16, torch.device("cuda"))
    after_kv    = torch.cuda.memory_allocated()

    B, T        = 4, 32
    token_ids   = torch.randint(0, config.vocab_size, (B, T), device="cuda")
    block_tables = torch.zeros(B, 256, dtype=torch.int32, device="cuda")
    for i in range(B): block_tables[i, :2] = torch.tensor([i*2, i*2+1])
    seq_lens    = torch.full((B,), T, dtype=torch.int32, device="cuda")

    with torch.no_grad():
        logits = model(token_ids, kv_caches, block_tables, seq_lens, is_prefill=True)
    after_fwd   = torch.cuda.max_memory_allocated()

    print(f"  Baseline (context):   {before/1e6:.1f} MB")
    print(f"  After model load:     {after_model/1e6:.1f} MB  (+{(after_model-before)/1e6:.1f} MB weights)")
    print(f"  After KV pre-alloc:   {after_kv/1e6:.1f} MB  (+{(after_kv-after_model)/1e6:.1f} MB KV cache)")
    print(f"  Peak during forward:  {after_fwd/1e6:.1f} MB  (+{(after_fwd-after_kv)/1e6:.1f} MB activations)")
    total_mem  = torch.cuda.get_device_properties(0).total_memory
    print(f"  GPU total:            {total_mem/1e9:.1f} GB")
    print(f"  Used fraction:        {after_fwd/total_mem*100:.1f}%")
