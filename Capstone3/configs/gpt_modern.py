"""
GPT-Modern — LLaMA-style architecture on Shakespeare.

Same size as gpt_nano but uses modern components:
  • RoPE instead of learned positional embedding
  • RMSNorm instead of LayerNorm
  • SwiGLU FFN instead of GELU FFN
  • No bias anywhere

This lets you directly compare classic vs modern architecture
on the same dataset and training procedure.
"""

from transformer.blocks.transformer import GPTConfig

CONFIG = dict(
    run_name    = "gpt_modern_shakespeare",
    dataset     = "shakespeare",

    model       = GPTConfig(
        vocab_size  = 65,
        max_len     = 256,
        d_model     = 384,
        n_layers    = 6,
        n_heads     = 6,
        dropout     = 0.0,       # LLaMA uses no dropout
        bias        = False,
        pos_enc     = "rope",    # RoPE
        use_flash   = True,
        use_rope    = True,
        use_rmsnorm = True,      # RMSNorm
        use_swiglu  = True,      # SwiGLU
    ),

    # Same training settings as gpt_nano for fair comparison
    epochs          = 5000,
    batch_size      = 64,
    max_len         = 256,
    lr              = 3e-4,
    weight_decay    = 1e-1,
    grad_clip       = 1.0,
    warmup_steps    = 100,
    use_amp         = True,
    eval_interval   = 250,
    eval_iters      = 50,
    log_interval    = 50,
    seed            = 42,

    gen_interval    = 500,
    gen_max_tokens  = 200,
    gen_temperature = 0.8,
    gen_top_k       = 40,

    data_dir   = "./data",
    ckpt_dir   = "./experiments/checkpoints",
    log_dir    = "./experiments/logs",
)
