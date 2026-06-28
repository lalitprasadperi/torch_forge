"""
GPT-Nano — Tiny GPT for Shakespeare training on a single GPU.

~10M parameters, trains in ~15 minutes on RTX 2000.
This is based on Karpathy's nanoGPT settings.

Architecture:
  Vocab: 65 characters (Shakespeare character set)
  Context: 256 tokens
  d_model: 384, n_layers: 6, n_heads: 6
  FFN: 4 × 384 = 1536
  Positional: learned
  Normalisation: LayerNorm (standard)
  Activation: GELU
"""

from transformer.blocks.transformer import GPTConfig

CONFIG = dict(
    # Run identity
    run_name    = "gpt_nano_shakespeare",
    dataset     = "shakespeare",

    # Model
    model       = GPTConfig(
        vocab_size = 65,
        max_len    = 256,
        d_model    = 384,
        n_layers   = 6,
        n_heads    = 6,
        d_ff       = 1536,
        dropout    = 0.1,
        bias       = True,
        pos_enc    = "learned",
        use_flash  = True,
        use_rope   = False,
        use_rmsnorm= False,
        use_swiglu = False,
    ),

    # Training
    epochs          = 5000,          # iterations (steps), not epochs
    batch_size      = 64,
    max_len         = 256,
    lr              = 3e-4,
    weight_decay    = 1e-1,
    grad_clip       = 1.0,
    warmup_steps    = 100,
    use_amp         = True,
    eval_interval   = 250,           # evaluate every N steps
    eval_iters      = 50,            # batches to average for eval loss
    log_interval    = 50,
    seed            = 42,

    # Generation
    gen_interval    = 500,           # generate sample text every N steps
    gen_max_tokens  = 200,
    gen_temperature = 0.8,
    gen_top_k       = 40,

    # Paths
    data_dir   = "./data",
    ckpt_dir   = "./experiments/checkpoints",
    log_dir    = "./experiments/logs",
)
