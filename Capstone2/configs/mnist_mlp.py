"""
Config for: MNIST + MLP

Dataset : MNIST  (60k train / 10k val, 28×28 grayscale, 10 classes)
Model   : MLP    (784 → 256 → 128 → 10, BN + Dropout)
Target  : ~99% val accuracy in 20 epochs
"""

CONFIG = {
    # ── Identity ──────────────────────────────────────────────────────────────
    "run_name": "mnist_mlp",

    # ── Data ──────────────────────────────────────────────────────────────────
    "dataset":      "mnist",
    "data_dir":     "./data",
    "num_workers":  4,
    "pin_memory":   True,
    "batch_size":   256,

    # ── Model ─────────────────────────────────────────────────────────────────
    "model":        "mlp",
    "input_dim":    784,
    "hidden_dims":  [256, 128],
    "num_classes":  10,
    "dropout":      0.2,

    # ── Optimizer ─────────────────────────────────────────────────────────────
    "optimizer":    "adamw",
    "lr":           1e-3,
    "weight_decay": 1e-4,

    # ── Scheduler ─────────────────────────────────────────────────────────────
    "scheduler":    "cosine",
    "epochs":       20,

    # ── Training ──────────────────────────────────────────────────────────────
    "use_amp":            True,
    "grad_accum_steps":   1,
    "max_grad_norm":      1.0,
    "seed":               42,

    # ── Paths ─────────────────────────────────────────────────────────────────
    "log_dir":     "experiments/logs",
    "ckpt_dir":    "experiments/checkpoints",
}
