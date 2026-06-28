"""
Config for: CIFAR-10 + CNN

Dataset : CIFAR-10  (50k train / 10k val, 32×32 RGB, 10 classes)
Model   : CNN       (3 × ConvBlock + GlobalAvgPool + Linear)
Target  : ~87% val accuracy in 50 epochs
"""

CONFIG = {
    # ── Identity ──────────────────────────────────────────────────────────────
    "run_name": "cifar10_cnn",

    # ── Data ──────────────────────────────────────────────────────────────────
    "dataset":      "cifar10",
    "data_dir":     "./data",
    "num_workers":  4,
    "pin_memory":   True,
    "batch_size":   256,

    # ── Model ─────────────────────────────────────────────────────────────────
    "model":        "cnn",
    "in_channels":  3,
    "num_classes":  10,

    # ── Optimizer ─────────────────────────────────────────────────────────────
    "optimizer":    "sgd",
    "lr":           0.1,
    "momentum":     0.9,
    "weight_decay": 5e-4,

    # ── Scheduler ─────────────────────────────────────────────────────────────
    "scheduler":    "cosine",
    "epochs":       50,

    # ── Training ──────────────────────────────────────────────────────────────
    "use_amp":            True,
    "grad_accum_steps":   1,
    "max_grad_norm":      5.0,
    "seed":               42,

    # ── Paths ─────────────────────────────────────────────────────────────────
    "log_dir":     "experiments/logs",
    "ckpt_dir":    "experiments/checkpoints",
}
