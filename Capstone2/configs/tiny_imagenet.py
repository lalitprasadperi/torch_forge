"""
Config for: Tiny ImageNet + MiniResNet

Dataset : Tiny ImageNet  (100k train / 10k val, 64×64 RGB, 200 classes)
Model   : MiniResNet-18  (ResNet-18 adapted for 64×64 input)
Target  : ~55% top-1 / ~78% top-5 val accuracy in 90 epochs

Download Tiny ImageNet first:
  wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
  unzip tiny-imagenet-200.zip -d data/

Gradient accumulation: effective batch = batch_size × grad_accum_steps = 512
Using 2 accum steps so 16GB GPU can fit batch=256 comfortably.
"""

CONFIG = {
    # ── Identity ──────────────────────────────────────────────────────────────
    "run_name": "tiny_imagenet_resnet",

    # ── Data ──────────────────────────────────────────────────────────────────
    "dataset":      "tiny_imagenet",
    "data_dir":     "./data",
    "num_workers":  8,
    "pin_memory":   True,
    "batch_size":   256,

    # ── Model ─────────────────────────────────────────────────────────────────
    "model":        "resnet",
    "in_channels":  3,
    "num_classes":  200,

    # ── Optimizer ─────────────────────────────────────────────────────────────
    "optimizer":    "sgd",
    "lr":           0.1,
    "momentum":     0.9,
    "weight_decay": 1e-4,

    # ── Scheduler ─────────────────────────────────────────────────────────────
    "scheduler":       "warmup_cosine",
    "epochs":          90,
    "warmup_epochs":   5,

    # ── Training ──────────────────────────────────────────────────────────────
    "use_amp":            True,
    "grad_accum_steps":   2,      # effective batch = 256 × 2 = 512
    "max_grad_norm":      5.0,
    "seed":               42,

    # ── Paths ─────────────────────────────────────────────────────────────────
    "log_dir":     "experiments/logs",
    "ckpt_dir":    "experiments/checkpoints",
}
