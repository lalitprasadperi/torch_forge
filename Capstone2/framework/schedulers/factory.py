"""
Learning rate scheduler factory.

Why do we need a scheduler?
  A fixed learning rate is usually suboptimal:
  • Too high early → unstable, diverges
  • Too high late  → bounces around the minimum, never converges
  • Warmup phase   → gradient noise is high on batch 0; a tiny LR helps stabilise

Supported schedules
───────────────────
step:
  Multiply lr by gamma every step_size epochs.
  lr_t = lr_0 × gamma^(epoch // step_size)
  Simple, predictable. Good for CNNs.

cosine:
  lr_t = lr_min + 0.5 × (lr_max − lr_min) × (1 + cos(π × t/T))
  Smoothly decays from lr_0 to eta_min over T epochs, following a half-cosine.
  The "cosine annealing" trick: the slow end of the cosine lets the model
  fine-tune in the low-lr regime without overshooting.

warmup_cosine:
  Linear warmup for warmup_epochs, then cosine decay for the rest.
  Used in transformer training (BERT, ViT, GPT) because large models need
  a careful warmup to stabilise the first few gradient steps.
  Implemented with torch.optim.lr_scheduler.SequentialLR.

config keys:
  scheduler     : "step" | "cosine" | "warmup_cosine"
  epochs        : total training epochs
  step_size     : (for step) epoch interval between lr drops
  gamma         : (for step) multiplicative factor (default 0.1)
  eta_min       : (for cosine) minimum lr at end (default 0)
  warmup_epochs : (for warmup_cosine) number of warmup epochs (default 5)
"""

import torch.optim as optim
from torch.optim import Optimizer
from torch.optim import lr_scheduler


def build_scheduler(optimizer: Optimizer, config: dict) -> lr_scheduler.LRScheduler:
    name   = config.get("scheduler", "cosine").lower()
    epochs = config["epochs"]

    if name == "step":
        sched = lr_scheduler.StepLR(
            optimizer,
            step_size = config.get("step_size", max(1, epochs // 3)),
            gamma     = config.get("gamma", 0.1),
        )
    elif name == "cosine":
        sched = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max   = epochs,
            eta_min = config.get("eta_min", 0),
        )
    elif name == "warmup_cosine":
        warmup = config.get("warmup_epochs", 5)
        warmup_sched = lr_scheduler.LinearLR(
            optimizer,
            start_factor = 1e-3,   # tiny lr at epoch 0
            end_factor   = 1.0,    # full lr at epoch warmup
            total_iters  = warmup,
        )
        cosine_sched = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max   = max(1, epochs - warmup),
            eta_min = config.get("eta_min", 0),
        )
        sched = lr_scheduler.SequentialLR(
            optimizer,
            schedulers  = [warmup_sched, cosine_sched],
            milestones  = [warmup],
        )
    else:
        raise ValueError(f"Unknown scheduler: {name!r}. Choose: step, cosine, warmup_cosine")

    print(f"  Scheduler : {name}  epochs={epochs}")
    return sched
