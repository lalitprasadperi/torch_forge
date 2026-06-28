"""
Optimizer factory.

An optimizer holds a reference to model.parameters() and calls
.step() to apply the parameter update after gradients are computed.

The three workhorses:
─────────────────────

SGD (Stochastic Gradient Descent):
  θ ← θ − lr × ∇θ L                  (plain SGD)
  v ← momentum × v + ∇θ L            (with momentum)
  θ ← θ − lr × v

  momentum=0.9 means the update is 90% the previous velocity + 10% new gradient.
  This dampens oscillations and accelerates convergence in shallow directions.
  weight_decay adds an L2 regularisation term: ∇ → ∇ + wd × θ

Adam (Adaptive Moment Estimation):
  m ← β1 × m + (1−β1) × g            (1st moment — mean)
  v ← β2 × v + (1−β2) × g²           (2nd moment — variance)
  m̂ = m / (1−β1^t)                   (bias correction)
  v̂ = v / (1−β2^t)
  θ ← θ − lr × m̂ / (√v̂ + ε)

  Each parameter gets its own adaptive learning rate.
  Default β1=0.9, β2=0.999, ε=1e-8.

AdamW (Adam + decoupled weight decay):
  Same as Adam but weight_decay is applied directly to θ BEFORE the
  Adam update, not inside the gradient. This is the theoretically correct
  form of L2 regularisation with Adam (the original Adam conflates them).
  Use AdamW for transformers; Adam or SGD+momentum for CNNs.

config keys:
  optimizer  : "sgd" | "adam" | "adamw"
  lr         : float, learning rate (required)
  momentum   : float, for SGD only (default 0.9)
  weight_decay: float (default 1e-4)
  betas      : [β1, β2] for Adam/AdamW (default [0.9, 0.999])
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer


def build_optimizer(model: nn.Module, config: dict) -> Optimizer:
    name  = config.get("optimizer", "adamw").lower()
    lr    = config["lr"]
    wd    = config.get("weight_decay", 1e-4)
    params = model.parameters()

    if name == "sgd":
        opt = torch.optim.SGD(
            params,
            lr=lr,
            momentum=config.get("momentum", 0.9),
            weight_decay=wd,
            nesterov=True,   # Nesterov momentum: evaluate gradient at lookahead point
        )
    elif name == "adam":
        opt = torch.optim.Adam(
            params,
            lr=lr,
            betas=tuple(config.get("betas", [0.9, 0.999])),
            weight_decay=wd,
        )
    elif name == "adamw":
        opt = torch.optim.AdamW(
            params,
            lr=lr,
            betas=tuple(config.get("betas", [0.9, 0.999])),
            weight_decay=wd,
        )
    else:
        raise ValueError(f"Unknown optimizer: {name!r}. Choose: sgd, adam, adamw")

    print(f"  Optimizer : {opt.__class__.__name__}  lr={lr}  wd={wd}")
    return opt
