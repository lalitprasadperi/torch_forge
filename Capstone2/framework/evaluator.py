"""
Evaluator — runs the validation loop and computes accuracy.

This is intentionally separate from Trainer so you can:
  • Evaluate a checkpoint without re-training
  • Swap evaluation logic (e.g. test-time augmentation) without touching training

What happens in eval mode?
  model.eval() sets two things:
    1. BatchNorm: uses stored running_mean / running_var (not batch stats)
       During training, BN normalises each batch independently.
       At eval, the batch might be size 1 — that's too noisy. Instead,
       BN uses exponential moving averages accumulated during training.
    2. Dropout: disabled (all neurons pass through with full activation)
       Dropout is a training regulariser. At eval we want deterministic output.

  torch.no_grad():
    Disables the autograd engine — no computation graph is built, no
    intermediate activations are stored. 2-3× less memory, ~10% faster.
    Use this whenever you don't need gradients (val, test, inference).
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .utils.metrics import AverageMeter, TopKAccuracy


class Evaluator:
    def __init__(self, device: str = "cuda"):
        self.device = device

    def evaluate(
        self,
        model:      nn.Module,
        loader:     DataLoader,
        criterion:  nn.Module,
    ) -> dict:
        """
        Run one pass over the validation loader.

        Returns:
            dict with keys: loss, acc1, acc5
        """
        model.eval()

        loss_meter = AverageMeter("loss")
        acc1_meter = TopKAccuracy(k=1)
        acc5_meter = TopKAccuracy(k=5)

        with torch.no_grad():
            for inputs, targets in loader:
                inputs  = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                logits = model(inputs)
                loss   = criterion(logits, targets)

                loss_meter.update(loss.item(), n=inputs.size(0))
                acc1_meter.update(logits, targets)
                acc5_meter.update(logits, targets)

        return {
            "loss": loss_meter.avg,
            "acc1": acc1_meter.compute(),
            "acc5": acc5_meter.compute(),
        }
