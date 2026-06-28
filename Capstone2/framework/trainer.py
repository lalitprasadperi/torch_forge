"""
Trainer — the complete training loop.

Implements:
  • Standard forward → loss → backward → step cycle
  • Mixed precision (AMP) with GradScaler
  • Gradient accumulation (simulate large batch with small GPU memory)
  • Gradient clipping (prevent exploding gradients)
  • Automatic best-model checkpointing
  • Clean epoch logging via Logger

────────────────────────────────────────────────────────────────────────────────
CONCEPTS
────────────────────────────────────────────────────────────────────────────────

Mixed Precision (AMP)
─────────────────────
Modern GPUs (Ampere, Ada Lovelace) have dedicated Tensor Cores that run
float16 (FP16) 4-8× faster than float32. But FP16 has a small dynamic range
(max ~65504) which causes gradient underflow/overflow.

torch.amp.autocast: runs the forward pass in FP16 where safe, keeps FP32 for
sensitive ops (softmax, BN, loss). PyTorch decides automatically which ops to
cast — you just wrap the forward pass.

GradScaler: multiplies the loss by a large scalar before backward() to prevent
gradient underflow. After unscaling, checks for inf/nan. If clean, calls
optimizer.step(); otherwise skips the step (keeps the scale, tries next batch).

  loss = criterion(logits, targets)
  scaler.scale(loss).backward()           # backward in FP16 with scaled loss
  scaler.unscale_(optimizer)              # unscale grads back to FP32
  torch.nn.utils.clip_grad_norm_(...)     # clip AFTER unscaling
  scaler.step(optimizer)                  # step only if grads are finite
  scaler.update()                         # adjust scale for next iteration

Gradient Accumulation
─────────────────────
Problem: you want batch size 1024 but only 128 fit in GPU memory.
Solution: run 8 mini-batches of 128, accumulate gradients, then step once.

  for i, (x, y) in enumerate(loader):
      loss = criterion(model(x), y) / 8    ← divide to keep scale correct
      loss.backward()                       ← gradients ADD to .grad buffers
      if (i + 1) % 8 == 0:
          optimizer.step()
          optimizer.zero_grad()

  Key: gradients accumulate because we call optimizer.zero_grad() only every
  N steps, not every batch. PyTorch .grad tensors are += each backward() call.

Gradient Clipping
─────────────────
clip_grad_norm_(params, max_norm) computes the global L2 norm of all parameter
gradients and rescales them if norm > max_norm. This prevents a single bad
batch from causing a huge parameter update that destabilises training.
Rule of thumb: max_norm=1.0 for transformers, 5.0 for CNNs.
"""

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from .utils.metrics import AverageMeter, TopKAccuracy
from .logger import Logger
from .checkpoint import Checkpoint
from .evaluator import Evaluator


class Trainer:
    def __init__(
        self,
        model:       nn.Module,
        criterion:   nn.Module,
        optimizer:   Optimizer,
        scheduler:   LRScheduler,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        config:      dict,
        logger:      Logger,
        checkpoint:  Checkpoint,
        evaluator:   Evaluator,
    ):
        self.model        = model
        self.criterion    = criterion
        self.optimizer    = optimizer
        self.scheduler    = scheduler
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.config       = config
        self.logger       = logger
        self.checkpoint   = checkpoint
        self.evaluator    = evaluator

        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Mixed precision setup
        self.use_amp = config.get("use_amp", True) and self.device == "cuda"
        self.scaler  = GradScaler(enabled=self.use_amp)

        # Gradient accumulation
        self.grad_accum_steps = config.get("grad_accum_steps", 1)

        # Gradient clipping (0 = disabled)
        self.max_grad_norm = config.get("max_grad_norm", 1.0)

        self.epochs        = config["epochs"]
        self.start_epoch   = 0
        self.best_val_acc1 = 0.0

    # ── Resume from checkpoint ────────────────────────────────────────────────

    def resume(self, path: str | None = None) -> None:
        """Load checkpoint and set start_epoch so training continues."""
        ckpt_path = path or self.checkpoint.latest()
        if ckpt_path is None:
            return
        state = self.checkpoint.load(
            ckpt_path, self.model, self.optimizer, self.scheduler,
            self.scaler, device=self.device,
        )
        self.start_epoch   = state["epoch"] + 1
        self.best_val_acc1 = state["metrics"].get("val_acc1", 0.0)

    # ── Main training entry ───────────────────────────────────────────────────

    def fit(self) -> None:
        print(f"\n{'─'*60}")
        print(f"  Training on : {self.device}")
        print(f"  AMP enabled : {self.use_amp}")
        print(f"  Grad accum  : {self.grad_accum_steps} steps")
        print(f"  Epochs      : {self.start_epoch} → {self.epochs}")
        print(f"{'─'*60}\n")

        for epoch in range(self.start_epoch, self.epochs):
            self.logger.epoch_start()

            # ── Train ──
            train_metrics = self._train_epoch(epoch)
            self.logger.log(
                epoch  = epoch + 1,
                phase  = "train",
                loss   = train_metrics["loss"],
                acc1   = train_metrics["acc1"],
                acc5   = train_metrics["acc5"],
                lr     = self._current_lr(),
            )

            # ── Validate ──
            val_metrics = self.evaluator.evaluate(
                self.model, self.val_loader, self.criterion
            )
            self.logger.log(
                epoch  = epoch + 1,
                phase  = "val",
                loss   = val_metrics["loss"],
                acc1   = val_metrics["acc1"],
                acc5   = val_metrics["acc5"],
                lr     = self._current_lr(),
            )

            # ── Scheduler step (once per epoch) ──
            self.scheduler.step()

            # ── Checkpoint ──
            is_best = val_metrics["acc1"] > self.best_val_acc1
            if is_best:
                self.best_val_acc1 = val_metrics["acc1"]
            self.checkpoint.save(
                epoch     = epoch + 1,
                model     = self.model,
                optimizer = self.optimizer,
                scheduler = self.scheduler,
                scaler    = self.scaler,
                metrics   = {"val_acc1": val_metrics["acc1"],
                             "val_acc5": val_metrics["acc5"],
                             "val_loss": val_metrics["loss"]},
                config    = self.config,
                is_best   = is_best,
            )

        self.logger.close()
        print(f"\nTraining complete. Best val acc@1: {self.best_val_acc1:.2%}")

    # ── Single epoch ─────────────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> dict:
        self.model.train()

        loss_meter = AverageMeter("loss")
        acc1_meter = TopKAccuracy(k=1)
        acc5_meter = TopKAccuracy(k=5)

        self.optimizer.zero_grad()   # clear at epoch start

        for step, (inputs, targets) in enumerate(self.train_loader):
            inputs  = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # ── Forward (AMP context) ──────────────────────────────────────
            with autocast(device_type=self.device, enabled=self.use_amp):
                logits = self.model(inputs)
                loss   = self.criterion(logits, targets)

            # ── Scale loss for gradient accumulation ──────────────────────
            # Dividing by N makes the effective gradient equal to what you'd
            # get from a single batch of N× the size (mathematically correct).
            loss = loss / self.grad_accum_steps

            # ── Backward ──────────────────────────────────────────────────
            self.scaler.scale(loss).backward()

            # ── Optimizer step every grad_accum_steps batches ─────────────
            if (step + 1) % self.grad_accum_steps == 0:
                # Unscale before clipping — clip_grad_norm_ must see true FP32 grads
                self.scaler.unscale_(self.optimizer)

                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()   # reset grad buffers for next accum window

            # ── Metrics (unscaled loss for display) ───────────────────────
            loss_meter.update(loss.item() * self.grad_accum_steps, n=inputs.size(0))
            acc1_meter.update(logits.detach(), targets)
            acc5_meter.update(logits.detach(), targets)

        return {
            "loss": loss_meter.avg,
            "acc1": acc1_meter.compute(),
            "acc5": acc5_meter.compute(),
        }

    def _current_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]
