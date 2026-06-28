"""
Checkpoint — save and restore full training state.

What goes into a checkpoint?
  A checkpoint captures everything needed to resume training from exactly
  where it stopped. If you only save model weights, you lose:
    • optimizer state   → momentum buffers, Adam m/v, are warm-started
    • scheduler state   → which epoch you're on, current LR
    • epoch counter     → know where to resume
    • best metric       → to correctly implement "save best model" logic
    • GradScaler state  → AMP loss scale history (avoids under/overflow)
    • config            → reproducibility; know what settings produced this run

  state_dict() / load_state_dict() explained:
    Every nn.Module has a .state_dict() that returns an OrderedDict of
    {name: tensor} for all parameters AND buffers. Saving this (not the model
    object itself) is portable across PyTorch versions and lets you load weights
    into a differently-structured class at inference time.

Checkpoint formats:
  torch.save(obj, path)  — pickles any Python object (including state_dicts)
  torch.load(path)       — unpickles it back

  We save as .pt (PyTorch tensors) not .pkl to signal content type.
"""

from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.amp import GradScaler


class Checkpoint:
    def __init__(self, ckpt_dir: str = "experiments/checkpoints", run_name: str = "run"):
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self.best_metric: float = 0.0

    # ── Saving ────────────────────────────────────────────────────────────────

    def save(
        self,
        epoch:     int,
        model:     nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        scaler:    GradScaler | None,
        metrics:   dict,
        config:    dict,
        is_best:   bool = False,
    ) -> Path:
        state = {
            "epoch":     epoch,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler":    scaler.state_dict() if scaler is not None else None,
            "metrics":   metrics,
            "config":    config,
        }

        path = self.ckpt_dir / f"{self.run_name}_epoch{epoch:03d}.pt"
        torch.save(state, path)

        if is_best:
            best_path = self.ckpt_dir / f"{self.run_name}_best.pt"
            torch.save(state, best_path)
            self.best_metric = metrics.get("val_acc1", 0.0)
            print(f"  [ckpt] Best model saved → {best_path.name}  "
                  f"(val_acc1={self.best_metric:.2%})")

        return path

    # ── Loading ───────────────────────────────────────────────────────────────

    def load(
        self,
        path:      str | Path,
        model:     nn.Module,
        optimizer: Optimizer | None = None,
        scheduler: LRScheduler | None = None,
        scaler:    GradScaler | None = None,
        device:    str = "cpu",
    ) -> dict:
        """
        Load checkpoint and restore all components in-place.

        Returns the saved metrics dict so the caller can inspect them.

        map_location=device:
          If the checkpoint was saved on GPU but we're loading on CPU
          (e.g. for inference or debugging), map_location redirects tensor
          storage to the specified device without requiring CUDA.
        """
        state = torch.load(path, map_location=device, weights_only=False)

        # model.load_state_dict(strict=True) requires exact key match.
        # strict=False allows loading a subset (e.g. pretrained backbone
        # without the classification head).
        model.load_state_dict(state["model"], strict=True)

        if optimizer is not None and "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None and "scheduler" in state:
            scheduler.load_state_dict(state["scheduler"])
        if scaler is not None and state.get("scaler") is not None:
            scaler.load_state_dict(state["scaler"])

        epoch   = state.get("epoch", 0)
        metrics = state.get("metrics", {})
        print(f"  [ckpt] Loaded {Path(path).name}  epoch={epoch}  "
              f"val_acc1={metrics.get('val_acc1', 0):.2%}")
        return state

    def latest(self) -> Path | None:
        """Return the most recently saved epoch checkpoint, or None."""
        pattern = f"{self.run_name}_epoch*.pt"
        ckpts   = sorted(self.ckpt_dir.glob(pattern))
        return ckpts[-1] if ckpts else None

    def best(self) -> Path | None:
        """Return the best checkpoint path, or None."""
        p = self.ckpt_dir / f"{self.run_name}_best.pt"
        return p if p.exists() else None
