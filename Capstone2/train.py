"""
train.py — CLI entry point for Capstone 2 training framework.

Usage:
  # Train on MNIST with default config
  python train.py --config mnist_mlp

  # Train on CIFAR-10
  python train.py --config cifar10_cnn

  # Resume from latest checkpoint
  python train.py --config cifar10_cnn --resume

  # Resume from specific checkpoint
  python train.py --config cifar10_cnn --resume-path experiments/checkpoints/cifar10_cnn_best.pt

  # Override any config key from the command line
  python train.py --config mnist_mlp --lr 5e-4 --epochs 10 --batch-size 128

  # Evaluate only (no training)
  python train.py --config mnist_mlp --eval-only --resume
"""

import argparse
import importlib
import sys
import torch
import torch.nn as nn
from pathlib import Path

# Add Capstone2 directory to path so `framework` and `configs` are importable
sys.path.insert(0, str(Path(__file__).parent))

from framework.models      import build_model
from framework.data        import build_datasets, build_dataloaders
from framework.optimizers  import build_optimizer
from framework.schedulers  import build_scheduler
from framework.logger      import Logger
from framework.checkpoint  import Checkpoint
from framework.evaluator   import Evaluator
from framework.trainer     import Trainer
from framework.utils       import set_seed


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Capstone 2 — Train PyTorch models")
    p.add_argument("--config", required=True,
                   choices=["mnist_mlp", "cifar10_cnn", "tiny_imagenet"],
                   help="Config module name in configs/")
    p.add_argument("--resume", action="store_true",
                   help="Resume from latest checkpoint")
    p.add_argument("--resume-path", type=str, default=None,
                   help="Resume from specific checkpoint path")
    p.add_argument("--eval-only", action="store_true",
                   help="Run evaluation on val set only, no training")
    p.add_argument("--device", type=str, default=None,
                   help="Override device (cuda / cpu)")

    # Config overrides
    p.add_argument("--lr",         type=float, default=None)
    p.add_argument("--epochs",     type=int,   default=None)
    p.add_argument("--batch-size", type=int,   default=None)
    p.add_argument("--seed",       type=int,   default=None)
    p.add_argument("--no-amp",     action="store_true", help="Disable mixed precision")
    return p.parse_args()


# ── Setup helpers ─────────────────────────────────────────────────────────────

def load_config(name: str) -> dict:
    mod = importlib.import_module(f"configs.{name}")
    return dict(mod.CONFIG)   # copy so we don't mutate the module


def apply_overrides(config: dict, args) -> dict:
    if args.lr         is not None: config["lr"]           = args.lr
    if args.epochs     is not None: config["epochs"]       = args.epochs
    if args.batch_size is not None: config["batch_size"]   = args.batch_size
    if args.seed       is not None: config["seed"]         = args.seed
    if args.device     is not None: config["device"]       = args.device
    if args.no_amp:                 config["use_amp"]      = False
    # Default device
    if "device" not in config:
        config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    return config


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    config = load_config(args.config)
    config = apply_overrides(config, args)

    print("=" * 60)
    print(f"  Capstone 2 — Training Framework")
    print(f"  Config  : {args.config}")
    print(f"  Device  : {config['device']}")
    print(f"  AMP     : {config.get('use_amp', True)}")
    print("=" * 60)

    # ── Reproducibility ───────────────────────────────────────────────────────
    set_seed(config.get("seed", 42))

    # ── Data ──────────────────────────────────────────────────────────────────
    print("\n[1/5] Building datasets...")
    train_ds, val_ds = build_datasets(config)
    train_loader, val_loader = build_dataloaders(train_ds, val_ds, config)

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\n[2/5] Building model...")
    model = build_model(config)
    print(f"  Model   : {model.__class__.__name__}")
    print(model.parameter_summary())

    # ── Loss ──────────────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()

    # ── Optimizer + Scheduler ─────────────────────────────────────────────────
    print("\n[3/5] Building optimizer & scheduler...")
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    # ── Logger + Checkpoint ───────────────────────────────────────────────────
    print("\n[4/5] Setting up logger & checkpoint...")
    logger     = Logger(log_dir=config["log_dir"], run_name=config["run_name"])
    checkpoint = Checkpoint(ckpt_dir=config["ckpt_dir"], run_name=config["run_name"])
    evaluator  = Evaluator(device=config["device"])

    # ── Trainer ───────────────────────────────────────────────────────────────
    print("\n[5/5] Building trainer...")
    trainer = Trainer(
        model        = model,
        criterion    = criterion,
        optimizer    = optimizer,
        scheduler    = scheduler,
        train_loader = train_loader,
        val_loader   = val_loader,
        config       = config,
        logger       = logger,
        checkpoint   = checkpoint,
        evaluator    = evaluator,
    )

    # ── Resume ────────────────────────────────────────────────────────────────
    if args.resume or args.resume_path:
        trainer.resume(path=args.resume_path)

    # ── Eval only ─────────────────────────────────────────────────────────────
    if args.eval_only:
        print("\nEval-only mode...")
        metrics = evaluator.evaluate(model, val_loader, criterion)
        print(f"  val loss  : {metrics['loss']:.4f}")
        print(f"  val acc@1 : {metrics['acc1']:.2%}")
        print(f"  val acc@5 : {metrics['acc5']:.2%}")
        return

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer.fit()


if __name__ == "__main__":
    main()
