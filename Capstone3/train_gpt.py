"""
train_gpt.py — Train a GPT on tiny Shakespeare

Usage:
  python train_gpt.py --config gpt_nano          # classic GPT (LayerNorm + GELU + learned PE)
  python train_gpt.py --config gpt_modern        # LLaMA-style (RMSNorm + SwiGLU + RoPE)
  python train_gpt.py --config gpt_nano --resume # continue from latest checkpoint
  python train_gpt.py --config gpt_nano --generate-only --prompt "To be or not"

The training loop here is step-based (not epoch-based) because language model
training is commonly measured in "steps" (gradient updates) rather than passes
over a fixed-size dataset. We sample random batches from the dataset each step.
"""

import sys
import argparse
import importlib
import time
import math
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

sys.path.insert(0, str(Path(__file__).parent))

from transformer.blocks.transformer import GPT
from transformer.generate import generate
from transformer.kv_cache import KVCache
from data import load_shakespeare, CharTokenizer


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",    default="gpt_nano",
                   choices=["gpt_nano", "gpt_modern"])
    p.add_argument("--resume",    action="store_true")
    p.add_argument("--generate-only", action="store_true")
    p.add_argument("--prompt",    default="HAMLET:\n", type=str)
    p.add_argument("--device",    default=None)
    return p.parse_args()


# ── Training utilities ────────────────────────────────────────────────────────

def get_lr(step: int, lr: float, warmup_steps: int, total_steps: int) -> float:
    """Cosine decay with linear warmup."""
    if step < warmup_steps:
        return lr * step / max(1, warmup_steps)
    if step >= total_steps:
        return lr * 0.1   # minimum LR = 10% of peak
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return lr * 0.1 + 0.5 * lr * 0.9 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def estimate_loss(model, train_data, val_data, config, device, eval_iters=50):
    """Estimate train and val loss by averaging over eval_iters random batches."""
    model.eval()
    losses = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        L = []
        for _ in range(eval_iters):
            x, y = get_batch(data, config["batch_size"], config["max_len"], device)
            _, loss = model(x, targets=y)
            L.append(loss.item())
        losses[split] = sum(L) / len(L)
    model.train()
    return losses


def get_batch(
    dataset, batch_size: int, max_len: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a random batch of (input, target) from the text dataset."""
    import random
    tokens = dataset.tokens
    # Random starting positions
    ix = torch.randint(len(tokens) - max_len - 1, (batch_size,))
    x  = torch.stack([tokens[i    : i + max_len    ] for i in ix])
    y  = torch.stack([tokens[i + 1: i + max_len + 1] for i in ix])
    return x.to(device), y.to(device)


def generate_sample(model, tokenizer, prompt: str, config: dict, device: str) -> str:
    prompt_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0)
    out = generate(
        model,
        prompt_ids,
        max_new_tokens  = config["gen_max_tokens"],
        temperature     = config["gen_temperature"],
        top_k           = config["gen_top_k"],
        eos_token_id    = None,
    )
    return tokenizer.decode(out[0].tolist())


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    cfg_mod = importlib.import_module(f"configs.{args.config}")
    config  = dict(cfg_mod.CONFIG)
    model_cfg = config["model"]

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print(f"  Capstone 3 — GPT from Scratch")
    print(f"  Config   : {args.config}")
    print(f"  Device   : {device}")
    print(f"  AMP      : {config.get('use_amp', True)}")
    print("=" * 60)

    # ── Seed ──────────────────────────────────────────────────────────────────
    torch.manual_seed(config.get("seed", 42))

    # ── Data ──────────────────────────────────────────────────────────────────
    print("\n[1/4] Loading data...")
    train_ds, val_ds, tokenizer = load_shakespeare(
        config["data_dir"], max_len=config["max_len"]
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\n[2/4] Building model...")
    model = GPT(model_cfg).to(device)
    print(f"  Architecture : {model_cfg.n_layers}L × {model_cfg.n_heads}H "
          f"× {model_cfg.d_model}D")
    print(f"  RoPE={model_cfg.use_rope}  RMSNorm={model_cfg.use_rmsnorm}  "
          f"SwiGLU={model_cfg.use_swiglu}  Flash={model_cfg.use_flash}")
    print(model.parameter_summary())

    # ── Generate-only mode ────────────────────────────────────────────────────
    if args.generate_only:
        ckpt_path = Path(config["ckpt_dir"]) / f"{config['run_name']}_best.pt"
        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(state["model"])
            print(f"  Loaded {ckpt_path.name}")
        text = generate_sample(model, tokenizer, args.prompt, config, device)
        print(f"\n{'─'*50}\n{text}\n{'─'*50}")
        return

    # ── Optimizer ─────────────────────────────────────────────────────────────
    print("\n[3/4] Building optimizer...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = config["lr"],
        weight_decay = config["weight_decay"],
        betas        = (0.9, 0.95),
        fused        = (device == "cuda"),   # faster fused AdamW on CUDA
    )
    scaler = GradScaler(enabled=(device == "cuda" and config.get("use_amp", True)))

    # ── Resume ────────────────────────────────────────────────────────────────
    start_step = 0
    best_val   = float("inf")
    ckpt_dir   = Path(config["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if args.resume:
        ckpts = sorted(ckpt_dir.glob(f"{config['run_name']}_step*.pt"))
        if ckpts:
            state = torch.load(ckpts[-1], map_location=device, weights_only=False)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            start_step = state["step"] + 1
            best_val   = state.get("best_val", float("inf"))
            print(f"  Resumed from {ckpts[-1].name} at step {start_step}")

    # ── Training loop ─────────────────────────────────────────────────────────
    print("\n[4/4] Training...")
    log_dir = Path(config["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = log_dir / f"{config['run_name']}.csv"
    csv_file  = open(csv_path, "a", newline="")
    csv_writer = csv.writer(csv_file)
    if start_step == 0:
        csv_writer.writerow(["step", "train_loss", "val_loss", "lr", "time_ms"])

    total_steps  = config["epochs"]
    model.train()
    t_step = time.perf_counter()

    for step in range(start_step, total_steps):
        # LR schedule
        lr = get_lr(step, config["lr"], config["warmup_steps"], total_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Forward + backward
        x, y = get_batch(train_ds, config["batch_size"], config["max_len"], device)
        with autocast(device_type=device, enabled=(device == "cuda" and config.get("use_amp", True))):
            _, loss = model(x, targets=y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        # Logging
        if step % config["log_interval"] == 0:
            elapsed = (time.perf_counter() - t_step) * 1000 / config["log_interval"]
            print(f"  step {step:5d}/{total_steps}  "
                  f"loss={loss.item():.4f}  lr={lr:.2e}  {elapsed:.0f}ms/step")
            t_step = time.perf_counter()

        # Evaluation
        if step % config["eval_interval"] == 0:
            losses = estimate_loss(model, train_ds, val_ds, config, device,
                                   config["eval_iters"])
            print(f"  ── eval step={step}  "
                  f"train={losses['train']:.4f}  val={losses['val']:.4f}")
            csv_writer.writerow([step, f"{losses['train']:.4f}",
                                  f"{losses['val']:.4f}", f"{lr:.2e}",
                                  f"{elapsed:.0f}"])
            csv_file.flush()

            # Save best
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save({
                    "step": step, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_val": best_val, "config": config,
                }, ckpt_dir / f"{config['run_name']}_best.pt")
                print(f"  ── best model saved (val={best_val:.4f})")

        # Periodic checkpoint
        if step % 1000 == 0 and step > 0:
            torch.save({
                "step": step, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val": best_val, "config": config,
            }, ckpt_dir / f"{config['run_name']}_step{step:06d}.pt")

        # Generate sample
        if step % config["gen_interval"] == 0 and step > 0:
            sample = generate_sample(model, tokenizer, args.prompt, config, device)
            print(f"\n  ── sample (step {step}) ──\n{sample[:300]}\n  ──────────")
            model.train()

    csv_file.close()
    print(f"\nDone. Best val loss: {best_val:.4f}")
    print(f"Generate text with: python train_gpt.py --config {args.config} --generate-only")


if __name__ == "__main__":
    main()
