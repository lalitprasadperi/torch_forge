# 🔥 Capstone 2 — Build a Complete Training Framework from Scratch

> Train MNIST, CIFAR-10, and Tiny ImageNet with your own Trainer, Dataset, DataLoader,
> Optimizer, Scheduler, Logger, Checkpoint, and Evaluator — built piece by piece so you
> understand every line.

---

## 🗺️ What You'll Build

```
Capstone2/
├── framework/              ← The training library you build
│   ├── models/
│   │   ├── base.py         ← BaseModel: param count, weight init, hooks API
│   │   ├── mlp.py          ← MLP for MNIST
│   │   ├── cnn.py          ← CNN for CIFAR-10
│   │   └── resnet.py       ← MiniResNet-18 for Tiny ImageNet
│   ├── data/
│   │   ├── dataset.py      ← Dataset wrappers + custom TinyImageNet
│   │   ├── transforms.py   ← Augmentation pipelines per dataset
│   │   └── dataloader.py   ← DataLoader factory (num_workers, pin_memory, etc.)
│   ├── optimizers/
│   │   └── factory.py      ← Build SGD / Adam / AdamW from config
│   ├── schedulers/
│   │   └── factory.py      ← Step / Cosine / WarmupCosine schedulers
│   ├── utils/
│   │   ├── metrics.py      ← AverageMeter, TopKAccuracy
│   │   └── seed.py         ← set_seed(), worker_init_fn()
│   ├── logger.py           ← Console table + CSV logging
│   ├── checkpoint.py       ← Save / load full training state
│   ├── evaluator.py        ← Validation loop
│   └── trainer.py          ← AMP + grad accum + grad clip + the full loop
│
├── configs/
│   ├── mnist_mlp.py        ← Config dict for MNIST run
│   ├── cifar10_cnn.py      ← Config dict for CIFAR-10 run
│   └── tiny_imagenet.py    ← Config dict for Tiny ImageNet run
│
├── tours/                  ← Narrated educational scripts
│   ├── module_tour.py      ← nn.Module internals (7 lessons)
│   ├── hooks_demo.py       ← Forward / backward hooks (6 lessons)
│   └── amp_demo.py         ← Mixed precision training (6 lessons)
│
├── experiments/
│   ├── checkpoints/        ← .pt checkpoint files
│   └── logs/               ← CSV training logs
│
├── data/                   ← Auto-downloaded datasets
└── train.py                ← CLI entry point
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install torch torchvision pillow
```

### 2. Run the educational tours first
```bash
# Learn nn.Module internals before touching the training code
python tours/module_tour.py

# Learn hooks (activation/gradient inspection)
python tours/hooks_demo.py

# Learn mixed precision (AMP)
python tours/amp_demo.py
```

### 3. Train on MNIST (downloads automatically, ~170 MB)
```bash
python train.py --config mnist_mlp
```

### 4. Train on CIFAR-10 (downloads automatically, ~163 MB)
```bash
python train.py --config cifar10_cnn
```

### 5. Train on Tiny ImageNet (manual download required, ~248 MB)
```bash
# Download first:
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip -d data/

python train.py --config tiny_imagenet
```

---

## 📟 Sample Training Output

```
════════════════════════════════════════════════════════════
  Capstone 2 — Training Framework
  Config  : mnist_mlp
  Device  : cuda
  AMP     : True
════════════════════════════════════════════════════════════

[1/5] Building datasets...
  Dataset : mnist
  Train   : 60,000 samples
  Val     : 10,000 samples
  DataLoader: batch=256  workers=4  pin_memory=True  prefetch=2
  Train batches: 234   Val batches: 20

[2/5] Building model...
  Model   : MLP
  Trainable parameters :      269,322
  Frozen    parameters :            0
  Total     parameters :      269,322

[3/5] Building optimizer & scheduler...
  Optimizer : AdamW  lr=0.001  wd=0.0001
  Scheduler : cosine  epochs=20

Training on : cuda
AMP enabled : True
Grad accum  : 1 steps
Epochs      : 0 → 20
──────────────────────────────────────────────────────────

┌──────┬────────┬───────────┬──────────┬──────────┬────────────┬──────────┐
│ epoch│  phase │   loss    │  acc@1   │  acc@5   │     lr     │ time(s)  │
├──────┼────────┼───────────┼──────────┼──────────┼────────────┼──────────┤
│    1 │  train │   0.2841  │  91.45%  │  99.98%  │  1.00e-03  │    3.2   │
│    1 │    val │   0.1123  │  96.68%  │  99.97%  │  1.00e-03  │    0.8   │
│    5 │  train │   0.0612  │  98.12%  │  99.99%  │  7.94e-04  │    3.1   │
│    5 │    val │   0.0589  │  98.34%  │  99.99%  │  7.94e-04  │    0.8   │
│   20 │  train │   0.0268  │  99.21%  │ 100.00%  │  1.00e-07  │    3.0   │
│   20 │    val │   0.0301  │  99.14%  │ 100.00%  │  1.00e-07  │    0.8   │

Training complete. Best val acc@1: 99.14%
```

---

## 🧠 Concepts You'll Master

### 🔷 nn.Module Internals
| Concept | What You Learn |
|---------|----------------|
| `nn.Parameter` | Tensors that get gradients and optimizer updates |
| `register_buffer()` | Save state (like BN running stats) without gradients |
| `state_dict()` | Portable weight serialisation |
| `load_state_dict()` | Restoring weights, strict vs non-strict |
| `named_modules()` | Walking the full module tree |
| `apply()` | Recursive weight initialisation |
| `model.train()` / `model.eval()` | BatchNorm and Dropout mode switching |

### 🔷 Hooks
| Hook Type | Usage |
|-----------|-------|
| Forward hook | Capture activations, build feature extractors |
| Backward hook | Monitor gradient flow, detect vanishing gradients |
| Dead ReLU detection | Spot neurons stuck at zero |
| `handle.remove()` | Avoid memory leaks |

### 🔷 Mixed Precision (AMP)
| Concept | What You Learn |
|---------|----------------|
| FP16 dynamic range | Why gradients underflow without scaling |
| `GradScaler` | Loss scaling algorithm (scale → backward → unscale → check) |
| `autocast` | Which ops run in FP16 (matmul, conv) vs FP32 (softmax, loss) |
| Real speedup | 2-4× on Tensor Core GPUs |

### 🔷 Training Loop
| Feature | How It Works |
|---------|-------------|
| Gradient accumulation | `loss / N` over N batches → one `optimizer.step()` |
| Gradient clipping | `clip_grad_norm_()` after `scaler.unscale_()` |
| Best model saving | Compare `val_acc1` to running best, save `.pt` |
| Full checkpoint | model + optimizer + scheduler + scaler + epoch + metrics |

---

## ⚙️ CLI Reference

```bash
# Basic training
python train.py --config mnist_mlp

# Override hyperparameters
python train.py --config cifar10_cnn --lr 0.01 --epochs 30 --batch-size 128

# Resume from latest checkpoint
python train.py --config cifar10_cnn --resume

# Resume from specific checkpoint
python train.py --config cifar10_cnn --resume-path experiments/checkpoints/cifar10_cnn_best.pt

# Evaluate only (no training)
python train.py --config mnist_mlp --eval-only --resume

# Disable AMP (for debugging)
python train.py --config mnist_mlp --no-amp

# Override device
python train.py --config mnist_mlp --device cpu
```

---

## 🏋️ Target Performance

| Config | Dataset | Model | Epochs | Target Val Acc@1 |
|--------|---------|-------|--------|-----------------|
| `mnist_mlp` | MNIST | MLP 784→256→128→10 | 20 | ~99.1% |
| `cifar10_cnn` | CIFAR-10 | 3×ConvBlock + GAP | 50 | ~87% |
| `tiny_imagenet` | Tiny ImageNet | MiniResNet-18 | 90 | ~55% |

---

## 📐 Architecture Diagrams

### MLP (MNIST)
```
Input (B, 1, 28, 28)
      │
   Flatten ──────────────────────→ (B, 784)
      │
   Linear(784→256) + BN + ReLU + Dropout(0.2)
      │
   Linear(256→128) + BN + ReLU + Dropout(0.2)
      │
   Linear(128→10)
      │
   Logits (B, 10)
```

### CNN (CIFAR-10)
```
Input (B, 3, 32, 32)
      │
   Conv(3→32,3×3)+BN+ReLU+MaxPool ──→ (B, 32, 16, 16)
      │
   Conv(32→64,3×3)+BN+ReLU+MaxPool ──→ (B, 64, 8, 8)
      │
   Conv(64→128,3×3)+BN+ReLU+MaxPool ──→ (B, 128, 4, 4)
      │
   AdaptiveAvgPool(1) ──────────────→ (B, 128, 1, 1)
      │
   Flatten + Linear(128→10)
      │
   Logits (B, 10)
```

### MiniResNet (Tiny ImageNet)
```
Input (B, 3, 64, 64)
      │
   Stem: Conv(3→64, 3×3, s=1) + BN + ReLU ──→ (B, 64, 64, 64)
      │
   Stage 1: 2×BasicBlock(64→64,  s=1) ──────→ (B, 64, 64, 64)
      │
   Stage 2: 2×BasicBlock(64→128,  s=2) ─────→ (B, 128, 32, 32)
      │
   Stage 3: 2×BasicBlock(128→256, s=2) ─────→ (B, 256, 16, 16)
      │
   Stage 4: 2×BasicBlock(256→512, s=2) ─────→ (B, 512, 8, 8)
      │
   AdaptiveAvgPool(1) ────────────────────────→ (B, 512)
      │
   Linear(512→200) → Logits
```

### BasicBlock (Residual Connection)
```
x ──────────────────────────────→  shortcut (identity or 1×1 proj)
│                                                   │
├──→ Conv(3×3,s) → BN → ReLU → Conv(3×3) → BN ──→ (+) → ReLU → out
```

---

## 📚 Checkpoint Format

Every checkpoint at `experiments/checkpoints/<run_name>_epoch<N>.pt` stores:

```python
{
    "epoch":     int,            # which epoch this is
    "model":     OrderedDict,    # model.state_dict()
    "optimizer": dict,           # optimizer.state_dict()
    "scheduler": dict,           # scheduler.state_dict()
    "scaler":    dict | None,    # GradScaler.state_dict()
    "metrics":   {               # validation results this epoch
        "val_acc1": float,
        "val_acc5": float,
        "val_loss": float,
    },
    "config":    dict,           # full config for reproducibility
}
```

---

## 📈 Plot Training Curves

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("experiments/logs/mnist_mlp.csv")
train = df[df["phase"] == "train"]
val   = df[df["phase"] == "val"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(train["epoch"], train["loss"],  label="train")
ax1.plot(val["epoch"],   val["loss"],    label="val")
ax1.set_title("Loss"); ax1.legend()

ax2.plot(train["epoch"], train["acc1"].astype(float), label="train")
ax2.plot(val["epoch"],   val["acc1"].astype(float),   label="val")
ax2.set_title("Top-1 Accuracy"); ax2.legend()
plt.tight_layout(); plt.show()
```

---

## ✅ Learning Checklist

After completing Capstone 2, you should be able to explain:

- [ ] Why `nn.Parameter` has gradients but `register_buffer` does not
- [ ] What `state_dict()` contains and why we save it instead of the model object
- [ ] What `model.eval()` changes (BatchNorm and Dropout behaviour)
- [ ] How `autocast` decides which ops run in FP16 vs FP32
- [ ] Why GradScaler multiplies the loss before backward
- [ ] Why gradient accumulation requires `loss / N`
- [ ] Why we `unscale_` before `clip_grad_norm_`
- [ ] What a residual connection does for gradient flow
- [ ] Why we use AdaptiveAvgPool instead of Flatten → Linear for CNNs
- [ ] What `worker_init_fn` does and why all workers need different seeds

---

## 🔭 What's Next — Capstone 3

> Distributed Training & Profiling
> - `torch.distributed` / DDP (DistributedDataParallel)
> - NCCL all-reduce — how gradients sync across GPUs
> - `torch.profiler` deep dive
> - Kernel fusion with `torch.compile`
