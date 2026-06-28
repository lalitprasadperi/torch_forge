"""
Dataset wrappers.

A PyTorch Dataset must implement two methods:
  __len__()           → how many samples total
  __getitem__(index)  → return (input, label) for sample `index`

The DataLoader calls __getitem__ repeatedly (in worker processes) and
collates the results into a batch tensor.

We wrap torchvision datasets so that:
  1. Downloads happen automatically to config['data_dir']
  2. Transforms are injected from our transforms.py
  3. Tiny ImageNet (not in torchvision) is handled with a custom class

Key concept — Dataset vs DataLoader:
  Dataset:    knows WHAT the data is — returns one sample at a time
  DataLoader: knows HOW to serve data — batching, shuffling, parallelism
"""

import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.datasets as tvd
from .transforms import get_transforms


# ── Standard torchvision datasets ────────────────────────────────────────────

def build_datasets(config: dict) -> tuple[Dataset, Dataset]:
    """
    Build (train_dataset, val_dataset) from config.

    config keys used:
      dataset  : "mnist" | "cifar10" | "tiny_imagenet"
      data_dir : path where data is downloaded / stored
    """
    name     = config["dataset"].lower().replace("-", "_")
    data_dir = config.get("data_dir", "./data")
    train_tf, val_tf = get_transforms(name)

    if name == "mnist":
        train_ds = tvd.MNIST(data_dir, train=True,  download=True, transform=train_tf)
        val_ds   = tvd.MNIST(data_dir, train=False, download=True, transform=val_tf)

    elif name == "cifar10":
        train_ds = tvd.CIFAR10(data_dir, train=True,  download=True, transform=train_tf)
        val_ds   = tvd.CIFAR10(data_dir, train=False, download=True, transform=val_tf)

    elif name == "tiny_imagenet":
        root = Path(data_dir) / "tiny-imagenet-200"
        if not root.exists():
            raise FileNotFoundError(
                f"Tiny ImageNet not found at {root}.\n"
                "Download it:\n"
                "  wget http://cs231n.stanford.edu/tiny-imagenet-200.zip\n"
                "  unzip tiny-imagenet-200.zip -d data/"
            )
        train_ds = TinyImageNet(root, split="train", transform=train_tf)
        val_ds   = TinyImageNet(root, split="val",   transform=val_tf)

    else:
        raise ValueError(f"Unknown dataset: {name!r}")

    print(f"  Dataset : {name}")
    print(f"  Train   : {len(train_ds):,} samples")
    print(f"  Val     : {len(val_ds):,} samples")
    return train_ds, val_ds


# ── Tiny ImageNet (custom, not in torchvision) ────────────────────────────────

class TinyImageNet(Dataset):
    """
    Tiny ImageNet dataset.
    Download: http://cs231n.stanford.edu/tiny-imagenet-200.zip

    Directory structure after unzip:
      tiny-imagenet-200/
        train/
          n01443537/
            images/  *.JPEG
        val/
          images/  *.JPEG
          val_annotations.txt   ← maps filename → class label

    This class shows what a custom Dataset looks like end-to-end:
      __init__: discover all file paths and labels
      __len__:  return total count
      __getitem__: load one image, apply transform, return (tensor, label)
    """

    def __init__(self, root: Path, split: str = "train", transform=None):
        self.root      = Path(root)
        self.split     = split
        self.transform = transform

        # Build class → integer index mapping from train directory
        train_dir  = self.root / "train"
        class_dirs = sorted(p.name for p in train_dir.iterdir() if p.is_dir())
        self.class_to_idx = {cls: i for i, cls in enumerate(class_dirs)}

        self.samples: list[tuple[Path, int]] = []

        if split == "train":
            for cls_name, cls_idx in self.class_to_idx.items():
                img_dir = train_dir / cls_name / "images"
                for img_path in sorted(img_dir.glob("*.JPEG")):
                    self.samples.append((img_path, cls_idx))

        elif split == "val":
            ann_file = self.root / "val" / "val_annotations.txt"
            val_img_dir = self.root / "val" / "images"
            with open(ann_file) as f:
                for line in f:
                    parts = line.strip().split("\t")
                    fname, cls_name = parts[0], parts[1]
                    cls_idx = self.class_to_idx[cls_name]
                    self.samples.append((val_img_dir / fname, cls_idx))

        else:
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        # PIL Image → RGB (3 channels; some Tiny-IN images are grayscale)
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label
