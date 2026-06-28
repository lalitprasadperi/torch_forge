"""
Transform pipelines for each dataset.

A transform is applied to each sample inside Dataset.__getitem__ before it
is returned to the DataLoader. They run in worker processes when num_workers>0.

Why normalise?
  Pixels from ToTensor are in [0,1]. Deep networks converge faster when inputs
  are zero-centred with unit variance. The mean/std values below are the
  per-channel statistics computed over each dataset's full training set.

Why augment at train time only?
  Augmentation is random noise that helps generalisation, but makes validation
  metrics non-comparable across epochs. Val/test always uses the clean pipeline.
"""

import torch
import torchvision.transforms as T


# ── MNIST ─────────────────────────────────────────────────────────────────────
# 1-channel grayscale, 28×28

_MNIST_MEAN = (0.1307,)
_MNIST_STD  = (0.3081,)

MNIST_TRAIN = T.Compose([
    T.ToTensor(),                                    # PIL → (C,H,W) float [0,1]
    T.Normalize(_MNIST_MEAN, _MNIST_STD),
    T.RandomErasing(p=0.1),                          # erase a small patch
])

MNIST_VAL = T.Compose([
    T.ToTensor(),
    T.Normalize(_MNIST_MEAN, _MNIST_STD),
])


# ── CIFAR-10 ──────────────────────────────────────────────────────────────────
# 3-channel RGB, 32×32

_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

CIFAR10_TRAIN = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomCrop(32, padding=4),      # pad 4 each side, random crop back to 32
    T.ToTensor(),
    T.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
])

CIFAR10_VAL = T.Compose([
    T.ToTensor(),
    T.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
])


# ── Tiny ImageNet ─────────────────────────────────────────────────────────────
# 3-channel RGB, 64×64, 200 classes

_TINY_IN_MEAN = (0.4802, 0.4481, 0.3975)
_TINY_IN_STD  = (0.2770, 0.2691, 0.2821)

TINY_IN_TRAIN = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomCrop(64, padding=8),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    T.ToTensor(),
    T.Normalize(_TINY_IN_MEAN, _TINY_IN_STD),
])

TINY_IN_VAL = T.Compose([
    T.ToTensor(),
    T.Normalize(_TINY_IN_MEAN, _TINY_IN_STD),
])


def get_transforms(dataset_name: str) -> tuple:
    """Return (train_transform, val_transform) for a given dataset name."""
    name = dataset_name.lower().replace("-", "_")
    mapping = {
        "mnist":         (MNIST_TRAIN,    MNIST_VAL),
        "cifar10":       (CIFAR10_TRAIN,  CIFAR10_VAL),
        "tiny_imagenet": (TINY_IN_TRAIN,  TINY_IN_VAL),
    }
    if name not in mapping:
        raise ValueError(f"Unknown dataset: {name!r}. Choices: {list(mapping)}")
    return mapping[name]
