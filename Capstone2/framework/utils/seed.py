"""
Reproducibility helpers.

Why reproducibility is hard in deep learning:
  • Python random, NumPy random, and PyTorch random are THREE separate RNGs.
  • CUDA has its own RNG — separate again.
  • cuDNN autotuner (benchmark=True) picks the fastest algorithm on first run,
    which can vary between runs → non-deterministic results even with seeds set.
  • DataLoader workers have independent RNG state — needs per-worker seeding.

set_seed() handles all of these. deterministic=True trades speed for perfect
reproducibility (useful for debugging; too slow for production training).
"""

import random
import os
import torch
import numpy as np


def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    Seed all RNGs for reproducible training.

    Args:
        seed:          Integer seed value.
        deterministic: If True, force cuDNN into deterministic mode
                       (slower, but byte-for-byte reproducible).
    """
    # Python built-in
    random.seed(seed)
    # NumPy
    np.random.seed(seed)
    # PyTorch CPU
    torch.manual_seed(seed)
    # PyTorch CUDA (all GPUs)
    torch.cuda.manual_seed_all(seed)
    # Env var used by some CUDA ops
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        # cuDNN always picks the same algorithm — reproducible but 10-30% slower
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # PyTorch ≥ 1.8: catch ops that have no deterministic implementation
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        # cuDNN benchmark: first run tests several algorithms, picks the fastest.
        # This is non-deterministic (algorithm choice can vary) but faster.
        torch.backends.cudnn.benchmark = True


def worker_init_fn(worker_id: int) -> None:
    """
    Seed each DataLoader worker differently.

    DataLoader spawns `num_workers` subprocesses. Without this function,
    all workers start with the same RNG state → identical random augmentations
    across workers → less effective data augmentation diversity.

    Pass this to DataLoader as:  DataLoader(..., worker_init_fn=worker_init_fn)
    """
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
