"""
DataLoader factory.

torch.utils.data.DataLoader wraps a Dataset and handles:
  • Batching       — collating N samples into one (B, ...) tensor
  • Shuffling      — random order each epoch (train only)
  • Parallelism    — spawning worker processes to load data asynchronously
  • Prefetching    — preparing the next batch while the GPU runs the current one

Key parameters explained
─────────────────────────
num_workers (int):
  Number of subprocesses spawned for data loading.
  0 → single-process (simple, good for debugging)
  4 → 4 workers fetch batches in parallel while GPU trains
  Rule of thumb: num_workers = num_CPU_cores / num_GPUs, capped at 8-16.

pin_memory (bool):
  If True, the DataLoader pre-allocates batches in PINNED CPU memory.
  Pinned memory is page-locked — the GPU DMA engine can copy it directly
  without going through the OS page cache → 2-3× faster H2D transfers.
  Always True when training on GPU.

persistent_workers (bool):
  If True, worker processes are kept alive between epochs instead of being
  respawned. Eliminates ~1s of worker startup overhead per epoch.
  Set True when num_workers > 0.

prefetch_factor (int):
  How many batches each worker prefetches ahead of the main process.
  Default 2: one batch in flight + one ready to deliver.

drop_last (bool):
  If True, drop the last batch if it's smaller than batch_size.
  Keeps batch sizes consistent, which matters for BatchNorm statistics.
"""

from torch.utils.data import DataLoader, Dataset
from ..utils.seed import worker_init_fn


def build_dataloaders(
    train_ds: Dataset,
    val_ds:   Dataset,
    config:   dict,
) -> tuple[DataLoader, DataLoader]:
    """
    Build train and val DataLoaders from config.

    config keys used:
      batch_size       : int, default 256
      num_workers      : int, default 4
      pin_memory       : bool, default True
      prefetch_factor  : int, default 2
    """
    batch_size = config.get("batch_size", 256)
    nw         = config.get("num_workers", 4)
    pin        = config.get("pin_memory", True)
    pf         = config.get("prefetch_factor", 2) if nw > 0 else None
    persistent = nw > 0

    common = dict(
        num_workers      = nw,
        pin_memory       = pin,
        prefetch_factor  = pf,
        persistent_workers = persistent,
        worker_init_fn   = worker_init_fn,   # unique seed per worker
    )

    train_loader = DataLoader(
        train_ds,
        batch_size = batch_size,
        shuffle    = True,        # new random order every epoch
        drop_last  = True,        # avoid tiny last-batch for stable BN stats
        **common,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size = batch_size * 2,  # no grad → can fit 2× batch in memory
        shuffle    = False,           # deterministic val
        drop_last  = False,
        **common,
    )

    print(f"  DataLoader: batch={batch_size}  workers={nw}  "
          f"pin_memory={pin}  prefetch={pf}")
    print(f"  Train batches: {len(train_loader)}   Val batches: {len(val_loader)}")
    return train_loader, val_loader
