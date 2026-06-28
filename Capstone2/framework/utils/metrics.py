"""
Lightweight metric trackers used during training and evaluation.

AverageMeter  — running mean of any scalar (loss, time, etc.)
TopKAccuracy  — top-k classification accuracy

Design note: these are stateful objects that you reset() at the start of each
epoch, update() after every batch, and then read .avg / .compute() at the end.
This avoids collecting all values in memory.
"""

import torch


class AverageMeter:
    """
    Tracks a running mean and sum.

    Usage:
        meter = AverageMeter("loss")
        for x, y in loader:
            loss = criterion(model(x), y)
            meter.update(loss.item(), n=x.size(0))   # weight by batch size
        print(meter.avg)
    """

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.sum: float = 0.0
        self.count: int = 0

    def update(self, val: float, n: int = 1):
        """val is the per-sample mean; n is the number of samples in this batch."""
        self.sum += val * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count > 0 else 0.0

    def __repr__(self):
        return f"AverageMeter(name={self.name!r}, avg={self.avg:.4f}, n={self.count})"


class TopKAccuracy:
    """
    Top-k classification accuracy.

    Top-1: correct if argmax(logits) == target             (standard accuracy)
    Top-5: correct if target is among the 5 highest logits (ImageNet convention)

    How it works:
        logits  shape (B, C) — raw scores before softmax
        targets shape (B,)   — integer class labels

        pred = top-k column indices of logits, shape (k, B)
        correct[i, j] = (pred[i, j] == targets[j])
        sample j is correct if ANY of the k rows is true for that column
    """

    def __init__(self, k: int = 1):
        self.k = k
        self.reset()

    def reset(self):
        self.correct: int = 0
        self.total: int = 0

    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        with torch.no_grad():
            batch_size = targets.size(0)
            # topk returns (values, indices); we want indices, shape (B, k)
            _, pred = logits.topk(min(self.k, logits.size(1)), dim=1, largest=True, sorted=True)
            pred = pred.t()                               # (k, B)
            correct = pred.eq(targets.view(1, -1).expand_as(pred))  # (k, B) bool
            self.correct += correct.any(dim=0).sum().item()          # any of k rows
            self.total += batch_size

    def compute(self) -> float:
        """Return accuracy in [0, 1]."""
        return self.correct / self.total if self.total > 0 else 0.0

    def reset_and_compute(self) -> float:
        val = self.compute()
        self.reset()
        return val
