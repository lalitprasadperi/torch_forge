"""
Logger — console table output + CSV file logging.

Responsibilities:
  • Print a formatted row after each epoch (loss, accuracy, lr, time)
  • Append the same row to a CSV file for later plotting/analysis
  • Print a header separator on the first epoch

Why log to CSV?
  Console output disappears when the terminal closes. A CSV lets you:
    pandas.read_csv("logs/run.csv").plot(x="epoch", y="val_acc")
  even after training finishes.
"""

import csv
import time
from pathlib import Path


class Logger:
    COLS = [
        ("epoch",    5,  "d"),
        ("phase",    6,  "s"),
        ("loss",     9,  ".4f"),
        ("acc@1",    8,  ".2%"),
        ("acc@5",    8,  ".2%"),
        ("lr",       10, ".2e"),
        ("time(s)",  8,  ".1f"),
    ]

    def __init__(self, log_dir: str = "experiments/logs", run_name: str = "run"):
        self.log_dir  = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.log_dir / f"{run_name}.csv"
        self._csv_file   = None
        self._csv_writer = None
        self._header_printed = False
        self._epoch_start: float = 0.0

    # ── Public API ────────────────────────────────────────────────────────────

    def epoch_start(self):
        self._epoch_start = time.perf_counter()

    def log(
        self,
        epoch:   int,
        phase:   str,    # "train" or "val"
        loss:    float,
        acc1:    float,  # fraction in [0, 1]
        acc5:    float,
        lr:      float,
    ):
        elapsed = time.perf_counter() - self._epoch_start
        row = dict(epoch=epoch, phase=phase, loss=loss,
                   acc1=acc1, acc5=acc5, lr=lr, elapsed=elapsed)
        self._print(row)
        self._write_csv(row)

    def close(self):
        if self._csv_file:
            self._csv_file.close()
            self._csv_file = None

    # ── Internal ──────────────────────────────────────────────────────────────

    def _print(self, row: dict):
        if not self._header_printed:
            self._print_header()
            self._header_printed = True

        vals = [
            row["epoch"],
            row["phase"],
            row["loss"],
            row["acc1"],
            row["acc5"],
            row["lr"],
            row["elapsed"],
        ]
        fmts = [f"{{{i}:{col[2]}}}" for i, col in enumerate(self.COLS)]
        widths = [col[1] for col in self.COLS]
        parts = [f"{v:{w}{f[3:-1]}}" for v, (_, w, f) in zip(vals, self.COLS)]
        print("│ " + " │ ".join(p.center(w) for p, (_, w, __) in zip(parts, self.COLS)) + " │")

    def _print_header(self):
        names   = [col[0] for col in self.COLS]
        widths  = [col[1] for col in self.COLS]
        sep     = "├─" + "─┼─".join("─" * w for w in widths) + "─┤"
        top     = "┌─" + "─┬─".join("─" * w for w in widths) + "─┐"
        header  = "│ " + " │ ".join(n.center(w) for n, w in zip(names, widths)) + " │"
        print(top)
        print(header)
        print(sep)

    def _write_csv(self, row: dict):
        if self._csv_writer is None:
            self._csv_file   = open(self.csv_path, "w", newline="")
            fieldnames = ["epoch", "phase", "loss", "acc1", "acc5", "lr", "elapsed"]
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=fieldnames)
            self._csv_writer.writeheader()
        self._csv_writer.writerow({
            "epoch":   row["epoch"],
            "phase":   row["phase"],
            "loss":    f"{row['loss']:.6f}",
            "acc1":    f"{row['acc1']:.4f}",
            "acc5":    f"{row['acc5']:.4f}",
            "lr":      f"{row['lr']:.2e}",
            "elapsed": f"{row['elapsed']:.1f}",
        })
        self._csv_file.flush()
