"""Plot training and validation curves for the SAR train-stat normalization run."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


HW_DIR = Path(__file__).resolve().parent
RUN_DIR = HW_DIR / "results" / "sar_train_norm"
OUTPUT = HW_DIR / "sar_train_norm_history.png"


def main() -> None:
    with (RUN_DIR / "history.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    epochs = [int(row["epoch"]) for row in rows]
    train_loss = [float(row["train_loss"]) for row in rows]
    val_loss = [float(row["val_loss"]) for row in rows]
    val_recall1 = [float(row["val_recall1"]) for row in rows]
    best_epoch = epochs[max(range(len(rows)), key=lambda i: val_recall1[i])]

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, train_loss, label="train loss")
    axes[0].plot(epochs, val_loss, label="val loss")
    axes[0].set(title="SAR train normalization: loss", xlabel="Эпоха", ylabel="Triplet loss")
    axes[0].legend()
    axes[1].plot(epochs, val_recall1, marker="o", markersize=3, label="val Recall@1, N=2100")
    axes[1].axvline(best_epoch, linestyle="--", color="gray", label=f"лучшая эпоха {best_epoch}")
    axes[1].set(title="Полная val-галерея N=2100", xlabel="Эпоха", ylabel="Recall@1", ylim=(0, 1))
    axes[1].legend()
    for axis in axes:
        axis.grid(True, linestyle="--", alpha=0.4)
        axis.set_xticks(list(range(1, len(epochs) + 1, 2)))
    figure.tight_layout()
    figure.savefig(OUTPUT, dpi=160)
    plt.close(figure)
    print(f"Сохранён график: {OUTPUT}; лучшая val Recall@1 на N=2100 — {max(val_recall1):.6f} (эпоха {best_epoch}).")


if __name__ == "__main__":
    main()
