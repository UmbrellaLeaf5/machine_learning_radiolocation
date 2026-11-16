"""Evaluate all available candidates through the standalone Submission interface."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np

from measure_submission_baseline import load_submission


HW_DIR = Path(__file__).resolve().parent
ROOT = HW_DIR.parents[1]
VAL_DIR = ROOT / "data" / "QXSLAB_SAROPT" / "val"
BASELINE_JSON = HW_DIR / "first_run_metrics.json"
OUT_JSON = HW_DIR / "model_comparison_metrics.json"
OUT_PNG = HW_DIR / "model_comparison.png"
CHECKPOINTS = {
    "first_run_dim256": HW_DIR / "results" / "first_run" / "submission.pt",
    "dim128_saved": HW_DIR / "results" / "dim128" / "submission.pt",
    "dim128_notebook_rerun": HW_DIR / "results" / "dim128" / "submission_notebook_rerun.pt",
    "sar_train_norm": HW_DIR / "results" / "sar_train_norm" / "hw1_best.pt",
}


def file_id(path: Path) -> int:
    match = re.search(r"\d+", path.stem)
    if match is None:
        raise ValueError(path)
    return int(match.group())


def distances_and_metrics(
    sar: np.ndarray, opt: np.ndarray
) -> tuple[np.ndarray, dict[str, float | int], np.ndarray]:
    distances = np.sqrt(
        np.maximum((sar**2).sum(1)[:, None] + (opt**2).sum(1)[None, :] - 2 * sar @ opt.T, 0)
    )
    ranks = 1 + (distances < np.diag(distances)[:, None]).sum(axis=1)
    positives = np.sort(np.diag(distances))
    threshold = positives[int(np.ceil(0.95 * len(positives))) - 1]
    negatives = distances[~np.eye(len(distances), dtype=bool)]
    metrics: dict[str, float | int] = {
        "Recall@1": float((ranks <= 1).mean()),
        "Recall@5": float((ranks <= 5).mean()),
        "Recall@10": float((ranks <= 10).mean()),
        "MRR": float((1.0 / ranks).mean()),
        "FPR95": float((negatives <= threshold).mean()),
        "gallery_size": int(len(ranks)),
    }
    return distances, metrics, ranks


def read_batch(paths: list[Path]) -> list[np.ndarray]:
    images = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(path)
        images.append(image)
    return images


def main() -> None:
    sar_paths = sorted(VAL_DIR.glob("sar*.png"), key=file_id)
    if len(sar_paths) != 2100:
        raise ValueError(f"Expected full val set of 2100 pairs, got {len(sar_paths)}")
    opt_paths = [path.with_name(path.name.replace("sar", "opt", 1)) for path in sar_paths]
    if not all(path.is_file() for path in opt_paths):
        raise FileNotFoundError("Missing optical validation pair")

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    submission_class = load_submission()
    report: dict[str, Any] = {
        "device": device,
        "selection_rule": "highest Recall@1 on full val gallery N=2100; tie-break by MRR, then Recall@10",
        "evaluations": {},
    }

    for name, checkpoint in CHECKPOINTS.items():
        submission = submission_class(checkpoint, device=device)
        start = time.perf_counter()
        # Preserve the same batch/file order for every model and mirror the official reader.
        sar_parts = []
        opt_parts = []
        for offset in range(0, len(sar_paths), 100):
            sar_parts.append(submission.embed_sar(read_batch(sar_paths[offset : offset + 100])))
            opt_parts.append(submission.embed_opt(read_batch(opt_paths[offset : offset + 100])))
        sar_emb = np.concatenate(sar_parts).astype(np.float64)
        opt_emb = np.concatenate(opt_parts).astype(np.float64)
        embedding_time = time.perf_counter() - start
        if sar_emb.shape != (2100, submission.dim) or opt_emb.shape != (2100, submission.dim):
            raise ValueError(f"Bad embedding shape for {name}: {sar_emb.shape}, {opt_emb.shape}")
        if not np.isfinite(sar_emb).all() or not np.isfinite(opt_emb).all():
            raise ValueError(f"Non-finite embeddings for {name}")

        _, full_metrics, full_ranks = distances_and_metrics(sar_emb, opt_emb)
        _, first_1000_metrics, _ = distances_and_metrics(sar_emb[:1000], opt_emb[:1000])
        id_groups = np.array_split(np.arange(2100), 4)
        sample = read_batch(sar_paths[:4])
        batch = submission.embed_sar(sample)
        single = submission.embed_sar(sample[:1])
        batch_independent = bool(np.abs(batch[:1] - single).max() < 1e-3)
        report["evaluations"][name] = {
            "checkpoint": str(checkpoint.relative_to(HW_DIR)),
            "dim": submission.dim,
            "metric": submission.metric,
            "validation_n2100": full_metrics,
            "validation_query_id_quartiles_full_gallery_n2100": [
                {
                    "id_min": file_id(sar_paths[int(indices[0])]),
                    "id_max": file_id(sar_paths[int(indices[-1])]),
                    "n_queries": len(indices),
                    "Recall@1": float((full_ranks[indices] <= 1).mean()),
                    "Recall@10": float((full_ranks[indices] <= 10).mean()),
                }
                for indices in id_groups
            ],
            "validation_n1000": {
                **first_1000_metrics,
                "n_pairs": 1000,
                "batch_independent": batch_independent,
            },
            "embedding_time_s": round(embedding_time, 2),
        }

    saved_baseline = json.loads(BASELINE_JSON.read_text(encoding="utf-8"))
    measured_first_run = report["evaluations"]["first_run_dim256"]["validation_n1000"]["Recall@10"]
    recorded_first_run = saved_baseline["validation_n1000_submission"]["results"]["first_run"]["Recall@10"]
    report["baseline_json_cross_check"] = {
        "gallery_size": 1000,
        "recorded_first_run_Recall@10": recorded_first_run,
        "remeasured_first_run_Recall@10": measured_first_run,
        "absolute_difference": abs(measured_first_run - recorded_first_run),
    }
    selected = max(
        report["evaluations"],
        key=lambda key: (
            report["evaluations"][key]["validation_n2100"]["Recall@1"],
            report["evaluations"][key]["validation_n2100"]["MRR"],
            report["evaluations"][key]["validation_n2100"]["Recall@10"],
        ),
    )
    report["selected_model"] = selected
    report["selected_weights"] = str(CHECKPOINTS[selected].relative_to(HW_DIR))

    names = list(report["evaluations"])
    labels = ["first_run\ndim=256", "dim=128\nsaved", "dim=128\nnotebook run", "SAR train\nnorm"]
    x = np.arange(len(names))
    width = 0.23
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    for i, metric in enumerate(("Recall@1", "Recall@5", "Recall@10")):
        values = [report["evaluations"][name]["validation_n2100"][metric] for name in names]
        axes[0].bar(x + (i - 1) * width, values, width, label=metric)
    axes[0].set(
        title="Полная val-галерея N=2100",
        ylabel="Recall",
        xticks=x,
        xticklabels=labels,
        ylim=(0, 1),
    )
    axes[0].legend()
    vals1000 = [report["evaluations"][name]["validation_n1000"]["Recall@10"] for name in names]
    colors = ("#31688e", "#35b779", "#f0a202", "#d1495b")
    axes[1].bar(x, vals1000, color=colors[: len(x)])
    axes[1].axhline(0.8, linestyle="--", color="black", label="цель закрытого теста 0,8")
    axes[1].set(
        title="Первые 1000 val-пар, галерея N=1000",
        ylabel="Recall@10",
        xticks=x,
        xticklabels=labels,
        ylim=(0, 1),
    )
    axes[1].legend()
    for name, label in zip(names, labels, strict=True):
        recall_by_range = [
            segment["Recall@10"]
            for segment in report["evaluations"][name]["validation_query_id_quartiles_full_gallery_n2100"]
        ]
        axes[2].plot(range(1, 5), recall_by_range, marker="o", label=label.replace("\n", " "))
    axes[2].set(
        title="Recall@10 по диапазонам ID (N=2100)",
        xlabel="Четверть ID по возрастанию",
        ylabel="Recall@10",
        xticks=range(1, 5),
        ylim=(0, 1),
    )
    axes[2].legend(fontsize=8)
    for axis in axes:
        axis.grid(axis="y", linestyle="--", alpha=0.35)
        axis.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=160)
    plt.close(fig)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
