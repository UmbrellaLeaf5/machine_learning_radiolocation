"""Diagnose full-val retrieval errors using the saved first_run checkpoint."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from measure_submission_baseline import load_submission


HW_DIR = Path(__file__).resolve().parent
ROOT = HW_DIR.parents[1]
VAL_DIR = ROOT / "data" / "QXSLAB_SAROPT" / "val"
WEIGHTS = HW_DIR / "results" / "first_run" / "hw1_best.pt"
REPORT_PATH = HW_DIR / "first_run_val_diagnostics.json"
FIGURE_PATH = HW_DIR / "first_run_val_diagnostics.png"


def file_id(path: Path) -> int:
    match = re.search(r"\d+", path.stem)
    if match is None:
        raise ValueError(f"Нет числового идентификатора в имени {path}")
    return int(match.group())


def embed_paths(embed: Any, paths: list[Path]) -> np.ndarray:
    parts = []
    for start in range(0, len(paths), 100):
        images = []
        for path in paths[start : start + 100]:
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise FileNotFoundError(path)
            images.append(image)
        parts.append(embed(images))
    return np.concatenate(parts).astype(np.float64)


def intensity_statistics(paths: list[Path], grayscale: bool) -> tuple[np.ndarray, np.ndarray]:
    means = np.empty(len(paths), dtype=np.float64)
    saturation = np.empty(len(paths), dtype=np.float64)
    for i, path in enumerate(paths):
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(path)
        if grayscale and image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        means[i] = float(image.mean())
        saturation[i] = float((image >= 250).mean())
    return means, saturation


def pixel_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.ndim == 3:
        a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    if b.ndim == 3:
        b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    x, y = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    sar_paths = sorted(VAL_DIR.glob("sar*.png"), key=file_id)
    if len(sar_paths) != 2100:
        raise ValueError(f"Ожидалось 2100 val-пар, найдено {len(sar_paths)}")
    opt_paths = [path.with_name(path.name.replace("sar", "opt", 1)) for path in sar_paths]
    if not all(path.is_file() for path in opt_paths):
        raise FileNotFoundError("В val найдена неполная SAR–Opt пара")

    submission_class = load_submission()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    submission = submission_class(WEIGHTS, device=device)
    start = time.perf_counter()
    sar = embed_paths(submission.embed_sar, sar_paths)
    opt = embed_paths(submission.embed_opt, opt_paths)
    embed_time = time.perf_counter() - start
    distances = np.sqrt(
        np.maximum((sar**2).sum(1)[:, None] + (opt**2).sum(1)[None, :] - 2 * sar @ opt.T, 0)
    )
    ranks = 1 + (distances < np.diag(distances)[:, None]).sum(axis=1)
    ids = np.array([file_id(path) for path in sar_paths])

    sar_mean, sar_saturation = intensity_statistics(sar_paths, grayscale=False)
    opt_mean, _ = intensity_statistics(opt_paths, grayscale=True)
    saturation_groups = np.array_split(np.argsort(sar_saturation), 4)
    brightness_groups = np.array_split(np.argsort(opt_mean), 4)
    id_groups = np.array_split(np.arange(len(ids)), 4)

    near_wrong = np.where(ranks > 1)[0]
    nearest = distances.argmin(axis=1)
    close_id_errors = [i for i in near_wrong if abs(int(ids[i] - ids[nearest[i]])) <= 16]
    corr_values = []
    absdiff_values = []
    for i in close_id_errors:
        true_image = cv2.imread(str(opt_paths[i]), cv2.IMREAD_UNCHANGED)
        wrong_image = cv2.imread(str(opt_paths[nearest[i]]), cv2.IMREAD_UNCHANGED)
        corr_values.append(pixel_corr(true_image, wrong_image))
        gray_true = cv2.cvtColor(true_image, cv2.COLOR_BGR2GRAY)
        gray_wrong = cv2.cvtColor(wrong_image, cv2.COLOR_BGR2GRAY)
        absdiff_values.append(float(np.abs(gray_true.astype(float) - gray_wrong).mean()))

    report: dict[str, Any] = {
        "checkpoint": str(WEIGHTS.relative_to(ROOT)),
        "device": device,
        "gallery_size": 2100,
        "n_pairs": len(sar_paths),
        "embedding_time_s": round(embed_time, 2),
        "rank_distribution": {
            "rank_1": int((ranks == 1).sum()),
            "rank_2_10": int(((ranks >= 2) & (ranks <= 10)).sum()),
            "rank_11_20": int(((ranks >= 11) & (ranks <= 20)).sum()),
            "rank_gt_20": int((ranks > 20).sum()),
        },
        "recall_at_1_5_10": {
            str(k): float((ranks <= k).mean()) for k in (1, 5, 10)
        },
        "mrr": float((1.0 / ranks).mean()),
        "rank_by_sar_saturation_quartile_low_to_high": [
            {
                "mean_saturation_fraction": float(sar_saturation[group].mean()),
                "recall_at_10": float((ranks[group] <= 10).mean()),
                "n": len(group),
            }
            for group in saturation_groups
        ],
        "rank_by_opt_mean_intensity_quartile_dark_to_bright": [
            {
                "mean_intensity": float(opt_mean[group].mean()),
                "recall_at_10": float((ranks[group] <= 10).mean()),
                "n": len(group),
            }
            for group in brightness_groups
        ],
        "rank_by_contiguous_id_quartile": [
            {
                "id_min": int(ids[group].min()),
                "id_max": int(ids[group].max()),
                "recall_at_10": float((ranks[group] <= 10).mean()),
                "n": len(group),
            }
            for group in id_groups
        ],
        "nearest_wrong_by_id_gap_le_16": {
            "n_errors": len(close_id_errors),
            "fraction_of_all_errors": len(close_id_errors) / max(len(near_wrong), 1),
            "mean_opt_pixel_correlation": float(np.mean(corr_values)) if corr_values else None,
            "median_opt_gray_abs_difference": float(np.median(absdiff_values)) if absdiff_values else None,
            "interpretation": "Pixel similarity was measured directly; nearby IDs alone are not treated as proof of overlap.",
        },
        "preprocessing_audit": {
            "sar": "cv2 IMREAD_UNCHANGED grayscale, duplicated to RGB, /255, ImageNet mean/std; no augmentation",
            "opt": "cv2 IMREAD_UNCHANGED BGR converted to RGB, /255, ImageNet mean/std; no augmentation",
            "embedding": "L2 normalized in each independent branch; Euclidean ranking",
            "training_alignment": "same prepare_image function is used for Dataset and is reproduced in standalone Submission",
        },
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    bins = np.arange(1, min(int(ranks.max()), 101) + 2) - 0.5
    axes[0].hist(ranks, bins=bins, color="#31688e")
    axes[0].set(title="Ранги правильной Opt (val, N=2100)", xlabel="Ранг", ylabel="Число SAR")
    axes[0].set_xlim(0.5, min(int(ranks.max()), 100) + 0.5)
    sat_recall = [row["recall_at_10"] for row in report["rank_by_sar_saturation_quartile_low_to_high"]]
    axes[1].plot(range(1, 5), sat_recall, marker="o", color="#d1495b")
    axes[1].set(
        title="Recall@10 по насыщению SAR (val, N=2100)",
        xlabel="Квартиль насыщения: низкий → высокий",
        ylabel="Recall@10",
        xticks=range(1, 5),
        ylim=(0, 1),
    )
    for axis in axes:
        axis.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=160)
    plt.close(fig)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\nДиагностические графики и срезы относятся к val-галерее N=2100; квартильные срезы служат только для локализации ошибок.")


if __name__ == "__main__":
    main()
