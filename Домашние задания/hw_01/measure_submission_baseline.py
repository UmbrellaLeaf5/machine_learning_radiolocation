"""Measure comparable Submission Recall@10 baselines on the first 1,000 val pairs."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
HW_DIR = Path(__file__).resolve().parent
NOTEBOOK_PATH = HW_DIR / "HW1.ipynb"
METRICS_PATH = HW_DIR / "first_run_metrics.json"
VAL_DIR = ROOT / "data" / "QXSLAB_SAROPT" / "val"
RUNS = {
    "first_run": HW_DIR / "results" / "first_run" / "hw1_best.pt",
    "dim128": HW_DIR / "results" / "dim128" / "hw1_best.pt",
}


def numeric_id(path: Path) -> int:
    match = re.search(r"\d+", path.stem)
    if match is None:
        raise ValueError(f"Нет числового идентификатора в имени {path}")
    return int(match.group())


def load_submission() -> type:
    notebook: dict[str, Any] = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    source = next(
        "".join(cell.get("source", []))
        for cell in cells
        if cell["cell_type"] == "code"
        and "".join(cell.get("source", [])).startswith("# === SUBMISSION ===")
    )
    namespace: dict[str, Any] = {}
    exec(compile(source, str(NOTEBOOK_PATH), "exec"), namespace)
    return namespace["Submission"]


def embed_all(embed_fn: Any, paths: list[Path], batch_size: int = 100) -> np.ndarray:
    chunks = []
    for start in range(0, len(paths), batch_size):
        images = []
        for path in paths[start : start + batch_size]:
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise FileNotFoundError(path)
            images.append(image)
        output = np.asarray(embed_fn(images), dtype=np.float64)
        expected = min(batch_size, len(paths) - start)
        if output.ndim != 2 or output.shape[0] != expected:
            raise ValueError(f"Ожидался массив ({expected}, D), получен {output.shape}")
        chunks.append(output)
    embeddings = np.concatenate(chunks)
    if not np.isfinite(embeddings).all():
        raise ValueError("В эмбеддингах есть NaN или inf")
    return embeddings


def measure(submission_class: type, name: str, weights: Path, paths: list[Path], device: str) -> dict[str, Any]:
    submission = submission_class(weights, device=device)
    start = time.perf_counter()
    sar = embed_all(submission.embed_sar, paths)
    opt = embed_all(submission.embed_opt, [path.with_name(path.name.replace("sar", "opt", 1)) for path in paths])
    elapsed = time.perf_counter() - start

    small = embed_all(submission.embed_sar, paths[:4], batch_size=4)
    single = embed_all(submission.embed_sar, paths[:1], batch_size=1)
    scale = np.abs(sar[:4]).max() + 1e-12
    batch_independent = bool(
        np.abs(small - sar[:4]).max() / scale < 1e-3
        and np.abs(single - sar[:1]).max() / scale < 1e-3
    )

    if submission.metric == "euclidean":
        squared = (sar**2).sum(1)[:, None] + (opt**2).sum(1)[None, :] - 2 * sar @ opt.T
        distances = np.sqrt(np.maximum(squared, 0))
    elif submission.metric == "cosine":
        sar = sar / (np.linalg.norm(sar, axis=1, keepdims=True) + 1e-12)
        opt = opt / (np.linalg.norm(opt, axis=1, keepdims=True) + 1e-12)
        distances = 1 - sar @ opt.T
    else:
        raise ValueError(f"Неподдерживаемая метрика: {submission.metric}")

    ranks = 1 + (distances < np.diag(distances)[:, None]).sum(axis=1)
    return {
        "run_name": name,
        "n_pairs": len(paths),
        "gallery_size": int(len(paths)),
        "dim": int(sar.shape[1]),
        "metric": submission.metric,
        "Recall@10": float((ranks <= 10).mean()),
        "embedding_time_s": round(elapsed, 2),
        "batch_independent": batch_independent,
    }


def main() -> None:
    if not torch.cuda.is_available():
        device = "cpu"
    else:
        device = "cuda"
    sar_paths = sorted(VAL_DIR.glob("sar*.png"), key=numeric_id)[:1000]
    if len(sar_paths) != 1000:
        raise ValueError(f"Ожидалось 1000 val-пар, найдено {len(sar_paths)}")
    for path in sar_paths:
        opt_path = path.with_name(path.name.replace("sar", "opt", 1))
        if not opt_path.is_file():
            raise FileNotFoundError(opt_path)

    submission_class = load_submission()
    reports = {
        name: measure(submission_class, name, weights, sar_paths, device)
        for name, weights in RUNS.items()
    }
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    metrics["validation_n1000_submission"] = {
        "evaluation": "first 1000 validation pairs; identical ordering and Submission interface",
        "device": device,
        "results": reports,
    }
    METRICS_PATH.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics["validation_n1000_submission"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
