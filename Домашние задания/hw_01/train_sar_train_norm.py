"""Fine-tune first_run with train-only global SAR normalization as one controlled change."""

from __future__ import annotations

import csv
import argparse
import json
import random
import re
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from measure_submission_baseline import load_submission


HW_DIR = Path(__file__).resolve().parent
ROOT = HW_DIR.parents[1]
DATA = ROOT / "data" / "QXSLAB_SAROPT"
OUT = HW_DIR / "results" / "sar_train_norm"
INITIAL_WEIGHTS = HW_DIR / "results" / "first_run" / "hw1_best.pt"
SEED = 2026
BATCH_SIZE = 64
EPOCHS = 30
PATIENCE = 7
MARGIN = 0.2
LR = 1e-4
OPT_PREPROCESSING = {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}


def file_id(path: Path) -> int:
    match = re.search(r"\d+", path.stem)
    if match is None:
        raise ValueError(path)
    return int(match.group())


def train_sar_statistics(paths: list[Path]) -> tuple[float, float, int]:
    count = 0
    total = 0.0
    squared_total = 0.0
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None or image.ndim != 2:
            raise ValueError(f"Expected grayscale SAR image: {path}")
        values = image.astype(np.float64) / 255.0
        total += float(values.sum())
        squared_total += float(np.square(values).sum())
        count += values.size
    mean = total / count
    std = max((squared_total / count - mean**2) ** 0.5, 1e-8)
    return mean, std, count


class SarOptPairs(Dataset):
    def __init__(self, ids: list[int], train: bool, mean: float, std: float) -> None:
        self.ids = ids
        self.train = train
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_id = self.ids[index]
        sar_path = DATA / ("train" if image_id <= 16868 else "val") / f"sar{image_id}.png"
        opt_path = DATA / ("train" if image_id <= 16868 else "val") / f"opt{image_id}.png"
        sar = cv2.imread(str(sar_path), cv2.IMREAD_UNCHANGED)
        opt = cv2.imread(str(opt_path), cv2.IMREAD_UNCHANGED)
        if sar is None or opt is None:
            raise FileNotFoundError(f"Missing pair {image_id}")
        if self.train:
            if random.random() < 0.5:
                sar, opt = np.fliplr(sar), np.fliplr(opt)
            k = random.randrange(4)
            sar, opt = np.rot90(sar, k), np.rot90(opt, k)
        sar = np.repeat(sar[..., None], 3, axis=2)
        sar_tensor = torch.from_numpy(np.ascontiguousarray(sar.transpose(2, 0, 1))).float() / 255.0
        sar_tensor = (sar_tensor - self.mean) / self.std
        opt_tensor = PREPARE_IMAGE(opt, OPT_PREPROCESSING)
        return sar_tensor, opt_tensor


def triplet_loss(sar: torch.Tensor, opt: torch.Tensor) -> torch.Tensor:
    distances = torch.cdist(sar.float(), opt.float())
    positive = distances.diagonal()
    negatives = distances.masked_fill(
        torch.eye(len(distances), device=distances.device, dtype=torch.bool), float("inf")
    )
    return (
        torch.relu(positive - negatives.min(dim=1).values + MARGIN).mean()
        + torch.relu(positive - negatives.min(dim=0).values + MARGIN).mean()
    ) / 2


@torch.no_grad()
def embed_loader(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    sar_parts, opt_parts = [], []
    for sar, opt in loader:
        sar_parts.append(model.embed_sar(sar.to(device)).cpu())
        opt_parts.append(model.embed_opt(opt.to(device)).cpu())
    return torch.cat(sar_parts).numpy(), torch.cat(opt_parts).numpy()


def retrieval_metrics(sar: np.ndarray, opt: np.ndarray) -> dict[str, float | int]:
    distances = np.sqrt(
        np.maximum((sar**2).sum(1)[:, None] + (opt**2).sum(1)[None, :] - 2 * sar @ opt.T, 0)
    )
    ranks = 1 + (distances < np.diag(distances)[:, None]).sum(axis=1)
    positives = np.sort(np.diag(distances))
    threshold = positives[int(np.ceil(0.95 * len(positives))) - 1]
    negatives = distances[~np.eye(len(distances), dtype=bool)]
    return {
        "Recall@1": float((ranks <= 1).mean()),
        "Recall@5": float((ranks <= 5).mean()),
        "Recall@10": float((ranks <= 10).mean()),
        "MRR": float((1 / ranks).mean()),
        "FPR95": float((negatives <= threshold).mean()),
        "gallery_size": int(len(ranks)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true", help="check one batch without updating weights")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_ids = list(range(1, 16869))
    val_ids = list(range(16869, 18969))
    train_sar_paths = [DATA / "train" / f"sar{i}.png" for i in train_ids]
    mean, std, n_pixels = train_sar_statistics(train_sar_paths)
    submission_class = load_submission()
    global PREPARE_IMAGE
    namespace: dict[str, Any] = {}
    notebook = json.loads((HW_DIR / "HW1.ipynb").read_text(encoding="utf-8"))
    source = next(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        and "".join(cell.get("source", [])).startswith("# === SUBMISSION ===")
    )
    exec(compile(source, str(HW_DIR / "HW1.ipynb"), "exec"), namespace)
    assert namespace["Submission"] is not submission_class
    PREPARE_IMAGE = namespace["prepare_submission_image"]
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(device_name)

    train_loader = DataLoader(
        SarOptPairs(train_ids, True, mean, std),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=2,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(SEED),
    )
    val_loader = DataLoader(
        SarOptPairs(val_ids, False, mean, std),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )

    model = namespace["PseudoSiameseNet"](dim=256)
    initial = torch.load(INITIAL_WEIGHTS, map_location="cpu", weights_only=True)
    model.load_state_dict(initial)
    model.to(device)

    if args.smoke_test:
        sar_batch, opt_batch = next(iter(train_loader))
        sar_batch, opt_batch = sar_batch.to(device), opt_batch.to(device)
        model.train()
        loss = triplet_loss(model.embed_sar(sar_batch), model.embed_opt(opt_batch))
        loss.backward()
        gradient_norm = torch.sqrt(
            sum(parameter.grad.detach().float().square().sum() for parameter in model.parameters() if parameter.grad is not None)
        )
        if not torch.isfinite(loss) or not torch.isfinite(gradient_norm) or gradient_norm <= 0:
            raise ValueError("SAR normalization smoke test produced invalid loss/gradients")
        print(json.dumps({"smoke_test": True, "batch_size": len(sar_batch), "loss": float(loss.detach()), "gradient_norm": float(gradient_norm), "weights_updated": False}))
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    config = {
        "run_name": "sar_train_norm",
        "seed": SEED,
        "initial_checkpoint": str(INITIAL_WEIGHTS.relative_to(ROOT)),
        "dim": 256,
        "epochs_limit": EPOCHS,
        "patience": PATIENCE,
        "batch_size": BATCH_SIZE,
        "margin": MARGIN,
        "lr": LR,
        "weight_decay": 1e-4,
        "scheduler": {"name": "StepLR", "step_size": 10, "gamma": 0.5},
        "preprocessing": {
            "sar": {"mean": [mean] * 3, "std": [std] * 3},
            "opt": OPT_PREPROCESSING,
        },
        "sar_train_pixels": n_pixels,
        "official_train_ids": [1, 16868],
        "official_val_ids": [16869, 18968],
        "loss": "manual symmetric batch-hard triplet",
    }
    best_recall1 = -1.0
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        n_train = 0
        for sar_batch, opt_batch in train_loader:
            sar_batch, opt_batch = sar_batch.to(device), opt_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = triplet_loss(model.embed_sar(sar_batch), model.embed_opt(opt_batch))
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite training loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            train_loss_sum += float(loss.detach()) * len(sar_batch)
            n_train += len(sar_batch)

        model.eval()
        val_loss_sum = 0.0
        val_seen = 0
        with torch.no_grad():
            for sar_batch, opt_batch in val_loader:
                a = model.embed_sar(sar_batch.to(device))
                b = model.embed_opt(opt_batch.to(device))
                if len(a) >= 2:
                    val_loss_sum += float(triplet_loss(a, b)) * len(a)
                    val_seen += len(a)
        sar_emb, opt_emb = embed_loader(model, val_loader, device)
        metrics = retrieval_metrics(sar_emb, opt_emb)
        row = {
            "epoch": epoch,
            "train_loss": train_loss_sum / n_train,
            "val_loss": val_loss_sum / val_seen,
            "val_recall1": float(metrics["Recall@1"]),
            "val_recall5": float(metrics["Recall@5"]),
            "val_recall10": float(metrics["Recall@10"]),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

        if row["val_recall1"] > best_recall1:
            best_recall1 = float(row["val_recall1"])
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {"state_dict": model.state_dict(), "config": config, "best_epoch": best_epoch},
                OUT / "hw1_best.pt",
            )
        else:
            stale_epochs += 1
        scheduler.step()
        if stale_epochs >= PATIENCE:
            break

    with (OUT / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    best_submission = submission_class(OUT / "hw1_best.pt", device=str(device))
    val_sar_paths = [DATA / "val" / f"sar{i}.png" for i in val_ids]
    val_opt_paths = [DATA / "val" / f"opt{i}.png" for i in val_ids]
    sar_emb = embed_paths_from_submission(best_submission.embed_sar, val_sar_paths)
    opt_emb = embed_paths_from_submission(best_submission.embed_opt, val_opt_paths)
    full_metrics = retrieval_metrics(sar_emb, opt_emb)
    n1000_metrics = retrieval_metrics(sar_emb[:1000], opt_emb[:1000])
    result = {
        **config,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "best_val_recall1": best_recall1,
        "training_time_s": round(time.perf_counter() - started, 2),
        "validation_n2100": full_metrics,
        "validation_n1000_submission": {
            **n1000_metrics,
            "n_pairs": 1000,
            "batch_independent": batch_independence(best_submission, val_sar_paths),
        },
    }
    (OUT / "run_metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def embed_paths_from_submission(embed_fn: Any, paths: list[Path]) -> np.ndarray:
    parts = []
    for start in range(0, len(paths), 100):
        images = []
        for path in paths[start : start + 100]:
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise FileNotFoundError(path)
            images.append(image)
        parts.append(embed_fn(images))
    return np.concatenate(parts).astype(np.float64)


def batch_independence(submission: Any, paths: list[Path]) -> bool:
    images = [cv2.imread(str(path), cv2.IMREAD_UNCHANGED) for path in paths[:4]]
    batch = submission.embed_sar(images)
    single = submission.embed_sar(images[:1])
    scale = np.abs(batch[:1]).max() + 1e-12
    return bool(np.abs(batch[:1] - single).max() / scale < 1e-3)


if __name__ == "__main__":
    main()
