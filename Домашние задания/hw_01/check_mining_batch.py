"""Compare manual hard and semi-hard selection on one deterministic train batch."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

HW_DIR = Path(__file__).resolve().parent
ROOT = HW_DIR.parents[1]
TRAIN_DIR = ROOT / "data" / "QXSLAB_SAROPT" / "train"
WEIGHTS = HW_DIR / "results" / "first_run" / "hw1_best.pt"
REPORT_PATH = HW_DIR / "mining_batch_check.json"
SEED = 2026
BATCH_SIZE = 32
MARGIN = 0.2


def main() -> None:
    torch.manual_seed(SEED)
    random.seed(SEED)
    ids = sorted(int(re.search(r"\d+", path.stem).group()) for path in TRAIN_DIR.glob("sar*.png"))
    random.Random(SEED).shuffle(ids)
    ids = ids[:BATCH_SIZE]

    namespace: dict[str, Any] = {}
    notebook = json.loads((HW_DIR / "HW1.ipynb").read_text(encoding="utf-8"))
    preprocessing = json.loads((HW_DIR / "first_run_metrics.json").read_text(encoding="utf-8"))["preprocessing"]
    source = next(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        and "".join(cell.get("source", [])).startswith("# === SUBMISSION ===")
    )
    exec(compile(source, str(HW_DIR / "HW1.ipynb"), "exec"), namespace)

    def prepare(modality: str, image_id: int) -> torch.Tensor:
        path = TRAIN_DIR / f"{modality}{image_id}.png"
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(path)
        return namespace["prepare_submission_image"](image, preprocessing[modality])

    sar = torch.stack([prepare("sar", image_id) for image_id in ids])
    opt = torch.stack([prepare("opt", image_id) for image_id in ids])
    model = namespace["PseudoSiameseNet"](dim=256)
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).train()
    sar_emb = model.embed_sar(sar.to(device))
    opt_emb = model.embed_opt(opt.to(device))
    distances = torch.cdist(sar_emb.float(), opt_emb.float())
    positive = distances.diagonal()
    diagonal = torch.eye(BATCH_SIZE, device=distances.device, dtype=torch.bool)
    negatives = distances.masked_fill(diagonal, float("inf"))

    hard_rows = negatives.argmin(dim=1)
    hard_columns = negatives.argmin(dim=0)
    hard_loss = (
        torch.relu(positive - negatives.min(dim=1).values + MARGIN).mean()
        + torch.relu(positive - negatives.min(dim=0).values + MARGIN).mean()
    ) / 2

    def semi_hard_indices(matrix: torch.Tensor) -> tuple[torch.Tensor, int]:
        selected: list[torch.Tensor] = []
        semi_hard_count = 0
        for anchor in range(BATCH_SIZE):
            candidates = torch.where(
                (~diagonal[anchor])
                & (matrix[anchor] > positive[anchor])
                & (matrix[anchor] < positive[anchor] + MARGIN)
            )[0]
            if candidates.numel():
                selected.append(candidates[matrix[anchor, candidates].argmin()])
                semi_hard_count += 1
            else:
                selected.append(matrix[anchor].masked_fill(diagonal[anchor], float("inf")).argmin())
        return torch.stack(selected), semi_hard_count

    semi_rows, row_count = semi_hard_indices(distances)
    semi_columns, column_count = semi_hard_indices(distances.T)
    semi_hard_loss = (
        torch.relu(positive - distances[torch.arange(BATCH_SIZE), semi_rows] + MARGIN).mean()
        + torch.relu(positive - distances[semi_columns, torch.arange(BATCH_SIZE)] + MARGIN).mean()
    ) / 2

    model.zero_grad(set_to_none=True)
    hard_loss.backward(retain_graph=True)
    hard_grad_norm = torch.sqrt(
        sum(parameter.grad.detach().float().square().sum() for parameter in model.parameters() if parameter.grad is not None)
    )
    model.zero_grad(set_to_none=True)
    semi_hard_loss.backward()
    semi_hard_grad_norm = torch.sqrt(
        sum(parameter.grad.detach().float().square().sum() for parameter in model.parameters() if parameter.grad is not None)
    )

    nearby_hard_negatives = sum(
        abs(ids[anchor] - ids[int(hard_rows[anchor])]) <= 16 for anchor in range(BATCH_SIZE)
    ) + sum(abs(ids[anchor] - ids[int(hard_columns[anchor])]) <= 16 for anchor in range(BATCH_SIZE))
    result = {
        "checkpoint": str(WEIGHTS.relative_to(ROOT)),
        "split": "train",
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "margin": MARGIN,
        "hard_loss": float(hard_loss.detach()),
        "semi_hard_loss": float(semi_hard_loss.detach()),
        "semi_hard_selected": row_count + column_count,
        "directional_selections": 2 * BATCH_SIZE,
        "hard_gradient_norm": float(hard_grad_norm),
        "semi_hard_gradient_norm": float(semi_hard_grad_norm),
        "hard_negative_id_gap_le_16": nearby_hard_negatives,
        "weights_updated": False,
        "decision": "Hard and semi-hard losses/selectors coincide on this deterministic batch; no full training justified by this check.",
    }
    if not np.isfinite([result["hard_loss"], result["semi_hard_loss"], result["hard_gradient_norm"], result["semi_hard_gradient_norm"]]).all():
        raise ValueError("Найдены не конечные loss или градиенты")
    REPORT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
