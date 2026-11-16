"""Package the dim128 checkpoint produced by the final top-to-bottom notebook run."""

from __future__ import annotations

import json
from pathlib import Path

import torch


HW_DIR = Path(__file__).resolve().parent


def main() -> None:
    metrics = json.loads((HW_DIR / "first_run_metrics.json").read_text(encoding="utf-8"))
    preprocessing = {
        name: {"mean": value["mean"], "std": value["std"]}
        for name, value in metrics["preprocessing"].items()
    }
    state = torch.load(
        HW_DIR / "results" / "dim128" / "hw1_best.pt",
        map_location="cpu",
        weights_only=True,
    )
    output = HW_DIR / "results" / "dim128" / "submission_notebook_rerun.pt"
    torch.save({"state_dict": state, "config": {"preprocessing": preprocessing}, "run_name": "dim128_notebook_rerun"}, output)
    print(json.dumps({"checkpoint": str(output), "dim": int(state["sar_branch.fc.weight"].shape[0])}))


if __name__ == "__main__":
    main()
