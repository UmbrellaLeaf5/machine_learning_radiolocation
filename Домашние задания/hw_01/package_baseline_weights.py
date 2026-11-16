"""Package existing raw model state dicts with their recorded input preprocessing."""

from __future__ import annotations

import json
from pathlib import Path

import torch


HW_DIR = Path(__file__).resolve().parent
METRICS_PATH = HW_DIR / "first_run_metrics.json"
RUNS = {
    "first_run": HW_DIR / "results" / "first_run",
    "dim128": HW_DIR / "results" / "dim128",
}


def main() -> None:
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    preprocessing = metrics["preprocessing"]
    config = {
        "preprocessing": {
            name: {"mean": values["mean"], "std": values["std"]}
            for name, values in preprocessing.items()
        }
    }
    for run_name, directory in RUNS.items():
        state = torch.load(directory / "hw1_best.pt", map_location="cpu", weights_only=True)
        package = {"state_dict": state, "config": config, "run_name": run_name}
        output = directory / "submission.pt"
        torch.save(package, output)
        print(json.dumps({"run": run_name, "weights": str(output), "dim": int(state["sar_branch.fc.weight"].shape[0])}))


if __name__ == "__main__":
    main()
