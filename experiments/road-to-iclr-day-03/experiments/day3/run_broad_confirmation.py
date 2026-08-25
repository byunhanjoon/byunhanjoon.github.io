"""Run one shard of the five-seed, four-architecture confirmation tier."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from .broad_data import config


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/day3/broad_benchmark"
PYTHON = "/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = config()
    datasets = [
        name
        for index, name in enumerate(cfg["architecture_confirmation_datasets"])
        if index % args.num_shards == args.shard
    ]
    calibration = json.loads((RESULTS / "selected_hyperparameters.json").read_text())["selected"]
    selection = json.loads((RESULTS / "confirmation_selection.json").read_text())
    remedies = selection["always_confirmed_controls"] + selection["selected_deployable_comparisons"]
    for remedy in remedies:
        setting = calibration.get(remedy, calibration.get("anchor_whiten_adamw", calibration["adamw"]))
        command = [
            PYTHON,
            "-m",
            "experiments.day3.run_broad_benchmark",
            "--datasets",
            *datasets,
            "--representations",
            "controlled",
            "--kappas",
            "1",
            "1000",
            "--models",
            *cfg["models"],
            "--remedies",
            remedy,
            "--seeds",
            *map(str, cfg["confirmation_seeds"]),
            "--learning-rate",
            str(setting["learning_rate"]),
            "--ridge",
            str(setting.get("ridge", 1e-8)),
            "--precondition-frequency",
            str(setting.get("precondition_frequency", 10)),
            "--device",
            args.device,
            "--output",
            str(args.output),
        ]
        subprocess.run(command, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
