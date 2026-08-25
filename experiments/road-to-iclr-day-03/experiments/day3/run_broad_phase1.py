"""Launch one deterministic dataset shard of broad sensitivity/remedy/natural tiers."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from .broad_data import config


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/day3/broad_benchmark"
PYTHON = Path("/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python")


def call(common: list[str], *, remedies: list[str], representations: list[str], models: list[str], setting: dict) -> None:
    command = [
        str(PYTHON),
        "-m",
        "experiments.day3.run_broad_benchmark",
        *common,
        "--representations",
        *representations,
        "--models",
        *models,
        "--remedies",
        *remedies,
        "--learning-rate",
        str(setting["learning_rate"]),
        "--ridge",
        str(setting.get("ridge", 1e-8)),
        "--precondition-frequency",
        str(setting.get("precondition_frequency", 10)),
    ]
    subprocess.run(command, check=True, cwd=ROOT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = config()
    datasets = [
        name for index, name in enumerate(cfg["datasets"]) if index % args.num_shards == args.shard
    ]
    selected = json.loads((RESULTS / "selected_hyperparameters.json").read_text())["selected"]
    common = [
        "--datasets",
        *datasets,
        "--kappas",
        "1",
        "1000",
        "--seeds",
        *map(str, cfg["broad_seeds"]),
        "--device",
        args.device,
        "--output",
        str(args.output),
    ]
    # Tier 1: architecture-wide AdamW sensitivity.
    call(
        common,
        remedies=["adamw"],
        representations=["controlled"],
        models=cfg["models"],
        setting=selected["adamw"],
    )
    # Tier 2: MLP optimizer and preprocessing remedies. The sketched exact
    # canonicalizer shares the validation-selected AdamW setting.
    for remedy in cfg["remedies"]:
        if remedy == "adamw":
            continue
        setting = selected.get(remedy, selected.get("anchor_whiten_adamw"))
        call(
            common,
            remedies=[remedy],
            representations=["controlled"],
            models=["mlp"],
            setting=setting,
        )
    # Tier 4: unperturbed natural exact encodings and standard controls.
    call(
        common,
        remedies=["adamw"],
        representations=[
            "cumulative_helmert",
            "local_adjacent",
            "raw_standard",
            "quantile_standard",
        ],
        models=["mlp", "resnet"],
        setting=selected["adamw"],
    )


if __name__ == "__main__":
    main()
