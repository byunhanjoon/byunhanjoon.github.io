"""Run one shard of the frozen five-dataset controlled-basis extension."""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path

from . import run_broad_benchmark as benchmark
from .broad_extension_data import extension_config, load_extension_dataset


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/day3/broad_benchmark"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = extension_config()
    datasets = [
        name
        for index, name in enumerate(cfg["datasets"])
        if index % args.num_shards == args.shard
    ]
    selected = json.loads((RESULTS / "selected_hyperparameters.json").read_text())["selected"]
    adamw = selected["adamw"]
    benchmark.load_broad_dataset = load_extension_dataset
    benchmark.run(
        Namespace(
            datasets=datasets,
            representations=[cfg["representation"]],
            kappas=[float(value) for value in cfg["kappas"]],
            models=list(cfg["models"]),
            remedies=[cfg["remedy"]],
            seeds=[int(value) for value in cfg["seeds"]],
            learning_rate=float(adamw["learning_rate"]),
            ridge=float(adamw["ridge"]),
            precondition_frequency=int(adamw["precondition_frequency"]),
            device=args.device,
            output=args.output,
        )
    )


if __name__ == "__main__":
    main()
