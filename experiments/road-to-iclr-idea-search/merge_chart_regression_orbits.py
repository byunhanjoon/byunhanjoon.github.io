"""Merge independently executed seed shards and recompute orbit statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from chart_regression_transfer_pilot import squared_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    archives = [np.load(path) for path in args.inputs]
    charts = archives[0]["charts"]
    labels = archives[0]["y_test"]
    scale = float(archives[0]["y_scale"])
    seeds = np.concatenate([archive["seeds"] for archive in archives])
    predictions = np.concatenate(
        [archive["adamw_predictions"] for archive in archives], axis=1
    )
    order = np.argsort(seeds)
    seeds = seeds[order]
    predictions = predictions[:, order]
    if len(np.unique(seeds)) != len(seeds):
        raise ValueError("Seed shards overlap")
    for archive in archives[1:]:
        if not np.array_equal(archive["charts"], charts):
            raise ValueError("Chart definitions differ")
        if not np.array_equal(archive["y_test"], labels):
            raise ValueError("Evaluation targets differ")

    output = {
        "design": {
            "dataset": args.dataset,
            "charts": charts.astype(str).tolist(),
            "seeds": seeds.tolist(),
            "source_shards": [str(path) for path in args.inputs],
        },
        "orbit": squared_summary(
            predictions, labels, ("chart", "seed"), scale
        ),
    }
    np.savez_compressed(
        args.output,
        charts=charts,
        seeds=seeds,
        y_test=labels,
        y_scale=np.asarray(scale),
        adamw_predictions=predictions,
    )
    args.output.with_suffix(".json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
