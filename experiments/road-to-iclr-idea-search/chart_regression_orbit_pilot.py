"""Retain a regression prediction orbit over Day-3 ordinal charts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from chart_regression_transfer_pilot import (
    CHARTS,
    _nonordinal_schema,
    _ordinal_blocks,
    combine,
    fit_adamw,
    load_dataset,
    make_prepared,
    squared_summary,
)


HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="black-friday")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(16)))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    if dataset.task != "regression":
        raise ValueError("This pilot supports regression datasets")
    schema = _nonordinal_schema(dataset)
    prepared = {}
    for chart in CHARTS:
        blocks, metadata = _ordinal_blocks(dataset, chart)
        prepared[chart] = make_prepared(dataset, combine([schema, *blocks]), metadata)

    predictions = np.empty(
        (len(CHARTS), len(args.seeds), len(dataset.y["test"]), 1),
        dtype=np.float64,
    )
    fits = []
    for chart_index, chart in enumerate(CHARTS):
        for seed_index, seed in enumerate(args.seeds):
            prediction, fit = fit_adamw(
                prepared[chart], seed, args.device, model_name="mlp"
            )
            predictions[chart_index, seed_index] = prediction
            fits.append({"chart": chart, "seed": seed, **fit})
            print(
                f"{args.dataset} {chart:26s} seed={seed:2d} "
                f"epoch={fit['best_epoch']:2d} seconds={fit['seconds']:.2f}",
                flush=True,
            )

    reference = prepared[CHARTS[0]]
    labels = reference.y["test"].astype(np.float64)
    output = {
        "design": {
            "dataset": args.dataset,
            "charts": list(CHARTS),
            "seeds": args.seeds,
            "split_fingerprint": dataset.split_fingerprint,
        },
        "orbit": squared_summary(
            predictions, labels, ("chart", "seed"), reference.y_scale
        ),
        "fits": fits,
    }
    np.savez_compressed(
        args.output,
        charts=np.asarray(CHARTS),
        seeds=np.asarray(args.seeds),
        y_test=labels,
        y_scale=np.asarray(reference.y_scale),
        adamw_predictions=predictions,
    )
    args.output.with_suffix(".json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
