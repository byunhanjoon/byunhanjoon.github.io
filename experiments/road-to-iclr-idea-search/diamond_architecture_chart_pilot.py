"""Seed-paired MLP/ResNet ranking gate over equivalent Diamond charts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

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
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(16)))
    parser.add_argument(
        "--mlp-input",
        type=Path,
        default=HERE / "chart_regression_diamond.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "diamond_architecture_chart.npz",
    )
    args = parser.parse_args()

    dataset = load_dataset("diamond")
    schema = _nonordinal_schema(dataset)
    prepared = {}
    for chart in CHARTS:
        blocks, metadata = _ordinal_blocks(dataset, chart)
        prepared[chart] = make_prepared(
            dataset, combine([schema, *blocks]), metadata
        )

    resnet = np.empty(
        (len(CHARTS), len(args.seeds), len(dataset.y["test"]), 1),
        dtype=np.float64,
    )
    fits = []
    for chart_index, chart in enumerate(CHARTS):
        for seed_index, seed in enumerate(args.seeds):
            prediction, fit = fit_adamw(
                prepared[chart], seed, args.device, model_name="resnet"
            )
            resnet[chart_index, seed_index] = prediction
            fits.append({"chart": chart, "seed": seed, **fit})
            print(
                f"resnet {chart:26s} seed={seed:2d} "
                f"epoch={fit['best_epoch']:2d} seconds={fit['seconds']:.2f}",
                flush=True,
            )

    mlp = np.load(args.mlp_input)["adamw_predictions"]
    if mlp.shape != resnet.shape:
        raise ValueError(f"MLP shape {mlp.shape} does not match ResNet {resnet.shape}")
    y = prepared[CHARTS[0]].y["test"].astype(np.float64)
    ranking = []
    for chart_index, chart in enumerate(CHARTS):
        mlp_mse = np.mean((mlp[chart_index, ..., 0] - y[None]) ** 2, axis=1)
        resnet_mse = np.mean(
            (resnet[chart_index, ..., 0] - y[None]) ** 2, axis=1
        )
        difference = mlp_mse - resnet_mse
        standard_error = float(stats.sem(difference))
        interval = stats.t.interval(
            0.95,
            len(difference) - 1,
            loc=float(difference.mean()),
            scale=standard_error,
        )
        ranking.append(
            {
                "chart": chart,
                "mlp_mean_mse": float(mlp_mse.mean()),
                "resnet_mean_mse": float(resnet_mse.mean()),
                "paired_mlp_minus_resnet": float(difference.mean()),
                "paired_95_interval": list(map(float, interval)),
                "paired_t_p": float(stats.ttest_rel(mlp_mse, resnet_mse).pvalue),
                "fraction_seeds_mlp_better": float(np.mean(difference < 0)),
            }
        )

    output = {
        "design": {
            "dataset": "diamond",
            "charts": list(CHARTS),
            "seeds": args.seeds,
            "warning": "exploratory paired tests; no multiplicity correction",
        },
        "ranking": ranking,
        "resnet_orbit": squared_summary(
            resnet,
            y,
            ("chart", "seed"),
            prepared[CHARTS[0]].y_scale,
        ),
        "fits": fits,
    }
    np.savez_compressed(
        args.output,
        charts=np.asarray(CHARTS),
        seeds=np.asarray(args.seeds),
        y_test=y,
        mlp_predictions=mlp,
        resnet_predictions=resnet,
    )
    args.output.with_suffix(".json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
