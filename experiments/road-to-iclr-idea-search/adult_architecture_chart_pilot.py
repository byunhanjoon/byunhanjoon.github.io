"""Seed-paired MLP/ResNet comparison over equivalent Adult charts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

from chart_orbit_pilot import (
    CHARTS,
    _nonordinal_schema,
    _ordinal_blocks,
    combine,
    fit_mlp_probabilities,
    load_dataset,
    make_prepared,
)
from orbit_anova import risk_summary


HERE = Path(__file__).resolve().parent


def brier_by_seed(predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
    targets = np.eye(predictions.shape[-1])[labels]
    return np.mean(np.sum((predictions - targets[None]) ** 2, axis=-1), axis=-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(16)))
    parser.add_argument(
        "--mlp-input", type=Path, default=HERE / "chart_orbit_adult_s32.npz"
    )
    parser.add_argument(
        "--output", type=Path, default=HERE / "adult_architecture_chart.npz"
    )
    args = parser.parse_args()

    dataset = load_dataset("adult")
    schema = _nonordinal_schema(dataset)
    prepared = {}
    for chart in CHARTS:
        blocks, metadata = _ordinal_blocks(dataset, chart)
        prepared[chart] = make_prepared(dataset, combine([schema, *blocks]), metadata)

    resnet = np.empty(
        (len(CHARTS), len(args.seeds), len(dataset.y["test"]), 2),
        dtype=np.float64,
    )
    fits = []
    for chart_index, chart in enumerate(CHARTS):
        for seed_index, seed in enumerate(args.seeds):
            prediction, fit = fit_mlp_probabilities(
                prepared[chart], seed, args.device, model_name="resnet"
            )
            resnet[chart_index, seed_index] = prediction
            fits.append({"chart": chart, "seed": seed, **fit})
            print(
                f"resnet {chart:26s} seed={seed:2d} "
                f"epoch={fit['best_epoch']:2d} seconds={fit['train_seconds']:.2f}",
                flush=True,
            )

    mlp_archive = np.load(args.mlp_input)
    archive_seeds = mlp_archive["seeds"].tolist()
    seed_indices = [archive_seeds.index(seed) for seed in args.seeds]
    mlp = mlp_archive["mlp_predictions"][:, seed_indices]
    labels = dataset.y["test"].astype(int)
    ranking = []
    for chart_index, chart in enumerate(CHARTS):
        mlp_brier = brier_by_seed(mlp[chart_index], labels)
        resnet_brier = brier_by_seed(resnet[chart_index], labels)
        difference = mlp_brier - resnet_brier
        interval = stats.t.interval(
            0.95,
            len(difference) - 1,
            loc=float(difference.mean()),
            scale=float(stats.sem(difference)),
        )
        ranking.append(
            {
                "chart": chart,
                "mlp_mean_brier": float(mlp_brier.mean()),
                "resnet_mean_brier": float(resnet_brier.mean()),
                "paired_mlp_minus_resnet": float(difference.mean()),
                "paired_95_interval": list(map(float, interval)),
                "paired_t_p": float(stats.ttest_rel(mlp_brier, resnet_brier).pvalue),
                "fraction_seeds_mlp_better": float(np.mean(difference < 0)),
            }
        )

    output = {
        "design": {
            "dataset": "adult",
            "charts": list(CHARTS),
            "seeds": args.seeds,
            "warning": "exploratory paired tests; no multiplicity correction",
        },
        "ranking": ranking,
        "resnet_orbit": risk_summary(resnet, labels, ("chart", "seed")),
        "fits": fits,
    }
    np.savez_compressed(
        args.output,
        charts=np.asarray(CHARTS),
        seeds=np.asarray(args.seeds),
        y_test=labels,
        mlp_predictions=mlp,
        resnet_predictions=resnet,
    )
    args.output.with_suffix(".json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
