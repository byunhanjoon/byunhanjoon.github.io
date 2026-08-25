"""Audit and summarize the frozen 64/128-bin local-basis confirmation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DATASETS = (
    "adult",
    "black-friday",
    "california",
    "churn",
    "diamond",
    "higgs-small",
    "house",
    "microsoft",
    "otto",
)
MODELS = ("mlp", "resnet")
SEEDS = (0, 1, 2, 3)
BINS = (64, 128)
REPRESENTATIONS = (
    "cumulative_ple",
    "local_ple",
    "cumulative_seedmate",
    "basis_blend",
    "basis_select",
    "seed_blend",
    "seed_select",
)
INDEX = ["dataset", "model", "seed", "bins"]


def load_results(results: Path) -> pd.DataFrame:
    paths = [
        results / f"local_basis_confirm{bins}_gpu{gpu}.csv"
        for bins in BINS
        for gpu in (0, 1)
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing confirmation files: " + ", ".join(missing))
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    expected = {
        (dataset, model, seed, bins, representation)
        for dataset in DATASETS
        for model in MODELS
        for seed in SEEDS
        for bins in BINS
        for representation in REPRESENTATIONS
    }
    observed = set(
        frame[[*INDEX, "representation"]].itertuples(index=False, name=None)
    )
    if expected != observed:
        raise RuntimeError(
            f"Incomplete sweep: {len(expected - observed)} missing, "
            f"{len(observed - expected)} unexpected"
        )
    if frame.duplicated([*INDEX, "representation"]).any():
        raise RuntimeError("Duplicate confirmation rows")
    return frame


def paired(frame: pd.DataFrame, representation: str) -> pd.DataFrame:
    base = frame[frame.representation == "cumulative_ple"].set_index(INDEX)
    candidate = frame[frame.representation == representation].set_index(INDEX)
    output = candidate.join(
        base[["task", "test_score", "test_loss"]], rsuffix="_baseline"
    )
    output["loss_reduction_pct"] = 100.0 * (
        output.test_loss_baseline - output.test_loss
    ) / output.test_loss_baseline
    higher_is_better = output.task != "regression"
    direction = np.where(higher_is_better, 1.0, -1.0)
    output["score_improvement_pct"] = (
        100.0
        * direction
        * (output.test_score - output.test_score_baseline)
        / output.test_score_baseline.abs()
    )
    return output


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    return tuple(np.quantile(values, [0.025, 0.975]).tolist())  # type: ignore[return-value]


def bootstrap_mean(
    values: np.ndarray, repetitions: int, seed: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(repetitions, len(values)), replace=True)
    return percentile_interval(samples.mean(axis=1))


def summarize(frame: pd.DataFrame, repetitions: int) -> tuple[pd.DataFrame, dict[str, object]]:
    basis = paired(frame, "basis_blend")
    seed = paired(frame, "seed_blend")
    basis_vs_seed = basis[["test_loss", "test_score", "task"]].join(
        seed[["test_loss", "test_score"]],
        lsuffix="_basis",
        rsuffix="_seed",
    )
    basis_vs_seed["loss_reduction_pct"] = 100.0 * (
        basis_vs_seed.test_loss_seed - basis_vs_seed.test_loss_basis
    ) / basis_vs_seed.test_loss_seed
    direction = np.where(basis_vs_seed.task != "regression", 1.0, -1.0)
    basis_vs_seed["score_improvement_pct"] = (
        100.0
        * direction
        * (basis_vs_seed.test_score_basis - basis_vs_seed.test_score_seed)
        / basis_vs_seed.test_score_seed.abs()
    )

    rows: list[dict[str, object]] = []
    for dataset_index, dataset in enumerate(DATASETS):
        selected = basis.xs(dataset, level="dataset")
        control = seed.xs(dataset, level="dataset")
        comparison = basis_vs_seed.xs(dataset, level="dataset")
        loss_ci = bootstrap_mean(
            selected.loss_reduction_pct.to_numpy(), repetitions, 100 + dataset_index
        )
        rows.append(
            {
                "dataset": dataset,
                "runs": len(selected),
                "basis_loss_wins": int((selected.loss_reduction_pct > 0.0).sum()),
                "basis_mean_loss_reduction_pct": selected.loss_reduction_pct.mean(),
                "basis_loss_reduction_ci_low": loss_ci[0],
                "basis_loss_reduction_ci_high": loss_ci[1],
                "basis_mean_score_improvement_pct": selected.score_improvement_pct.mean(),
                "basis_min_score_improvement_pct": selected.score_improvement_pct.min(),
                "mean_local_weight": selected.blend_alpha_local.mean(),
                "seed_mean_loss_reduction_pct": control.loss_reduction_pct.mean(),
                "basis_minus_seed_loss_reduction_pct": comparison.loss_reduction_pct.mean(),
                "basis_vs_seed_loss_wins": int(
                    (comparison.loss_reduction_pct > 0.0).sum()
                ),
            }
        )
    table = pd.DataFrame(rows)

    dataset_means = table.basis_mean_loss_reduction_pct.to_numpy()
    dataset_ci = bootstrap_mean(dataset_means, repetitions, 20_260_825)
    report: dict[str, object] = {
        "integrity": {
            "datasets": len(DATASETS),
            "models": len(MODELS),
            "seeds": len(SEEDS),
            "bin_settings": len(BINS),
            "paired_runs": len(basis),
            "result_rows": len(frame),
        },
        "basis_blend": {
            "proper_loss_wins": int((basis.loss_reduction_pct > 0.0).sum()),
            "proper_loss_losses": int((basis.loss_reduction_pct < 0.0).sum()),
            "score_wins": int((basis.score_improvement_pct > 0.0).sum()),
            "score_ties": int(np.isclose(basis.score_improvement_pct, 0.0).sum()),
            "score_losses": int((basis.score_improvement_pct < 0.0).sum()),
            "mean_loss_reduction_pct": basis.loss_reduction_pct.mean(),
            "median_loss_reduction_pct": basis.loss_reduction_pct.median(),
            "mean_score_improvement_pct": basis.score_improvement_pct.mean(),
            "datasets_with_positive_mean_loss_reduction": int(
                (table.basis_mean_loss_reduction_pct > 0.0).sum()
            ),
            "dataset_bootstrap_mean_loss_reduction_ci": list(dataset_ci),
            "mean_local_weight": basis.blend_alpha_local.mean(),
        },
        "seed_blend": {
            "proper_loss_wins": int((seed.loss_reduction_pct > 0.0).sum()),
            "proper_loss_losses": int((seed.loss_reduction_pct < 0.0).sum()),
            "mean_loss_reduction_pct": seed.loss_reduction_pct.mean(),
            "mean_score_improvement_pct": seed.score_improvement_pct.mean(),
        },
        "basis_vs_seed": {
            "basis_proper_loss_wins": int(
                (basis_vs_seed.loss_reduction_pct > 0.0).sum()
            ),
            "basis_proper_loss_losses": int(
                (basis_vs_seed.loss_reduction_pct < 0.0).sum()
            ),
            "mean_loss_reduction_pct": basis_vs_seed.loss_reduction_pct.mean(),
            "datasets_where_basis_has_lower_mean_loss": table.loc[
                table.basis_minus_seed_loss_reduction_pct > 0.0, "dataset"
            ].tolist(),
        },
        "cost": {
            "basis_models": 2,
            "seed_control_models": 2,
            "mean_basis_train_seconds": float(
                frame[frame.representation == "basis_blend"].train_seconds.mean()
            ),
            "mean_seed_train_seconds": float(
                frame[frame.representation == "seed_blend"].train_seconds.mean()
            ),
        },
    }
    return table, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=HERE / "results")
    parser.add_argument("--bootstrap-repetitions", type=int, default=20_000)
    parser.add_argument(
        "--table", type=Path, default=HERE / "results" / "local_basis_summary.csv"
    )
    parser.add_argument(
        "--report", type=Path, default=HERE / "results" / "local_basis_summary.json"
    )
    args = parser.parse_args()
    frame = load_results(args.results)
    table, report = summarize(frame, args.bootstrap_repetitions)
    table.to_csv(args.table, index=False, float_format="%.6f")
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(table.round(4).to_string(index=False))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
