"""Conditional 13-seed confirmation of four selected claim-instability cases."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
OLD = (
    HERE.parent
    / "road-to-iclr-day-03/results/day3/broad_benchmark/final_natural_encoding_pairs.csv"
)
RAW = {
    "mlp": [HERE / "compustat_claim_mlp_16seeds.csv", HERE / "claim_crossrope_mlp_16seeds.csv"],
    "resnet": [HERE / "compustat_claim_resnet_16seeds.csv", HERE / "claim_crossrope_resnet_16seeds.csv"],
}
DATASETS = (
    "churn",
    "compustat_kor_direction",
    "heloc_credit_risk",
    "polish_bankruptcy_2year",
)
CHARTS = ("cumulative_helmert", "local_adjacent")


def interval(values: np.ndarray, rng: np.random.Generator) -> list[float]:
    draws = values[rng.integers(0, len(values), size=(100_000, len(values)))].mean(axis=1)
    return np.quantile(draws, (0.025, 0.975)).tolist()


def sign_flip_p(values: np.ndarray) -> float:
    observed = abs(float(values.mean()))
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(values))))
    return float(np.mean(np.abs((signs * values).mean(axis=1)) >= observed - 1e-15))


def load_panel() -> pd.DataFrame:
    old = pd.read_csv(OLD)
    old = old[old.dataset.isin(DATASETS)][["dataset", "model", "seed", *CHARTS]]
    new_frames = []
    for model, paths in RAW.items():
        for path in paths:
            data = pd.read_csv(path)
            if data.failure.fillna("").str.strip().ne("").any():
                raise RuntimeError(f"failed cells in {path}")
            data = data[data.dataset.isin(DATASETS)]
            pivot = data.pivot(
                index=["dataset", "seed"], columns="representation", values="test_primary"
            ).reset_index()
            pivot["model"] = model
            new_frames.append(pivot[["dataset", "model", "seed", *CHARTS]])
    panel = pd.concat([old, *new_frames], ignore_index=True)
    expected = {
        (dataset, model, seed)
        for dataset in DATASETS
        for model in RAW
        for seed in range(16)
    }
    observed = set(zip(panel.dataset, panel.model, panel.seed))
    if observed != expected or panel.duplicated(["dataset", "model", "seed"]).any():
        raise ValueError(f"panel mismatch; missing={sorted(expected-observed)}")
    return panel.sort_values(["dataset", "seed", "model"]).reset_index(drop=True)


def summarize(values: dict[str, np.ndarray], index: np.ndarray, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    result = {}
    for name, all_values in values.items():
        selected = all_values[index]
        result[name] = {
            "mean": float(selected.mean()),
            "bootstrap_95": interval(selected, rng),
            "exact_two_sided_sign_flip_p": sign_flip_p(selected),
            "positive_seed_fraction": float((selected > 0).mean()),
        }
    result["representative_mean_range"] = [
        min(result[chart]["mean"] for chart in CHARTS),
        max(result[chart]["mean"] for chart in CHARTS),
    ]
    result["representative_means_have_opposite_sign"] = bool(
        result[CHARTS[0]]["mean"] * result[CHARTS[1]]["mean"] < 0
    )
    result["within_seed_chart_winner_change_fraction"] = float(
        np.mean(values[CHARTS[0]][index] * values[CHARTS[1]][index] < 0)
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE / "claim_confirmation_panel.json")
    parser.add_argument("--panel-output", type=Path, default=HERE / "claim_confirmation_panel.csv")
    args = parser.parse_args()
    panel = load_panel()
    panel.to_csv(args.panel_output, index=False)
    datasets = []
    for offset, (dataset, group) in enumerate(panel.groupby("dataset", sort=True)):
        models = {name: frame.set_index("seed") for name, frame in group.groupby("model")}
        values = {
            chart: (models["mlp"][chart] - models["resnet"][chart]).to_numpy()
            for chart in CHARTS
        }
        values["quotient"] = np.mean([values[chart] for chart in CHARTS], axis=0)
        values["chart_by_architecture"] = values[CHARTS[0]] - values[CHARTS[1]]
        datasets.append(
            {
                "dataset": dataset,
                "initial_3_seeds": summarize(values, np.arange(3), 10_000 + offset),
                "new_13_seed_confirmation": summarize(
                    values, np.arange(3, 16), 20_000 + offset
                ),
                "combined_16_seed_descriptive": summarize(
                    values, np.arange(16), 30_000 + offset
                ),
            }
        )
    confirmations = [row["new_13_seed_confirmation"] for row in datasets]
    result = {
        "estimand": "paired-seed MLP minus ResNet test ROC-AUC; positive favors MLP",
        "selection_warning": "all four datasets were selected from the three-seed screen; seeds 3--15 are the only conditional confirmation",
        "conditional_confirmation_aggregate": {
            "dataset_count": len(datasets),
            "opposite_representative_mean_count": sum(
                row["representative_means_have_opposite_sign"] for row in confirmations
            ),
            "mean_within_seed_chart_winner_change_fraction": float(
                np.mean([row["within_seed_chart_winner_change_fraction"] for row in confirmations])
            ),
            "quotient_sign_flip_p_below_0.05_count": sum(
                row["quotient"]["exact_two_sided_sign_flip_p"] < 0.05
                for row in confirmations
            ),
        },
        "datasets": datasets,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
