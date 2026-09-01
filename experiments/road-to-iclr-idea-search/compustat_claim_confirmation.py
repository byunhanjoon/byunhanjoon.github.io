"""Confirm the selected Compustat MLP-vs-ResNet chart contrast on new seeds.

Seeds 0--2 were inspected during Day 3.  Seeds 3--15 are the conditional
confirmation set and are always reported separately to avoid reusing the
selection evidence as confirmation.
"""

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
    "mlp": HERE / "compustat_claim_mlp_16seeds.csv",
    "resnet": HERE / "compustat_claim_resnet_16seeds.csv",
}
CHARTS = ("cumulative_helmert", "local_adjacent")


def bootstrap_interval(values: np.ndarray, rng: np.random.Generator) -> list[float]:
    indices = rng.integers(0, len(values), size=(100_000, len(values)))
    return np.quantile(values[indices].mean(axis=1), (0.025, 0.975)).tolist()


def exact_sign_flip_pvalue(values: np.ndarray) -> float:
    observed = abs(float(values.mean()))
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(values))))
    permuted = np.abs((signs * values[None, :]).mean(axis=1))
    return float(np.mean(permuted >= observed - 1e-15))


def load() -> pd.DataFrame:
    old = pd.read_csv(OLD)
    old = old[old.dataset.eq("compustat_kor_direction")].copy()
    old = old[["model", "seed", *CHARTS]]
    new_rows = []
    for model, path in RAW.items():
        data = pd.read_csv(path)
        if data.failure.fillna("").str.strip().ne("").any():
            raise RuntimeError(f"failed cells in {path}")
        pivot = data.pivot(index="seed", columns="representation", values="test_primary")
        for seed, row in pivot.iterrows():
            new_rows.append(
                {"model": model, "seed": int(seed), **{chart: row[chart] for chart in CHARTS}}
            )
    combined = pd.concat([old, pd.DataFrame(new_rows)], ignore_index=True)
    if combined.duplicated(["model", "seed"]).any():
        raise ValueError("duplicate model/seed cells")
    expected = {(model, seed) for model in RAW for seed in range(16)}
    observed = set(zip(combined.model, combined.seed))
    if observed != expected:
        raise ValueError(f"missing cells: {sorted(expected - observed)}")
    return combined.sort_values(["seed", "model"]).reset_index(drop=True)


def contrasts(data: pd.DataFrame) -> dict[str, np.ndarray]:
    indexed = {model: frame.set_index("seed") for model, frame in data.groupby("model")}
    values = {
        chart: (
            indexed["mlp"].loc[:, chart] - indexed["resnet"].loc[:, chart]
        ).to_numpy()
        for chart in CHARTS
    }
    values["quotient"] = np.mean([values[chart] for chart in CHARTS], axis=0)
    values["chart_by_architecture"] = values[CHARTS[0]] - values[CHARTS[1]]
    return values


def summarize(values: dict[str, np.ndarray], indices: np.ndarray, *, exact: bool) -> dict:
    rng = np.random.default_rng(20_260_826 + int(indices[0]))
    output = {}
    for name, all_values in values.items():
        selected = all_values[indices]
        output[name] = {
            "mean": float(selected.mean()),
            "standard_deviation": float(selected.std(ddof=1)),
            "bootstrap_95": bootstrap_interval(selected, rng),
            "positive_seed_fraction": float((selected > 0).mean()),
            "exact_two_sided_sign_flip_p": (
                exact_sign_flip_pvalue(selected) if exact else None
            ),
        }
    output["representative_mean_range"] = [
        min(output[chart]["mean"] for chart in CHARTS),
        max(output[chart]["mean"] for chart in CHARTS),
    ]
    output["within_seed_chart_winner_change_fraction"] = float(
        np.mean(
            values[CHARTS[0]][indices] * values[CHARTS[1]][indices] < 0
        )
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE / "compustat_claim_confirmation.json")
    parser.add_argument("--combined-output", type=Path, default=HERE / "compustat_claim_16seeds.csv")
    args = parser.parse_args()
    data = load()
    data.to_csv(args.combined_output, index=False)
    values = contrasts(data)
    result = {
        "dataset": "compustat_kor_direction",
        "estimand": "paired-seed MLP minus ResNet test ROC-AUC; positive favors MLP",
        "selection_warning": "seeds 0--2 selected this case; only seeds 3--15 are conditional confirmation",
        "initial_3_seeds": summarize(values, np.arange(0, 3), exact=True),
        "new_13_seed_confirmation": summarize(values, np.arange(3, 16), exact=True),
        "combined_16_seed_descriptive": summarize(values, np.arange(0, 16), exact=False),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
