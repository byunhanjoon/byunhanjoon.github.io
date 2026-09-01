"""Repetition bootstrap for high-dimensional nuisance-cover experiments."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent


def bootstrap_method(
    y: np.ndarray, predictions: np.ndarray, counts: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    repetitions, rows, classes = predictions.shape
    flat = predictions.astype(np.float64).reshape(repetitions, rows * classes)
    gram = flat @ flat.T / rows
    diagonal = np.diag(gram)
    first = counts @ diagonal
    quadratic = np.einsum("bi,ij,bj->b", counts, gram, counts, optimize=True) / repetitions
    risk = (first - quadratic) / (repetitions - 1)
    targets = np.eye(classes)[y.astype(int)]
    losses = np.mean(np.sum((predictions - targets[None, ...]) ** 2, axis=-1), axis=1)
    brier = counts @ losses / repetitions
    return np.maximum(risk, 0), brier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", choices=("field", "row"), required=True)
    args = parser.parse_args()
    if args.study == "field":
        config_name, directory, controls = "highdim_field_config.json", "highdim_field_cover", ("iid32",)
    else:
        config_name, directory, controls = "highdim_row_config.json", "highdim_row_cover", ("marginal32", "iid32")
    config = json.loads((HERE / config_name).read_text())
    input_dir = HERE / "results" / directory
    draws = 50_000
    rng = np.random.default_rng(2026082811 + (args.study == "row"))
    rows = []
    pooled: dict[str, list[np.ndarray]] = {method: [] for method in ("oa32", *controls)}
    pooled_brier: dict[str, list[np.ndarray]] = {method: [] for method in ("oa32", *controls)}
    for dataset, model in itertools.product(config["datasets"], config["models"]):
        stem = f"{dataset}__{model}"
        archive = np.load(input_dir / f"{stem}.npz")
        manifest = json.loads((input_dir / f"{stem}.json").read_text())
        methods = manifest["methods"]
        predictions = archive["test_predictions"]
        y = archive["test_y"]
        boot = {}
        for method in methods:
            reps = predictions.shape[1]
            counts = rng.multinomial(reps, np.repeat(1 / reps, reps), size=draws)
            boot[method] = bootstrap_method(y, predictions[methods.index(method)], counts)
            pooled[method].append(boot[method][0])
            pooled_brier[method].append(boot[method][1])
        for control in controls:
            ratio = np.divide(
                boot["oa32"][0], boot[control][0],
                out=np.full(draws, np.nan), where=boot[control][0] > 0,
            )
            brier_change = boot["oa32"][1] - boot[control][1]
            rows.append({
                "study": args.study, "dataset": dataset, "model": model, "control": control,
                "median_risk_ratio": float(np.nanmedian(ratio)),
                "risk_ratio_q025": float(np.nanquantile(ratio, 0.025)),
                "risk_ratio_q975": float(np.nanquantile(ratio, 0.975)),
                "bootstrap_probability_oa_lower_risk": float(np.nanmean(ratio < 1)),
                "median_brier_difference": float(np.median(brier_change)),
                "brier_difference_q025": float(np.quantile(brier_change, 0.025)),
                "brier_difference_q975": float(np.quantile(brier_change, 0.975)),
                "bootstrap_probability_oa_lower_brier": float(np.mean(brier_change < 0)),
            })
    summaries = {}
    for control in controls:
        oa = np.mean(np.stack(pooled["oa32"]), axis=0)
        comparison = np.mean(np.stack(pooled[control]), axis=0)
        ratio = oa / comparison
        oa_brier = np.mean(np.stack(pooled_brier["oa32"]), axis=0)
        comparison_brier = np.mean(np.stack(pooled_brier[control]), axis=0)
        summaries[control] = {
            "pooled_risk_ratio_median": float(np.median(ratio)),
            "pooled_risk_ratio_95_interval": [float(np.quantile(ratio, 0.025)), float(np.quantile(ratio, 0.975))],
            "bootstrap_probability_oa_lower_pooled_risk": float(np.mean(ratio < 1)),
            "pooled_brier_difference_median": float(np.median(oa_brier - comparison_brier)),
            "pooled_brier_difference_95_interval": [
                float(np.quantile(oa_brier - comparison_brier, 0.025)),
                float(np.quantile(oa_brier - comparison_brier, 0.975)),
            ],
        }
    summary = {
        "status": "complete", "study": args.study, "bootstrap_draws": draws,
        "inference_scope": "conditional_on_fixed_model_dataset_panel", "comparisons": summaries,
    }
    output = HERE / "results"
    pd.DataFrame(rows).to_csv(output / f"highdim_{args.study}_bootstrap_cells.csv", index=False)
    (output / f"highdim_{args.study}_bootstrap_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

