"""Post-gate repetition bootstrap for the prospective strength-3 panel."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_highdim_uncertainty import bootstrap_method


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
INPUT = RESULTS / "highdim_strength3_cover"
ACTION = "strength3_oa128"
CONTROLS = ("four_strength2_oa32", "four_marginal32", "iid128")


def quantiles(values: np.ndarray) -> list[float]:
    return [float(np.nanquantile(values, q)) for q in (0.025, 0.5, 0.975)]


def main() -> None:
    config = json.loads((HERE / "highdim_strength3_config.json").read_text())
    draws = 50_000
    rng = np.random.default_rng(2026082817)
    rows: list[dict[str, object]] = []
    pooled_risk: dict[str, list[np.ndarray]] = {
        method: [] for method in (ACTION, *CONTROLS)
    }
    pooled_brier: dict[str, list[np.ndarray]] = {
        method: [] for method in (ACTION, *CONTROLS)
    }

    for dataset, model in itertools.product(config["datasets"], config["models"]):
        stem = f"{dataset}__{model}"
        archive = np.load(INPUT / f"{stem}.npz")
        manifest = json.loads((INPUT / f"{stem}.json").read_text())
        predictions = archive["test_predictions"]
        y = archive["test_y"]
        boot: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for method in manifest["methods"]:
            reps = predictions.shape[1]
            counts = rng.multinomial(reps, np.full(reps, 1 / reps), size=draws)
            boot[method] = bootstrap_method(
                y, predictions[manifest["methods"].index(method)], counts
            )
            pooled_risk[method].append(boot[method][0])
            pooled_brier[method].append(boot[method][1])

        for control in CONTROLS:
            ratio = np.divide(
                boot[ACTION][0], boot[control][0],
                out=np.full(draws, np.nan), where=boot[control][0] > 0,
            )
            brier_difference = boot[ACTION][1] - boot[control][1]
            rows.append({
                "dataset": dataset,
                "model": model,
                "control": control,
                "risk_ratio_q025": quantiles(ratio)[0],
                "risk_ratio_median": quantiles(ratio)[1],
                "risk_ratio_q975": quantiles(ratio)[2],
                "bootstrap_probability_strength3_lower_risk": float(np.nanmean(ratio < 1)),
                "brier_difference_q025": quantiles(brier_difference)[0],
                "brier_difference_median": quantiles(brier_difference)[1],
                "brier_difference_q975": quantiles(brier_difference)[2],
                "bootstrap_probability_strength3_lower_brier": float(np.mean(brier_difference < 0)),
            })

    comparisons: dict[str, dict[str, object]] = {}
    for control in CONTROLS:
        action_risk = np.mean(np.stack(pooled_risk[ACTION]), axis=0)
        control_risk = np.mean(np.stack(pooled_risk[control]), axis=0)
        ratio = np.divide(
            action_risk, control_risk,
            out=np.full(draws, np.nan), where=control_risk > 0,
        )
        action_brier = np.mean(np.stack(pooled_brier[ACTION]), axis=0)
        control_brier = np.mean(np.stack(pooled_brier[control]), axis=0)
        brier_difference = action_brier - control_brier
        comparisons[control] = {
            "pooled_risk_ratio_q025_median_q975": quantiles(ratio),
            "bootstrap_probability_strength3_lower_pooled_risk": float(np.nanmean(ratio < 1)),
            "pooled_brier_difference_q025_median_q975": quantiles(brier_difference),
            "bootstrap_probability_strength3_lower_pooled_brier": float(np.mean(brier_difference < 0)),
        }

    summary = {
        "status": "complete",
        "bootstrap_draws": draws,
        "cells": len(config["datasets"]) * len(config["models"]),
        "inference_scope": "conditional_on_fixed_model_dataset_panel_and_four_observed_repetitions",
        "comparisons": comparisons,
    }
    pd.DataFrame(rows).to_csv(RESULTS / "highdim_strength3_bootstrap_cells.csv", index=False)
    (RESULTS / "highdim_strength3_bootstrap_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
