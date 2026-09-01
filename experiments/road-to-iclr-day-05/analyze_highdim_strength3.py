"""Analyze the frozen high-dimensional mixed strength-3 panel."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
INPUT = RESULTS / "highdim_strength3_cover"
METHODS = ("strength3_oa128", "four_strength2_oa32", "four_marginal32", "iid128")


def risk(values: np.ndarray) -> float:
    centered = values.astype(np.float64) - values.mean(axis=0, keepdims=True)
    repetitions = len(values)
    return float(np.sum(centered**2) / (repetitions * (repetitions - 1) * values.shape[1]))


def brier(y: np.ndarray, values: np.ndarray) -> float:
    targets = np.eye(2)[y.astype(int)]
    return float(np.mean(np.sum((values - targets[None]) ** 2, axis=-1)))


def sign_p(wins: int, total: int) -> float:
    tail = min(wins, total - wins)
    return min(1.0, 2 * sum(math.comb(total, k) for k in range(tail + 1)) / 2**total)


def main() -> None:
    config = json.loads((HERE / "highdim_strength3_config.json").read_text())
    rows = []
    for dataset in config["datasets"]:
        for model in config["models"]:
            stem = f"{dataset}__{model}"
            archive = np.load(INPUT / f"{stem}.npz")
            manifest = json.loads((INPUT / f"{stem}.json").read_text())
            for split in ("validation", "test"):
                values = archive[f"{split}_predictions"]
                y = archive[f"{split}_y"]
                for index, method in enumerate(manifest["methods"]):
                    rows.append({"dataset": dataset, "model": model, "split": split, "method": method,
                                 "prediction_risk": risk(values[index]), "brier": brier(y, values[index])})
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "highdim_strength3_metrics.csv", index=False)
    test = frame[frame.split == "test"].pivot(index=["dataset", "model"], columns="method")
    action = test.prediction_risk.strength3_oa128
    controls = [name for name in METHODS if name != "strength3_oa128"]
    wins = np.ones(len(action), dtype=bool)
    comparisons = {}
    for control in controls:
        current = action < test.prediction_risk[control]
        wins &= current
        comparisons[control] = {
            "cells_lower": int(current.sum()), "pooled_reduction": float(1 - action.mean() / test.prediction_risk[control].mean()),
            "sign_p": sign_p(int(current.sum()), len(current)),
        }
    dataset_wins = {}
    for dataset in config["datasets"]:
        current = test.loc[dataset].prediction_risk
        dataset_wins[dataset] = bool(all(current.strength3_oa128.mean() < current[c].mean() for c in controls))
    summary = {
        "status": "complete", "cells": len(action), "cells_strength3_lower_all": int(wins.sum()),
        "comparisons": comparisons, "dataset_means_lower_all": dataset_wins,
        "datasets_lower_all": int(sum(dataset_wins.values())),
        "frozen_gate_passed": bool(wins.sum() >= 6 and sum(dataset_wins.values()) >= 2),
        "mean_brier_by_method": {method: float(test.brier[method].mean()) for method in METHODS},
    }
    (RESULTS / "highdim_strength3_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

