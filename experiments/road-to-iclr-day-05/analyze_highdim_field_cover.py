"""Analyze high-dimensional field-wise OA-32 versus iid-32 ensembles."""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent


def brier(y: np.ndarray, predictions: np.ndarray) -> float:
    targets = np.eye(2)[y.astype(int)]
    return float(np.mean(np.sum((predictions - targets) ** 2, axis=-1)))


def repetition_metrics(y: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    repetitions = len(predictions)
    mean = predictions.mean(axis=0)
    unbiased_risk = float(
        repetitions / (repetitions - 1)
        * np.mean(np.sum((predictions - mean[None, ...]) ** 2, axis=-1))
    )
    losses = np.asarray([brier(y, prediction) for prediction in predictions])
    hard = np.argmax(predictions, axis=-1)
    return {
        "unbiased_between_repetition_risk": unbiased_risk,
        "mean_repetition_brier": float(losses.mean()),
        "repetition_brier_std": float(losses.std(ddof=1)),
        "grand_mean_brier": brier(y, mean),
        "hard_flip_fraction": float(np.mean(np.any(hard != hard[0:1], axis=0))),
    }


def exact_sign_p(wins: int, total: int) -> float:
    tail = min(wins, total - wins)
    return float(min(1, 2 * sum(math.comb(total, k) for k in range(tail + 1)) / 2**total))


def main() -> None:
    config = json.loads((HERE / "highdim_field_config.json").read_text())
    input_dir = HERE / "results" / "highdim_field_cover"
    rows = []
    for dataset, model in itertools.product(config["datasets"], config["models"]):
        stem = f"{dataset}__{model}"
        archive = np.load(input_dir / f"{stem}.npz")
        manifest = json.loads((input_dir / f"{stem}.json").read_text())
        for split in ("validation", "test"):
            predictions = archive[f"{split}_predictions"].astype(np.float64)
            y = archive[f"{split}_y"]
            method_metrics = [repetition_metrics(y, predictions[index]) for index in range(2)]
            grand_difference = float(np.mean(np.sum(
                (predictions[0].mean(axis=0) - predictions[1].mean(axis=0)) ** 2,
                axis=-1,
            )))
            for index, method in enumerate(manifest["methods"]):
                rows.append({
                    "dataset": dataset, "model": model, "split": split,
                    "method": method, "categorical_fields": manifest["categorical_fields"],
                    "binary_factors": manifest["binary_factors"],
                    "full_joint_cell_count": manifest["full_joint_cell_count"],
                    "grand_method_mean_squared_difference": grand_difference,
                    **method_metrics[index],
                })
    frame = pd.DataFrame(rows)
    test = frame[frame.split == "test"].pivot(
        index=["dataset", "model"], columns="method",
        values=["unbiased_between_repetition_risk", "mean_repetition_brier", "hard_flip_fraction"],
    )
    risk_oa = test[("unbiased_between_repetition_risk", "oa32")]
    risk_iid = test[("unbiased_between_repetition_risk", "iid32")]
    brier_oa = test[("mean_repetition_brier", "oa32")]
    brier_iid = test[("mean_repetition_brier", "iid32")]
    wins = int((risk_oa < risk_iid).sum())
    dataset_wins = {}
    for dataset in config["datasets"]:
        selection = test.loc[dataset]
        dataset_wins[dataset] = bool(
            selection[("unbiased_between_repetition_risk", "oa32")].mean()
            < selection[("unbiased_between_repetition_risk", "iid32")].mean()
        )
    summary = {
        "status": "complete", "cells": len(test),
        "cells_oa32_lower_risk": wins,
        "exact_two_sided_cell_sign_p": exact_sign_p(wins, len(test)),
        "datasets_oa32_lower_mean_risk": int(sum(dataset_wins.values())),
        "dataset_results": dataset_wins,
        "pooled_risk_reduction": float(1 - risk_oa.mean() / risk_iid.mean()),
        "mean_relative_brier_change": float(((brier_oa - brier_iid) / brier_iid).mean()),
        "cells_oa32_lower_brier": int((brier_oa < brier_iid).sum()),
        "mean_full_joint_cell_count": float(frame.full_joint_cell_count.mean()),
        "maximum_grand_method_mean_squared_difference": float(frame.grand_method_mean_squared_difference.max()),
    }
    output = HERE / "results"
    frame.to_csv(output / "highdim_field_cover_metrics.csv", index=False)
    (output / "highdim_field_cover_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

