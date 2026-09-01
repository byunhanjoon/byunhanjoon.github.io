"""Analyze field-wise plus row-order OA-32 against equal-budget controls."""

from __future__ import annotations

import importlib.util
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("highdim_metrics_row", HERE / "analyze_highdim_field_cover.py")
assert SPEC is not None and SPEC.loader is not None
METRICS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(METRICS)


def exact_sign_p(wins: int, total: int) -> float:
    tail = min(wins, total - wins)
    return float(min(1, 2 * sum(math.comb(total, k) for k in range(tail + 1)) / 2**total))


def main() -> None:
    config = json.loads((HERE / "highdim_row_config.json").read_text())
    input_dir = HERE / "results" / "highdim_row_cover"
    rows = []
    for dataset, model in itertools.product(config["datasets"], config["models"]):
        stem = f"{dataset}__{model}"
        archive = np.load(input_dir / f"{stem}.npz")
        manifest = json.loads((input_dir / f"{stem}.json").read_text())
        for split in ("validation", "test"):
            predictions = archive[f"{split}_predictions"].astype(np.float64)
            y = archive[f"{split}_y"]
            grand = predictions.mean(axis=1)
            max_mean_difference = max(
                float(np.mean(np.sum((grand[a] - grand[b]) ** 2, axis=-1)))
                for a in range(3) for b in range(a + 1, 3)
            )
            for index, method in enumerate(manifest["methods"]):
                rows.append({
                    "dataset": dataset, "model": model, "split": split, "method": method,
                    "categorical_fields": manifest["categorical_fields"],
                    "binary_factors": manifest["binary_factors"],
                    "full_joint_cell_count": manifest["full_joint_cell_count"],
                    "maximum_grand_method_mean_squared_difference": max_mean_difference,
                    **METRICS.repetition_metrics(y, predictions[index]),
                })
    frame = pd.DataFrame(rows)
    test = frame[frame.split == "test"].pivot(
        index=["dataset", "model"], columns="method",
        values=["unbiased_between_repetition_risk", "mean_repetition_brier"],
    )
    risk = {method: test[("unbiased_between_repetition_risk", method)] for method in ("oa32", "marginal32", "iid32")}
    brier = {method: test[("mean_repetition_brier", method)] for method in ("oa32", "marginal32", "iid32")}
    wins_marginal = int((risk["oa32"] < risk["marginal32"]).sum())
    wins_iid = int((risk["oa32"] < risk["iid32"]).sum())
    wins_both = int(((risk["oa32"] < risk["marginal32"]) & (risk["oa32"] < risk["iid32"])).sum())
    dataset_results = {}
    for dataset in config["datasets"]:
        selection = test.loc[dataset]
        oa = selection[("unbiased_between_repetition_risk", "oa32")].mean()
        dataset_results[dataset] = {
            "oa_lower_than_marginal": bool(oa < selection[("unbiased_between_repetition_risk", "marginal32")].mean()),
            "oa_lower_than_iid": bool(oa < selection[("unbiased_between_repetition_risk", "iid32")].mean()),
        }
    summary = {
        "status": "complete", "cells": len(test),
        "cells_oa_lower_risk_than_marginal": wins_marginal,
        "cells_oa_lower_risk_than_iid": wins_iid,
        "cells_oa_lower_risk_than_both": wins_both,
        "two_sided_sign_p_vs_marginal": exact_sign_p(wins_marginal, len(test)),
        "two_sided_sign_p_vs_iid": exact_sign_p(wins_iid, len(test)),
        "pooled_risk_reduction_vs_marginal": float(1 - risk["oa32"].mean() / risk["marginal32"].mean()),
        "pooled_risk_reduction_vs_iid": float(1 - risk["oa32"].mean() / risk["iid32"].mean()),
        "cells_oa_lower_brier_than_marginal": int((brier["oa32"] < brier["marginal32"]).sum()),
        "cells_oa_lower_brier_than_iid": int((brier["oa32"] < brier["iid32"]).sum()),
        "mean_relative_brier_change_vs_marginal": float(((brier["oa32"] - brier["marginal32"]) / brier["marginal32"]).mean()),
        "mean_relative_brier_change_vs_iid": float(((brier["oa32"] - brier["iid32"]) / brier["iid32"]).mean()),
        "dataset_results": dataset_results,
        "maximum_grand_method_mean_squared_difference": float(frame.maximum_grand_method_mean_squared_difference.max()),
    }
    output = HERE / "results"
    frame.to_csv(output / "highdim_row_cover_metrics.csv", index=False)
    (output / "highdim_row_cover_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
