"""Validation-only selection of frozen broad-benchmark optimizer settings."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/day3/broad_benchmark"


def load() -> pd.DataFrame:
    paths = [RESULTS / f"calibration_{name}.csv" for name in ("adult", "california", "otto")]
    base = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    # The initial scoped Shampoo/SOAP diagnostic was superseded before broad
    # execution by full-model implementations. Preserve but exclude those rows.
    base = base[~base.remedy.isin(["shampoo", "soap"])]
    matrix_paths = [RESULTS / f"calibration_full_matrix_{name}.csv" for name in ("adult", "california", "otto")]
    matrix = pd.concat([pd.read_csv(path) for path in matrix_paths], ignore_index=True)
    return pd.concat([base, matrix], ignore_index=True)


def paired(frame: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "dataset",
        "task",
        "remedy",
        "learning_rate_requested",
        "ridge_requested",
        "precondition_frequency_requested",
    ]
    wide = frame.pivot_table(index=keys, columns="target_kappa", values="val_primary").reset_index()
    return wide.dropna(subset=[1.0, 1000.0])


def main() -> None:
    frame = load()
    failures = frame.failure.fillna("").ne("")
    valid = frame[~failures & frame.val_primary.notna()].copy()
    wide = paired(valid)
    # A task-specific scale makes AUC/accuracy differences and RMSE differences
    # commensurate without mixing their raw units.
    scales = {}
    for dataset, part in wide[wide.remedy.eq("adamw")].groupby("dataset"):
        task = part.task.iloc[0]
        scales[dataset] = 1.0 if task != "regression" else max(abs(float(part[1.0].max())), 1e-12)
    wide["scale"] = wide.dataset.map(scales)
    wide["sensitivity_normalized"] = (wide[1000.0] - wide[1.0]) / wide.scale

    adam = wide[wide.remedy.eq("adamw")]
    adam_scores = (
        adam.assign(score=(adam[1.0] + adam[1000.0]) / (2 * adam.scale))
        .groupby("learning_rate_requested")
        .score.mean()
    )
    selected_adam_lr = float(adam_scores.idxmax())
    baseline = adam[adam.learning_rate_requested.eq(selected_adam_lr)][
        ["dataset", 1.0, 1000.0]
    ].rename(columns={1.0: "baseline_k1", 1000.0: "baseline_k1000"})
    scored = wide.merge(baseline, on="dataset")
    scored["k1_gain_normalized"] = (scored[1.0] - scored.baseline_k1) / scored.scale
    scored["endpoint_gain_normalized"] = (scored[1000.0] - scored.baseline_k1000) / scored.scale
    settings = [
        "remedy",
        "learning_rate_requested",
        "ridge_requested",
        "precondition_frequency_requested",
    ]
    summary = (
        scored.groupby(settings)
        .agg(
            datasets=("dataset", "nunique"),
            k1_gain_normalized=("k1_gain_normalized", "mean"),
            sensitivity_normalized=("sensitivity_normalized", "mean"),
            endpoint_gain_normalized=("endpoint_gain_normalized", "mean"),
        )
        .reset_index()
    )
    selected = {}
    for remedy, part in summary.groupby("remedy"):
        eligible = part[part.k1_gain_normalized >= -0.01]
        candidates = eligible if len(eligible) else part
        row = candidates.sort_values(
            ["endpoint_gain_normalized", "sensitivity_normalized"], ascending=False
        ).iloc[0]
        selected[remedy] = {
            "learning_rate": float(row.learning_rate_requested),
            "ridge": float(row.ridge_requested),
            "precondition_frequency": int(row.precondition_frequency_requested),
            "calibration_datasets": int(row.datasets),
            "mean_k1_gain_normalized": float(row.k1_gain_normalized),
            "mean_sensitivity_normalized": float(row.sensitivity_normalized),
            "mean_endpoint_gain_normalized": float(row.endpoint_gain_normalized),
            "selection_constraint_met": bool(row.k1_gain_normalized >= -0.01),
        }
    payload = {
        "selection_data": "validation only",
        "adamw_reference_learning_rate": selected_adam_lr,
        "failed_calibration_runs": int(failures.sum()),
        "selected": selected,
    }
    wide.to_csv(RESULTS / "calibration_paired.csv", index=False)
    summary.to_csv(RESULTS / "calibration_summary.csv", index=False)
    (RESULTS / "selected_hyperparameters.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
