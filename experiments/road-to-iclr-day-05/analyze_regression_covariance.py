"""Aggregate the frozen multi-seed regression covariance confirmation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
INPUT = HERE / "results" / "regression_covariance"
OUTPUT = HERE / "results"


def main() -> None:
    rows = []
    for path in sorted(INPUT.glob("*.json")):
        payload = json.loads(path.read_text())
        dataset = payload["design"]["dataset"]
        seed = payload["covariant_sgd"]["seed"]
        raw = payload["raw_adamw"]["summary"]
        for optimizer, key in (("sgd", "covariant_sgd"), ("field_vector_adam", "field_vector_adam")):
            current = payload[key]
            summary = current["summary"]
            rows.append({
                "dataset": dataset,
                "seed": seed,
                "optimizer": optimizer,
                "chart_risk": summary["anova"]["total"],
                "mean_member_mse": summary["mean_member_mse_standardized"],
                "quotient_mse": summary["orbit_mean_mse_standardized"],
                "raw_adam_mean_member_mse": raw["mean_member_mse_standardized"],
                "raw_adam_quotient_mse": raw["orbit_mean_mse_standardized"],
                "mse_change_vs_raw_adam_member": summary["mean_member_mse_standardized"] - raw["mean_member_mse_standardized"],
                "mse_change_vs_raw_adam_quotient": summary["orbit_mean_mse_standardized"] - raw["orbit_mean_mse_standardized"],
                "max_training_curve_range": current["max_training_curve_range"],
                "identity_error": summary["risk_identity_absolute_error"],
            })
    frame = pd.DataFrame(rows)
    if len(frame) != 12:
        raise RuntimeError(f"expected 12 rows, got {len(frame)}")
    aggregate = frame.groupby(["dataset", "optimizer"]).agg(
        seeds=("seed", "count"),
        mean_chart_risk=("chart_risk", "mean"),
        max_chart_risk=("chart_risk", "max"),
        mean_member_mse=("mean_member_mse", "mean"),
        mean_mse_change_vs_raw_adam_member=("mse_change_vs_raw_adam_member", "mean"),
        mean_mse_change_vs_raw_adam_quotient=("mse_change_vs_raw_adam_quotient", "mean"),
        maximum_curve_range=("max_training_curve_range", "max"),
    ).reset_index()
    summary = {
        "status": "complete",
        "runs": len(frame),
        "dataset_optimizer_cells": len(aggregate),
        "cells_closed_below_1e_10": int((aggregate.max_chart_risk < 1e-10).sum()),
        "cells_better_than_raw_adam_member_mse": int((aggregate.mean_mse_change_vs_raw_adam_member < 0).sum()),
        "cells_better_than_raw_adam_quotient_mse": int((aggregate.mean_mse_change_vs_raw_adam_quotient < 0).sum()),
        "maximum_identity_error": float(frame.identity_error.max()),
    }
    frame.to_csv(OUTPUT / "regression_covariance_runs.csv", index=False)
    aggregate.to_csv(OUTPUT / "regression_covariance_summary.csv", index=False)
    (OUTPUT / "regression_covariance_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
