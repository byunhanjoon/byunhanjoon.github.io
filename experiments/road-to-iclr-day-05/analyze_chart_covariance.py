"""Aggregate the Day-5 repeated chart-covariance mechanism runs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
INPUT = HERE / "results" / "chart_covariance"
OUTPUT = HERE / "results"


def main() -> None:
    rows = []
    for path in sorted(INPUT.glob("*.json")):
        payload = json.loads(path.read_text())
        risk = payload["risk"]
        rows.append({
            "optimizer": payload["design"]["optimizer"],
            "seed": payload["design"]["seed"],
            "chart_schema_risk": risk["anova"]["total"],
            "mean_member_brier": risk["mean_member_brier"],
            "quotient_brier": risk["orbit_mean_brier"],
            "reference_brier": risk["reference_brier"],
            "hard_flip_fraction": risk["instance_audit"]["hard_label_flip_fraction"],
            "max_training_curve_range": payload["max_training_curve_range"],
            "identity_error": risk["risk_identity_absolute_error"],
        })
    frame = pd.DataFrame(rows)
    expected = 15
    if len(frame) != expected:
        raise RuntimeError(f"expected {expected} runs, found {len(frame)}")
    aggregate = frame.groupby("optimizer").agg(
        seeds=("seed", "count"),
        mean_schema_risk=("chart_schema_risk", "mean"),
        max_schema_risk=("chart_schema_risk", "max"),
        mean_member_brier=("mean_member_brier", "mean"),
        mean_quotient_brier=("quotient_brier", "mean"),
        mean_hard_flip_fraction=("hard_flip_fraction", "mean"),
        max_curve_range=("max_training_curve_range", "max"),
    ).reset_index()
    by_name = aggregate.set_index("optimizer")
    adam = float(by_name.loc["adamw", "mean_schema_risk"])
    summary = {
        "status": "complete",
        "runs": len(frame),
        "sgd_chart_risk_reduction_vs_adamw": 1.0 - float(by_name.loc["sgd", "mean_schema_risk"]) / adam,
        "field_vector_adam_chart_risk_reduction_vs_adamw": 1.0 - float(by_name.loc["field_vector_adam", "mean_schema_risk"]) / adam,
        "sgd_brier_change_vs_adamw": float(by_name.loc["sgd", "mean_member_brier"] - by_name.loc["adamw", "mean_member_brier"]),
        "field_vector_adam_brier_change_vs_adamw": float(by_name.loc["field_vector_adam", "mean_member_brier"] - by_name.loc["adamw", "mean_member_brier"]),
        "maximum_identity_error": float(frame.identity_error.max()),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT / "chart_covariance_runs.csv", index=False)
    aggregate.to_csv(OUTPUT / "chart_covariance_summary.csv", index=False)
    (OUTPUT / "chart_covariance_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

