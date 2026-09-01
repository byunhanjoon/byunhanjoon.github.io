"""Analyze the frozen TabPFN v2.5 exact-schema support panel."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "road-to-iclr-idea-search"))
from orbit_anova import risk_summary  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=HERE / "results" / "tabpfn")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    args = parser.parse_args()
    datasets = (
        "australian_credit_approval", "bank_marketing_subscription",
        "german_credit_risk", "lendingclub_loan_default",
    )
    settings = ((1, "none"), (1, "default"), (8, "none"), (8, "default"))
    rows = []
    missing = []
    for dataset in datasets:
        path = args.input_dir / f"{dataset}.npz"
        if not path.exists():
            missing.append(dataset)
            continue
        archive = np.load(path)
        for estimators, policy in settings:
            for split in ("validation", "test"):
                predictions = archive[f"{split}__{estimators}__{policy}"].astype(np.float64)
                y = archive[f"{split}_y"]
                summary = risk_summary(predictions, y, ("feature", "category", "class"))
                row = {
                    "dataset": dataset,
                    "split": split,
                    "estimators": estimators,
                    "internal_shift_policy": policy,
                    "schema_risk": summary["anova"]["total"],
                    "mean_member_brier": summary["mean_member_brier"],
                    "quotient_brier": summary["orbit_mean_brier"],
                    "reference_brier": summary["reference_brier"],
                    "hard_flip_fraction": summary["instance_audit"]["hard_label_flip_fraction"],
                    "identity_error": summary["risk_identity_absolute_error"],
                }
                for key, value in summary["anova"].items():
                    if key not in {"component_sum_error", "prediction_reconstruction_max_error"}:
                        row[f"anova_{key}"] = value
                rows.append(row)
    if missing:
        raise RuntimeError(f"missing datasets: {missing}")
    frame = pd.DataFrame(rows)
    test = frame[frame.split == "test"]
    baseline = test[(test.estimators == 1) & (test.internal_shift_policy == "none")].set_index("dataset")
    default8 = test[(test.estimators == 8) & (test.internal_shift_policy == "default")].set_index("dataset")
    joined = baseline[["schema_risk", "reference_brier"]].join(
        default8[["schema_risk", "reference_brier"]], lsuffix="_baseline", rsuffix="_default8"
    )
    joined["schema_risk_reduction"] = 1 - joined.schema_risk_default8 / joined.schema_risk_baseline
    joined["relative_reference_brier_change"] = (
        joined.reference_brier_default8 - joined.reference_brier_baseline
    ) / joined.reference_brier_baseline
    summary = {
        "status": "complete",
        "datasets": len(datasets),
        "settings": len(settings),
        "datasets_where_default8_reduces_schema_risk": int((joined.schema_risk_reduction > 0).sum()),
        "mean_default8_schema_risk_reduction": float(joined.schema_risk_reduction.mean()),
        "mean_default8_relative_reference_brier_change": float(joined.relative_reference_brier_change.mean()),
        "maximum_identity_error": float(frame.identity_error.max()),
        "default8_exactly_closed_below_1e_12": int((joined.schema_risk_default8 < 1e-12).sum()),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "tabpfn_schema_summary.csv", index=False)
    joined.reset_index().to_csv(args.output_dir / "tabpfn_default8_comparison.csv", index=False)
    (args.output_dir / "tabpfn_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

