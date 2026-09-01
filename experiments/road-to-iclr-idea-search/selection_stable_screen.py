"""Prospective split screen for cells stable on the baseline validation split."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from selection_split_confirmation import exact_binomial_sign_p, exact_sign_flip_p


HERE = Path(__file__).resolve().parent
ROOT = HERE / "selection_split_stable_screen"
SEEDS = tuple(range(20_260_827, 20_260_834))
CASES = (
    ("adult", "ordinal_forest"),
    ("adult", "native_histgb"),
    ("churn", "native_histgb"),
)


def main() -> None:
    output = {
        "status": "prospective_screen_of_three_baseline_stable_binary_cells",
        "baseline_split_seed": 20_260_826,
        "prospective_split_seeds": list(SEEDS),
        "cases": {},
    }
    for dataset, family in CASES:
        records = []
        for seed in SEEDS:
            path = ROOT / f"{dataset}_{family}_s{seed}.json"
            data = json.loads(path.read_text())
            records.append(
                {
                    "split_seed": seed,
                    "identity_config": data["selection_protocol"][
                        "canonical_identity_selected_config"
                    ],
                    "selection_counts": data["selection_counts"],
                    "fraction_different_from_identity": data[
                        "fraction_different_from_identity_config"
                    ],
                    "schema_risk_difference": data[
                        "selection_minus_frozen_schema_risk"
                    ],
                    "orbit_mean_brier_difference": data[
                        "selection_minus_frozen_orbit_mean_brier"
                    ],
                }
            )
        risk = np.asarray([record["schema_risk_difference"] for record in records])
        brier = np.asarray(
            [record["orbit_mean_brier_difference"] for record in records]
        )
        output["cases"][f"{dataset}/{family}"] = {
            "records": records,
            "selection_unstable_splits": int(
                sum(record["fraction_different_from_identity"] > 0 for record in records)
            ),
            "higher_schema_risk_splits": int(np.sum(risk > 0)),
            "lower_schema_risk_splits": int(np.sum(risk < 0)),
            "mean_schema_risk_difference": float(risk.mean()),
            "schema_risk_difference_range": [float(risk.min()), float(risk.max())],
            "schema_risk_difference_exact_sign_flip_p": exact_sign_flip_p(risk),
            "schema_risk_difference_exact_binomial_sign_p": exact_binomial_sign_p(
                risk
            ),
            "mean_orbit_brier_difference": float(brier.mean()),
            "orbit_brier_difference_range": [float(brier.min()), float(brier.max())],
            "orbit_brier_difference_exact_sign_flip_p": exact_sign_flip_p(brier),
            "orbit_brier_difference_exact_binomial_sign_p": exact_binomial_sign_p(
                brier
            ),
        }
    destination = HERE / "selection_stable_screen.json"
    destination.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
