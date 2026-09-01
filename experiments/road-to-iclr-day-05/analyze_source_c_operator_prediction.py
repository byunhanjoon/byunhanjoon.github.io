"""Post-outcome fANOVA prediction of source-C disjoint-pair residuals."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analyze_disjoint_pair_cross import graph_theory


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
sys.path.insert(0, str(HERE.parent / "road-to-iclr-idea-search"))
from orbit_anova import decompose  # noqa: E402


def main() -> None:
    config = json.loads((HERE / "openml_late_source_c_cover_config.json").read_text())
    directory = RESULTS / "openml_late_source_c_cover"
    theory = graph_theory((4, 4, 2, 4))
    single = theory["single_cover_coefficients"]
    packed = theory["disjoint_pair_mean_coefficients"]
    calibration = pd.read_csv(RESULTS / "late_source_c_packing_calibration.csv")
    current = calibration[
        (calibration.family == "pair32")
        & calibration.method.isin(("disjoint_pair_mean32", "independent_pair_mean32"))
    ].pivot(index=["dataset", "model"], columns="method", values="prediction_residual")

    rows = []
    for dataset in config["datasets"]:
        for model in config["models"]:
            archive = np.load(directory / f"{dataset}__{model}.npz")
            predictions = archive["validation_predictions"].astype(np.float64)
            components = decompose(predictions, ("feature", "category", "class", "seed"))
            predicted_control = sum(single[name] * float(value) / 2
                                    for name, value in components.items() if name in single)
            predicted_action = sum(packed[name] * float(value)
                                   for name, value in components.items() if name in packed)
            observed_control = float(current.loc[(dataset, model), "independent_pair_mean32"])
            observed_action = float(current.loc[(dataset, model), "disjoint_pair_mean32"])
            triple = sum(float(value) for name, value in components.items()
                         if name.count(":") == 2)
            four = sum(float(value) for name, value in components.items()
                       if name.count(":") == 3)
            rows.append({
                "dataset": dataset, "model": model,
                "triple_energy": triple, "four_way_energy": four,
                "triple_share_surviving": triple / (triple + four) if triple + four else 0.0,
                "predicted_control_residual": predicted_control,
                "predicted_packed_residual": predicted_action,
                "predicted_ratio": predicted_action / predicted_control
                if predicted_control > 0 else float("nan"),
                "observed_control_residual": observed_control,
                "observed_packed_residual": observed_action,
                "observed_ratio": observed_action / observed_control
                if observed_control > 0 else float("nan"),
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "source_c_operator_prediction_cells.csv", index=False)
    nondegenerate = frame[frame.predicted_control_residual > 1e-12].copy()
    ratio_error = nondegenerate.observed_ratio - nondegenerate.predicted_ratio
    rho, pvalue = spearmanr(
        nondegenerate.triple_share_surviving,
        1 - nondegenerate.observed_ratio,
    )
    summary = {
        "status": "complete",
        "evidence_status": "post_outcome_mechanism_diagnostic",
        "candidate_cells": int(len(frame)),
        "nondegenerate_cells": int(len(nondegenerate)),
        "predicted_ratio_range": [
            float(nondegenerate.predicted_ratio.min()),
            float(nondegenerate.predicted_ratio.max()),
        ],
        "mean_absolute_observed_minus_predicted_ratio": float(np.mean(np.abs(ratio_error))),
        "maximum_absolute_observed_minus_predicted_ratio": float(np.max(np.abs(ratio_error))),
        "spearman_triple_share_vs_observed_reduction": float(rho),
        "spearman_two_sided_p": float(pvalue),
        "exact_graph_coefficients": {
            "triple_packed_to_independent_pair_range": [
                float(min(packed[name] / (single[name] / 2)
                          for name in packed if name.count(":") == 2)),
                float(max(packed[name] / (single[name] / 2)
                          for name in packed if name.count(":") == 2)),
            ],
            "four_way_packed_to_independent_pair": float(
                packed["feature:category:class:seed"]
                / (single["feature:category:class:seed"] / 2)
            ),
        },
        "interpretation": (
            "fresh tensors follow the pre-existing graph/fANOVA residual law; "
            "correlation is diagnostic, not a frozen hypothesis test"
        ),
    }
    (RESULTS / "source_c_operator_prediction_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
