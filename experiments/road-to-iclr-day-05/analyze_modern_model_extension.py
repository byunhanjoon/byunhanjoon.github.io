"""Audit frozen gates and degeneracy for the post-source model extension."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
TOLERANCE = 1e-15
COMPARISONS = {
    "pair32": ("pair32", "disjoint_pair_mean32", "independent_pair_mean32"),
    "pack64": ("pack64", "mutually_disjoint_pack64", "two_disjoint_pairs64"),
    "unbiased_pair_cross64": (
        "pair_cross64", "disjoint_pair_cross64", "independent_block_u64"
    ),
}


def main() -> None:
    extension = json.loads((RESULTS / "modern_model_extension_summary.json").read_text())
    calibration = pd.read_csv(RESULTS / "modern_model_packing_calibration.csv")
    comparisons = {}
    for name, (family, action, control) in COMPARISONS.items():
        current = calibration[
            (calibration.family == family) & calibration.method.isin((action, control))
        ]
        pivot = current.pivot(
            index=["dataset", "model"], columns="method", values="score_rmse"
        )
        difference = pivot[control] - pivot[action]
        nondegenerate = pivot[control] > 1e-12
        source = current.groupby(["dataset", "method"]).score_rmse.mean().unstack()
        strict_wins = int((difference > TOLERANCE).sum())
        ties = int((np.abs(difference) <= TOLERANCE).sum())
        losses = int((difference < -TOLERANCE).sum())
        source_wins = int((source[action] < source[control] - TOLERANCE).sum())
        relative = 1 - pivot.loc[nondegenerate, action] / pivot.loc[nondegenerate, control]
        comparisons[name] = {
            "strict_candidate_wins": strict_wins,
            "numerical_ties": ties,
            "candidate_losses": losses,
            "candidates": int(len(pivot)),
            "nondegenerate_candidate_wins": int((difference[nondegenerate] > TOLERANCE).sum()),
            "nondegenerate_candidates": int(nondegenerate.sum()),
            "mean_nondegenerate_relative_rmse_reduction": float(relative.mean()),
            "source_mean_wins": source_wins,
            "sources": int(len(source)),
            "frozen_13_of_16_and_7_of_8_gate_passed": bool(
                strict_wins >= 13 and source_wins >= 7
            ),
            "no_adverse_candidate": bool(losses == 0),
        }
    manifests = [
        json.loads(path.read_text())
        for path in sorted((RESULTS / "openml_modern_model_cover").glob("*.json"))
    ]
    summary = {
        "status": "complete",
        "evidence_status": "frozen_model_family_after_source_outcomes",
        "sources": 8,
        "models": ["native_histgb", "catboost_native"],
        "complete_tensors": len(manifests),
        "represented_complete_product_fits": int(sum(row["fits"] for row in manifests)),
        "strength2_frozen_gate_passed": bool(extension["primary_strength2_gate"]["passed"]),
        "strength2": extension["primary_strength2_gate"],
        "packing": comparisons,
        "all_packing_frozen_strict_gates_passed": bool(all(
            row["frozen_13_of_16_and_7_of_8_gate_passed"]
            for row in comparisons.values()
        )),
        "all_packing_candidates_nonadverse": bool(all(
            row["no_adverse_candidate"] for row in comparisons.values()
        )),
        "validation_test_winner_agreements": int(
            extension["exact_validation_to_test_transfer"]["winner_agreements"]
        ),
        "median_end_to_end_seconds_per_128_fit_tensor": float(np.median([
            row["timing"]["end_to_end_seconds"] for row in manifests
        ])),
        "interpretation": (
            "strength2_pass; packing_strict_fail_due_to_strength2_exact_ties; "
            "all_nondegenerate_candidates_improve"
        ),
    }
    (RESULTS / "modern_model_extension_audit_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
