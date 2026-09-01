"""Audit the prospective source-C gates without changing frozen thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
STRICT_TOLERANCE = 1e-15
NONDEGENERATE_TOLERANCE = 1e-12
COMPARISONS = {
    "pair32": ("pair32", "disjoint_pair_mean32", "independent_pair_mean32"),
    "pack64": ("pack64", "mutually_disjoint_pack64", "two_disjoint_pairs64"),
    "unbiased_pair_cross64": (
        "pair_cross64", "disjoint_pair_cross64", "independent_block_u64"
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-prefix", default="late_source_c")
    parser.add_argument("--tensor-dir", default="openml_late_source_c_cover")
    parser.add_argument("--evidence-status", default="prospective_sources_and_outcomes")
    args = parser.parse_args()
    extension = json.loads((RESULTS / f"{args.output_prefix}_extension_summary.json").read_text())
    calibration = pd.read_csv(RESULTS / f"{args.output_prefix}_packing_calibration.csv")
    strength = extension["primary_strength2_gate"]
    material_fraction = (
        strength["material_cell_wins_vs_all_controls"] / strength["material_cells"]
        if strength["material_cells"] else float("nan")
    )
    strength_gate = bool(
        strength["all_cell_wins_vs_iid_and_strength1"] >= 16
        and strength["source_mean_wins_vs_iid_and_strength1"] == 4
        and material_fraction >= .8
    )

    comparisons = {}
    for name, (family, action, control) in COMPARISONS.items():
        current = calibration[
            (calibration.family == family) & calibration.method.isin((action, control))
        ]
        pivot = current.pivot(
            index=["dataset", "model"], columns="method", values="score_rmse"
        )
        difference = pivot[control] - pivot[action]
        nondegenerate = pivot[control] > NONDEGENERATE_TOLERANCE
        source = current.groupby(["dataset", "method"]).score_rmse.mean().unstack()
        nondegenerate_wins = int((difference[nondegenerate] > STRICT_TOLERANCE).sum())
        nondegenerate_losses = int((difference[nondegenerate] < -STRICT_TOLERANCE).sum())
        nondegenerate_total = int(nondegenerate.sum())
        win_fraction = (
            nondegenerate_wins / nondegenerate_total
            if nondegenerate_total else float("nan")
        )
        source_wins = int((source[action] < source[control] - STRICT_TOLERANCE).sum())
        relative = 1 - pivot.loc[nondegenerate, action] / pivot.loc[nondegenerate, control]
        passed = bool(
            nondegenerate_total > 0
            and win_fraction >= .8
            and nondegenerate_losses == 0
            and source_wins == 4
        )
        comparisons[name] = {
            "candidates": int(len(pivot)),
            "nondegenerate_candidates": nondegenerate_total,
            "nondegenerate_wins": nondegenerate_wins,
            "nondegenerate_losses": nondegenerate_losses,
            "nondegenerate_win_fraction": float(win_fraction),
            "mean_nondegenerate_relative_score_rmse_reduction": float(relative.mean()),
            "source_mean_wins": source_wins,
            "sources": int(len(source)),
            "frozen_gate_passed": passed,
        }

    manifests = [
        json.loads(path.read_text())
        for path in sorted((RESULTS / args.tensor_dir).glob("*.json"))
    ]
    summary = {
        "status": "complete",
        "evidence_status": args.evidence_status,
        "complete_tensors": len(manifests),
        "represented_complete_product_fits": int(sum(row["fits"] for row in manifests)),
        "strength2": {
            **strength,
            "material_win_fraction": float(material_fraction),
            "complete_frozen_gate_passed": strength_gate,
        },
        "packing": comparisons,
        "all_packing_frozen_gates_passed": bool(all(
            row["frozen_gate_passed"] for row in comparisons.values()
        )),
        "validation_test_winner_agreements": int(
            extension["exact_validation_to_test_transfer"]["winner_agreements"]
        ),
        "mean_validation_selected_test_regret": float(
            extension["exact_validation_to_test_transfer"]
            ["mean_test_regret_of_validation_winner"]
        ),
        "median_end_to_end_seconds_per_128_fit_tensor": float(np.median([
            row["timing"]["end_to_end_seconds"] for row in manifests
        ])),
        "all_frozen_gates_passed": bool(
            strength_gate and all(row["frozen_gate_passed"] for row in comparisons.values())
        ),
    }
    (RESULTS / f"{args.output_prefix}_audit_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
