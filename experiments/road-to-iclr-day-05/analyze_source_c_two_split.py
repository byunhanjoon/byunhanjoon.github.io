"""Descriptive two-partition roll-up for the source-C panel."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def main() -> None:
    audits = [
        json.loads((RESULTS / "late_source_c_audit_summary.json").read_text()),
        json.loads((RESULTS / "late_source_c_split_audit_summary.json").read_text()),
    ]
    names = ("pair32", "pack64", "unbiased_pair_cross64")
    original_dir = RESULTS / "openml_late_source_c_cover"
    alternate_dir = RESULTS / "openml_late_source_c_split_cover"
    tensor_pairs = [(path, alternate_dir / path.name) for path in sorted(original_dir.glob("*.npz"))]
    distinct_pairs = sum(
        hashlib.sha256(left.read_bytes()).digest() != hashlib.sha256(right.read_bytes()).digest()
        for left, right in tensor_pairs
    )
    packing = {}
    for name in names:
        rows = [audit["packing"][name] for audit in audits]
        packing[name] = {
            "nondegenerate_wins": sum(row["nondegenerate_wins"] for row in rows),
            "nondegenerate_candidates": sum(row["nondegenerate_candidates"] for row in rows),
            "nondegenerate_losses": sum(row["nondegenerate_losses"] for row in rows),
            "dataset_split_source_mean_wins": sum(row["source_mean_wins"] for row in rows),
            "dataset_split_source_means": sum(row["sources"] for row in rows),
            "equal_split_mean_relative_score_rmse_reduction": sum(
                row["mean_nondegenerate_relative_score_rmse_reduction"] for row in rows
            ) / 2,
        }
    summary = {
        "status": "complete",
        "evidence_status": "descriptive_two_split_rollup_second_split_conditionally_frozen",
        "unique_sources": 4,
        "dataset_split_pairs": 8,
        "complete_tensors": sum(audit["complete_tensors"] for audit in audits),
        "represented_complete_product_fits": sum(
            audit["represented_complete_product_fits"] for audit in audits
        ),
        "artifact_independence_check": {
            "paired_tensor_artifacts": len(tensor_pairs),
            "byte_distinct_tensor_pairs": distinct_pairs,
            "effective_split_seeds": [2026083041, 2026083051],
            "passed": bool(len(tensor_pairs) == distinct_pairs == 20),
        },
        "strength2": {
            "literal_wins": sum(audit["strength2"]["all_cell_wins_vs_iid_and_strength1"] for audit in audits),
            "literal_cells": sum(audit["strength2"]["all_cells"] for audit in audits),
            "material_wins": sum(audit["strength2"]["material_cell_wins_vs_all_controls"] for audit in audits),
            "material_cells": sum(audit["strength2"]["material_cells"] for audit in audits),
            "dataset_split_source_mean_wins": sum(audit["strength2"]["source_mean_wins_vs_iid_and_strength1"] for audit in audits),
            "dataset_split_source_means": 8,
        },
        "packing": packing,
        "exact_validation_test_transfer": {
            "winner_agreements": sum(audit["validation_test_winner_agreements"] for audit in audits),
            "dataset_split_pairs": 8,
            "equal_split_mean_test_regret": sum(
                audit["mean_validation_selected_test_regret"] for audit in audits
            ) / 2,
        },
        "interpretation": "nuisance gains transport across both splits; winner transfer does not",
    }
    (RESULTS / "source_c_two_split_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
