#!/usr/bin/env python3
"""Numerically audit equivalence and non-oracle canonical coordinates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.basis_dependence import (  # noqa: E402
    anchor_canonical_representation,
    build_primary_representations,
    build_rbf_feature_matrix,
    load_dataset,
    oracle_inverse_representation,
    pca_canonical_representation,
    standardize_representation,
    whiten_representation,
)


def relative(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right) / max(np.linalg.norm(left), 1e-12))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", choices=["development", "prospective"], default="development")
    args = parser.parse_args()
    if args.panel == "prospective" and not (ROOT / "configs" / "FROZEN_METHOD_CONFIG.json").exists():
        raise RuntimeError("prospective coordinate audit locked until method freeze")
    config = yaml.safe_load((ROOT / "configs" / "development_protocol.yaml").read_text())
    panel = json.loads((ROOT / "configs" / "dataset_panel.json").read_text())
    equivalence_rows = []
    remedy_rows = []
    for spec in panel["datasets"]:
        if spec["panel"] != args.panel:
            continue
        data = load_dataset(spec, config)
        blocks = build_rbf_feature_matrix(data, config)
        reps = build_primary_representations(blocks, int(config["orbit_members"]))
        reference = reps[0]
        repair_builders = {
            "raw": lambda rep: rep,
            "standardization": standardize_representation,
            "whitening": whiten_representation,
            "pca_canonical": pca_canonical_representation,
            "anchor_canonical": lambda rep: anchor_canonical_representation(rep, data.key),
            "ORACLE INVERSE — NOT A METHOD": oracle_inverse_representation,
        }
        repaired_references = {name: builder(reference) for name, builder in repair_builders.items()}
        for rep in reps[1:]:
            for feature, record in rep.metadata["equivalence"].items():
                equivalence_rows.append({
                    "panel": args.panel, "dataset": data.key, "feature": feature,
                    "basis_a": "rbf", "basis_b": rep.variant, "scope": rep.scope, "member": rep.member,
                    "dimension": len(blocks.feature_blocks[feature]),
                    "condition_number": record["condition_number"],
                    "orthogonality_error": record["orthogonality_error"],
                    "reconstruction_error": record["reconstruction_error"],
                    "passes": record["reconstruction_error"] < 1e-6 and record["condition_number"] <= 3 + 1e-8,
                })
            for repair, builder in repair_builders.items():
                try:
                    repaired = builder(rep)
                    repaired_reference = repaired_references[repair]
                    train_difference = relative(repaired_reference.X_train, repaired.X_train)
                    test_difference = relative(repaired_reference.X_test, repaired.X_test)
                    anchor_records = repaired.metadata.get("anchor", {})
                    pca_records = repaired.metadata.get("pca", {})
                    remedy_rows.append({
                        "panel": args.panel, "dataset": data.key, "variant": rep.variant,
                        "scope": rep.scope, "member": rep.member, "repair": repair,
                        "train_relative_coordinate_difference": train_difference,
                        "test_relative_coordinate_difference": test_difference,
                        "output_dimension": repaired.X_train.shape[1],
                        "full_rank": all(record["full_rank"] for record in anchor_records.values()) if anchor_records else True,
                        "max_anchor_condition": max(
                            [record["anchor_condition_number"] for record in anchor_records.values()], default=np.nan
                        ),
                        "degenerate_block_count": sum(record["degenerate"] for record in pca_records.values()),
                        "passes_1e_5": test_difference < 1e-5,
                        "error": "",
                    })
                except Exception as error:
                    remedy_rows.append({
                        "panel": args.panel, "dataset": data.key, "variant": rep.variant,
                        "scope": rep.scope, "member": rep.member, "repair": repair,
                        "train_relative_coordinate_difference": np.nan,
                        "test_relative_coordinate_difference": np.nan, "output_dimension": np.nan,
                        "full_rank": False, "max_anchor_condition": np.nan,
                        "degenerate_block_count": np.nan, "passes_1e_5": False, "error": repr(error),
                    })
        print(f"[audited] {args.panel} {data.key}", flush=True)
    output = ROOT / "results" / "processed"
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(equivalence_rows).to_csv(output / f"equivalence_audit_{args.panel}.csv", index=False)
    pd.DataFrame(remedy_rows).to_csv(output / f"remedy_coordinate_audit_{args.panel}.csv", index=False)
    if args.panel == "development":
        pd.DataFrame(equivalence_rows).to_csv(output / "equivalence_audit.csv", index=False)
    summary = pd.DataFrame(remedy_rows).groupby(["repair", "variant"], dropna=False).agg(
        median_test_difference=("test_relative_coordinate_difference", "median"),
        max_test_difference=("test_relative_coordinate_difference", "max"),
        pass_rate=("passes_1e_5", "mean"), failures=("error", lambda values: sum(bool(value) for value in values)),
    )
    print(summary.to_string())


if __name__ == "__main__":
    main()
