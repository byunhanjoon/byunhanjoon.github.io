"""Apply the predeclared validation gate to field-local distillation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RAW_PATTERN = "field_local_distillation_*_selection.csv"


def read_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    paths = [
        path for path in sorted(RESULTS.glob(RAW_PATTERN))
        if path.name != "field_local_distillation_comparisons.csv"
    ]
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    keys = [(row["dataset"], row["model"], row["seed"], row["method"]) for row in rows]
    if len(keys) != 30 or len(set(keys)) != 30:
        raise ValueError(f"expected 30 unique selection cells, found {len(keys)}/{len(set(keys))}")
    if any(row["test_evaluated"].lower() != "false" for row in rows):
        raise ValueError("selection artifact contains a test-evaluated row")
    return rows


def main() -> None:
    rows = read_rows()
    by_key = {
        (row["dataset"], row["model"], int(row["seed"]), row["method"]): row
        for row in rows
    }
    comparisons: list[dict[str, object]] = []
    gated_cells: list[dict[str, object]] = []
    for dataset in ("weather", "cooking-time"):
        for model in ("mlp", "resnet", "ft_transformer"):
            def value(method: str) -> float:
                return float(by_key[(dataset, model, 20260827, method)]["val_rmse"])

            ple = value("ple")
            adapter = value("ple_adapter")
            noalign = value("semantic_noalign")
            semantic = value("semantic_local")
            wrong = value("semantic_wrong_local")
            clears = semantic < ple and semantic < adapter and semantic < wrong
            gated_cells.append({
                "dataset": dataset,
                "model": model,
                "ple_val_rmse": ple,
                "ple_adapter_val_rmse": adapter,
                "semantic_noalign_val_rmse": noalign,
                "semantic_local_val_rmse": semantic,
                "semantic_wrong_val_rmse": wrong,
                "semantic_gain_vs_ple_pct": 100.0 * (ple - semantic) / ple,
                "semantic_gain_vs_adapter_pct": 100.0 * (adapter - semantic) / adapter,
                "semantic_gain_vs_wrong_pct": 100.0 * (wrong - semantic) / wrong,
                "clears_cell_gate": clears,
            })
            for method, rmse in (
                ("ple_adapter", adapter),
                ("semantic_noalign", noalign),
                ("semantic_local", semantic),
                ("semantic_wrong_local", wrong),
            ):
                source = by_key[(dataset, model, 20260827, method)]
                comparisons.append({
                    "dataset": dataset,
                    "model": model,
                    "seed": 20260827,
                    "method": method,
                    "ple_val_rmse": ple,
                    "method_val_rmse": rmse,
                    "gain_vs_ple_pct": 100.0 * (ple - rmse) / ple,
                    "parameters": int(source["parameters"]),
                    "mean_abs_gate": float(source["mean_abs_gate"]),
                    "max_abs_gate": float(source["max_abs_gate"]),
                    "test_evaluated": False,
                })

    fields = list(comparisons[0])
    with (RESULTS / "field_local_distillation_comparisons.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(comparisons)

    clears = sum(bool(row["clears_cell_gate"]) for row in gated_cells)
    gains = np.asarray([float(row["semantic_gain_vs_ple_pct"]) for row in gated_cells])
    control_gaps = np.asarray(
        [float(row["semantic_gain_vs_wrong_pct"]) for row in gated_cells]
    )
    promoted = clears >= 5
    summary = {
        "status": "promote_to_multiseed_and_delivery_eta" if promoted else "stop_branch",
        "test_metrics_computed": False,
        "predeclared_gate": {
            "rule": (
                "semantic_local must have lower validation RMSE than PLE, the "
                "parameter-matched PLE adapter, and semantic_wrong_local in at "
                "least 5/6 Weather/Cooking architecture cells"
            ),
            "required_cells": 5,
            "cleared_cells": clears,
            "passed": promoted,
        },
        "semantic_local": {
            "wins_vs_ple": int(np.sum(gains > 0)),
            "mean_gain_vs_ple_pct": float(gains.mean()),
            "wins_vs_wrong_geometry": int(np.sum(control_gaps > 0)),
            "mean_gain_vs_wrong_geometry_pct": float(control_gaps.mean()),
        },
        "cells": gated_cells,
        "decision": (
            "Run three seeds and transfer unchanged to Delivery ETA."
            if promoted
            else (
                "Do not inspect Weather/Cooking test metrics, do not transfer to "
                "Delivery ETA, and stop the field-local contrastive branch."
            )
        ),
    }
    (RESULTS / "field_local_distillation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"Field-local gate: {clears}/6 cells; "
        f"decision={summary['status']}; test metrics computed=false"
    )


if __name__ == "__main__":
    main()
