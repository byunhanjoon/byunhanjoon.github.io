"""Analyze the frozen untouched classification HeteroBag placebo gate."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def main() -> None:
    config = json.loads((HERE / "heterobag_untouched_classification_config.json").read_text())
    frame = pd.read_csv(RESULTS / "heterobag_untouched_classification.csv")
    expected = {
        (dataset, model) for dataset in config["development_datasets"] for model in config["architectures"]
    }
    observed = set(zip(frame.dataset, frame.model))
    if len(frame) != len(expected) or observed != expected:
        raise RuntimeError(f"incomplete panel: rows={len(frame)}, missing={sorted(expected-observed)}")
    frame["semantic_minus_coordinate_pct"] = (
        frame.heterobag_relative_test_gain_vs_ttt_pct
        - frame.transformed_t_placebo_relative_test_gain_vs_ttt_pct
    )
    dataset = frame.groupby("dataset", as_index=False).agg(
        architectures=("model", "size"),
        semantic_mean_gain_pct=("heterobag_relative_test_gain_vs_ttt_pct", "mean"),
        coordinate_mean_gain_pct=("transformed_t_placebo_relative_test_gain_vs_ttt_pct", "mean"),
        homogeneous_q_mean_gain_pct=("alternate_homogeneous_relative_test_gain_vs_ttt_pct", "mean"),
        semantic_minus_coordinate_pct=("semantic_minus_coordinate_pct", "mean"),
        semantic_cell_wins=("heterobag_relative_test_gain_vs_ttt_pct", lambda values: int((values > 0).sum())),
    )
    rng = np.random.default_rng(2026082815)
    values = dataset.semantic_minus_coordinate_pct.to_numpy()
    boot = values[rng.integers(0, len(values), size=(100_000, len(values)))].mean(axis=1)
    clauses = {
        "positive_mean_semantic_minus_coordinate": float(values.mean()) > 0,
        "at_least_six_of_eight_dataset_means": int((values > 0).sum()) >= 6,
    }
    summary = {
        "status": "complete", "evidence_label": config["evidence_label"],
        "datasets": len(dataset), "cells": len(frame),
        "semantic_ttt_cell_wins": int((frame.heterobag_relative_test_gain_vs_ttt_pct > 0).sum()),
        "semantic_mean_gain_vs_ttt_pct": float(frame.heterobag_relative_test_gain_vs_ttt_pct.mean()),
        "coordinate_mean_gain_vs_ttt_pct": float(frame.transformed_t_placebo_relative_test_gain_vs_ttt_pct.mean()),
        "semantic_minus_coordinate_mean_pct": float(values.mean()),
        "datasets_semantic_exceeds_coordinate": int((values > 0).sum()),
        "dataset_bootstrap_semantic_minus_coordinate_95_ci_pct": [
            float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))
        ],
        "semantic_specific_gate_clauses": clauses,
        "semantic_specific_gate_passed": bool(all(clauses.values())),
    }
    frame.to_csv(RESULTS / "heterobag_untouched_cells.csv", index=False)
    dataset.to_csv(RESULTS / "heterobag_untouched_datasets.csv", index=False)
    (RESULTS / "heterobag_untouched_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
