"""Downstream hyperparameter selection with equal-compute nuisance covers."""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_robust_model_selection as RMS
from analyze_strength2_cover import proper_loss


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
INPUT = RESULTS / "hpo_quotient"


def analyze_cell(path: Path) -> list[dict]:
    archive = np.load(path)
    manifest = json.loads(path.with_suffix(".json").read_text())
    validation_raw = archive["validation_predictions"].astype(np.float64)
    test_raw = archive["test_predictions"].astype(np.float64)
    val_y, test_y = archive["validation_y"], archive["test_y"]
    candidates = validation_raw.shape[0]
    nuisance_shape = tuple(int(value) for value in validation_raw.shape[1:5])
    validation = validation_raw.reshape((candidates, -1) + validation_raw.shape[-2:])
    test = test_raw.reshape((candidates, -1) + test_raw.shape[-2:])
    quotient_val = np.asarray([proper_loss(val_y, values.mean(axis=0)) for values in validation])
    quotient_test = np.asarray([proper_loss(test_y, values.mean(axis=0)) for values in test])
    winner = int(np.argmin(quotient_val))
    actions = RMS.action_ids(
        nuisance_shape, RMS.stable_seed("hpo-cover-selection", manifest["dataset"], manifest["family"])
    )
    rows = []
    for method, ids in actions.items():
        candidate_val = np.stack([RMS.batched_losses(val_y, values, ids) for values in validation], axis=1)
        selected = np.argmin(candidate_val, axis=1)
        candidate_test = np.stack([RMS.batched_losses(test_y, values, ids) for values in test], axis=1)
        rows.append({
            "dataset": manifest["dataset"], "family": manifest["family"], "method": method,
            "selection_agreement": float(np.mean(selected == winner)),
            "selection_entropy_bits": RMS.entropy(selected, candidates),
            "validation_quotient_regret": float(np.mean(quotient_val[selected] - quotient_val[winner])),
            "selected_quotient_test_loss": float(np.mean(quotient_test[selected])),
            "selected_realized_test_loss": float(np.mean(candidate_test[np.arange(RMS.DRAWS), selected])),
        })
    return rows


def main() -> None:
    rows = []
    for filename in sorted(glob.glob(str(INPUT / "*.npz"))):
        rows.extend(analyze_cell(Path(filename)))
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "hpo_cover_selection_cells.csv", index=False)
    loss = frame.pivot(index=["dataset", "family"], columns="method", values="selected_realized_test_loss")
    agreement = frame.groupby("method").selection_agreement.mean()
    comparisons = {}
    for control in RMS.METHODS[1:]:
        difference = loss.strength2 - loss[control]
        comparisons[control] = {
            "mean_strength2_minus_control_test_brier": float(difference.mean()),
            "cells_strength2_lower": int((difference < 0).sum()),
        }
    gate = (
        all(value["mean_strength2_minus_control_test_brier"] < 0 for value in comparisons.values())
        and comparisons["iid16"]["cells_strength2_lower"] >= 6
        and agreement.strength2 > agreement.iid16
    )
    summary = {
        "status": "complete", "cells": len(loss), "draws_per_cell": RMS.DRAWS,
        "comparisons": comparisons,
        "mean_selection_agreement": {method: float(agreement[method]) for method in RMS.METHODS},
        "mean_metrics_by_method": frame.groupby("method").mean(numeric_only=True).to_dict(orient="index"),
        "frozen_hpo_selection_gate_passed": bool(gate),
    }
    (RESULTS / "hpo_cover_selection_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

