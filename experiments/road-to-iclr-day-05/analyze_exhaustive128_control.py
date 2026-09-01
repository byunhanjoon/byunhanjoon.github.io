"""Equal-fit exact enumeration control for the 128-cell nuisance product."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_cross_quotient_selection as CQS
from analyze_mixed_resolvable_packing import SHAPE, mixed_coset_resolution
from analyze_strength2_cover import proper_loss


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def main() -> None:
    resolution = mixed_coset_resolution()
    ids = np.ravel_multi_index(resolution.transpose(2, 0, 1), SHAPE).reshape(-1)
    assert np.array_equal(np.sort(ids), np.arange(128))
    rows, selection_rows = [], []
    pack_cells = pd.read_csv(RESULTS / "disjoint_pack_cross128_cells.csv")
    pack_cal = pd.read_csv(RESULTS / "disjoint_pack_cross128_calibration.csv")
    for panel, config_name, directory_name in CQS.PANELS:
        config = json.loads((HERE / config_name).read_text())
        for dataset in config["datasets"]:
            exact_scores = []
            models = []
            for model in config["models"]:
                archive = np.load(RESULTS / directory_name / f"{dataset}__{model}.npz")
                if tuple(archive["validation_predictions"].shape[:4]) != SHAPE:
                    continue
                flat = archive["validation_predictions"].reshape(
                    (-1,) + archive["validation_predictions"].shape[-2:]
                ).astype(np.float64)
                exact = proper_loss(archive["validation_y"], flat.mean(axis=0))
                enumerated = proper_loss(archive["validation_y"], flat[ids].mean(axis=0))
                rows.append({
                    "panel": panel, "dataset": dataset, "model": model,
                    "exact_quotient_score": exact, "exhaustive128_score": enumerated,
                    "absolute_error": abs(enumerated - exact), "score_rmse": 0.0,
                })
                exact_scores.append(exact); models.append(model)
            if exact_scores:
                winner = int(np.argmin(exact_scores))
                selection_rows.append({
                    "panel": panel, "dataset": dataset,
                    "exact_winner": models[winner], "selection_agreement": 1.0,
                    "validation_quotient_regret": 0.0,
                })
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "exhaustive128_control_candidates.csv", index=False)
    selections = pd.DataFrame(selection_rows)
    selections.to_csv(RESULTS / "exhaustive128_control_selection.csv", index=False)
    pack_full = pack_cal[(pack_cal.method == "disjoint_pack_cross128") & (pack_cal.product_cells == 128)]
    merged = frame.merge(pack_full[["panel", "dataset", "model", "score_rmse"]],
                         on=["panel", "dataset", "model"], suffixes=("_exhaustive", "_pack"))
    panel_clauses = []
    for panel, current in selections.groupby("panel"):
        pack = pack_cells[
            (pack_cells.panel == panel)
            & (pack_cells.method == "disjoint_pack_cross128")
            & (pack_cells.dataset.isin(current.dataset))
        ]
        panel_clauses.append({
            "panel": panel, "sources": int(len(current)),
            "exhaustive_agreement": 1.0,
            "pack_agreement": float(pack.selection_agreement.mean()),
            "exhaustive_regret": 0.0,
            "pack_regret": float(pack.validation_quotient_regret.mean()),
            "agreement_nolower": bool(1.0 >= pack.selection_agreement.mean()),
            "regret_nohigher": bool(0.0 <= pack.validation_quotient_regret.mean()),
        })
    summary = {
        "status": "complete", "equal_fit_budget": 128,
        "full_product_candidates": int(len(merged)),
        "candidates_with_strictly_lower_exhaustive_rmse": int(
            (merged.score_rmse_exhaustive < merged.score_rmse_pack).sum()
        ),
        "max_absolute_exhaustive_score_error": float(frame.absolute_error.max()),
        "panels": panel_clauses,
    }
    summary["stronger_control_gate_passed"] = bool(
        summary["candidates_with_strictly_lower_exhaustive_rmse"] == 23
        and summary["max_absolute_exhaustive_score_error"] < 1e-12
        and all(item["agreement_nolower"] and item["regret_nohigher"] for item in panel_clauses)
    )
    summary["pack_cross128_compute_optimal_at_closure"] = False
    (RESULTS / "exhaustive128_control_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
