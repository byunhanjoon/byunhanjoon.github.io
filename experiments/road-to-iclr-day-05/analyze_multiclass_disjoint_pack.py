"""Vector-Brier scope for disjoint pair32 and exhaustive pack64."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from analyze_disjoint_pair32 import analyze_dataset as analyze_pair
from analyze_disjoint_pack64 import analyze_dataset as analyze_pack


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def main() -> None:
    config = json.loads((HERE / "openml_multiclass_cover_config.json").read_text())
    directory = RESULTS / "openml_multiclass_cover"
    pair_rows, pair_cal, pack_rows, pack_cal = [], [], [], []
    for dataset in config["datasets"]:
        rows, calibration = analyze_pair("multiclass", dataset, config["models"], directory)
        pair_rows.extend(rows); pair_cal.extend(calibration)
        rows, calibration = analyze_pack("multiclass", dataset, config["models"], directory)
        pack_rows.extend(rows); pack_cal.extend(calibration)
    pair_cal = pd.DataFrame(pair_cal); pack_cal = pd.DataFrame(pack_cal)
    pair_cells = pd.DataFrame(pair_rows).groupby(
        ["dataset", "method"], as_index=False
    ).mean(numeric_only=True)
    pack_cells = pd.DataFrame(pack_rows).groupby(
        ["dataset", "method"], as_index=False
    ).mean(numeric_only=True)
    pair_cal.to_csv(RESULTS / "multiclass_disjoint_pair_calibration.csv", index=False)
    pack_cal.to_csv(RESULTS / "multiclass_disjoint_pack_calibration.csv", index=False)
    pair_cells.to_csv(RESULTS / "multiclass_disjoint_pair_cells.csv", index=False)
    pack_cells.to_csv(RESULTS / "multiclass_disjoint_pack_cells.csv", index=False)

    pair = pair_cal.pivot(index=["dataset", "model"], columns="method",
                          values=["score_rmse", "prediction_residual"])
    rmse_wins = int((pair[("score_rmse", "disjoint_pair_mean32")]
                     < pair[("score_rmse", "independent_pair_mean32")]).sum())
    residual_wins = int((pair[("prediction_residual", "disjoint_pair_mean32")]
                         < pair[("prediction_residual", "independent_pair_mean32")]).sum())
    packed = pack_cal[pack_cal.method == "mutually_disjoint_pack64"]
    max_error = float(packed.max_absolute_score_error.max())
    summary = {
        "status": "complete", "candidates": len(pair),
        "pair32_rmse_wins": rmse_wins,
        "pair32_prediction_residual_wins": residual_wins,
        "pack64_max_absolute_quotient_score_error": max_error,
        "pair32_panel_means": pair_cal.groupby("method").mean(numeric_only=True).reset_index().to_dict(orient="records"),
        "pair32_selection": pair_cells.groupby("method").mean(numeric_only=True).reset_index().to_dict(orient="records"),
        "pack64_selection": pack_cells.groupby("method").mean(numeric_only=True).reset_index().to_dict(orient="records"),
        "frozen_gate_passed": bool(
            rmse_wins == len(pair) and residual_wins == len(pair) and max_error < 1e-12
        ),
    }
    (RESULTS / "multiclass_disjoint_pack_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
