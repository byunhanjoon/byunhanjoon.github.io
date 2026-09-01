"""Equal-source uncertainty for 32/64-fit approximate log jackknives."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_robust_model_selection as RMS


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def interval(values: pd.Series, panel: str, metric: str):
    return RMS.cluster_interval(
        values.to_numpy(), RMS.stable_seed("log-jackknife-source", panel, metric)
    )


def main() -> None:
    cal32 = pd.read_csv(RESULTS / "log_quotient_jackknife_calibration.csv")
    cal64 = pd.read_csv(RESULTS / "log_jackknife_frontier_calibration.csv")
    cells64 = pd.read_csv(RESULTS / "log_jackknife_frontier_cells.csv")
    summary: dict[str, object] = {"status": "complete", "panels": {}}
    for panel in sorted(cal64.panel.unique()):
        c64 = cal64[cal64.panel == panel].groupby(["dataset", "method"]).score_rmse.mean().unstack()
        c32 = cal32[
            (cal32.panel == panel) & (cal32.method == "strength2_jackknife32")
        ].groupby("dataset").score_rmse.mean()
        cover_iid = c64.strength2_jackknife64 - c64.iid_jackknife64
        frontier = c64.strength2_jackknife64 - c32
        selection = cells64[cells64.panel == panel].pivot(
            index="dataset", columns="method", values="validation_log_quotient_regret"
        )
        regret = selection.strength2_jackknife64 - selection.iid_jackknife64
        record = {}
        for name, values in (
            ("cover_minus_iid64_rmse", cover_iid),
            ("cover64_minus_cover32_rmse", frontier),
            ("cover_minus_iid64_validation_regret", regret),
        ):
            ci = interval(values, panel, name)
            record[name] = {
                "equal_source_mean": float(values.mean()),
                "favorable_sources": int((values < 0).sum()),
                "tied_sources": int(np.isclose(values, 0, rtol=0, atol=1e-15).sum()),
                "sources": int(len(values)),
                "source_bootstrap_95_interval": ci,
                "interval_excludes_zero_favorably": bool(ci[1] < 0),
            }
        summary["panels"][panel] = record
    (RESULTS / "log_jackknife_uncertainty_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
