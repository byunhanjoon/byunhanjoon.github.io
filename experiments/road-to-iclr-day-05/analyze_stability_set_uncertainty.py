"""Equal-source uncertainty for replicated-selector stability sets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_robust_model_selection as RMS


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
ACTION = "cover_cross_union64"
CONTROL = "iid_u_union64"
METRICS = {
    "validation_winner_covered": 1,
    "set_size": -1,
    "wrong_singleton": -1,
    "best_validation_regret": -1,
    "worst_validation_regret": -1,
    "test_winner_covered": 1,
}


def main() -> None:
    cells = pd.read_csv(RESULTS / "stability_set_cells.csv")
    summary: dict[str, object] = {"status": "complete", "panels": {}}
    for panel, current in cells.groupby("panel"):
        record = {}
        for metric, direction in METRICS.items():
            pivot = current.pivot(index="dataset", columns="method", values=metric)
            difference = pivot[ACTION] - pivot[CONTROL]
            interval = RMS.cluster_interval(
                difference.to_numpy(), RMS.stable_seed("stability-set-source", panel, metric)
            )
            record[metric] = {
                "cover_minus_iid_mean": float(difference.mean()),
                "favorable_sources": int((difference * direction > 0).sum()),
                "tied_sources": int(np.isclose(difference, 0, rtol=0, atol=1e-15).sum()),
                "source_bootstrap_95_interval": interval,
                "interval_excludes_zero_favorably": bool(
                    interval[0] > 0 if direction > 0 else interval[1] < 0
                ),
            }
        summary["panels"][panel] = record
    (RESULTS / "stability_set_uncertainty_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
