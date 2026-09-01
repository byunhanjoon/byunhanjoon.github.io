"""Tail proper-loss of validation-selected nuisance ensembles."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
METHODS = ("strength2", "iid16", "four_strength1", "four_seed_blocks")


def summarize(values: pd.Series) -> pd.Series:
    q95 = float(values.quantile(0.95))
    return pd.Series({
        "loss_std": float(values.std(ddof=0)),
        "loss_q90": float(values.quantile(0.90)),
        "loss_q95": q95,
        "loss_cvar95": float(values[values >= q95].mean()),
    })


def main() -> None:
    draws = pd.read_csv(RESULTS / "robust_model_selection_draws.csv")
    cells = draws.groupby(["panel", "dataset", "method"]).selected_realized_test_loss.apply(summarize).unstack()
    cells = cells.reset_index()
    cells.to_csv(RESULTS / "selection_tail_risk_cells.csv", index=False)
    panels = {}
    all_pass = True
    for panel, current in cells.groupby("panel"):
        pivot = current.pivot(index="dataset", columns="method", values="loss_q95")
        means = current.groupby("method").mean(numeric_only=True)
        controls = METHODS[1:]
        clauses = {
            "mean_q95_lower_all_controls": bool(all(means.loc["strength2", "loss_q95"] < means.loc[c, "loss_q95"] for c in controls)),
            "dataset_q95_lower_iid_at_least_60pct": bool(np.mean(pivot.strength2 < pivot.iid16) >= 0.6),
        }
        all_pass &= all(clauses.values())
        panels[panel] = {
            "datasets": len(pivot), "clauses": clauses,
            "datasets_strength2_q95_lower_iid": int((pivot.strength2 < pivot.iid16).sum()),
            "mean_tail_metrics_by_method": means.to_dict(orient="index"),
        }
    summary = {"status": "complete", "panels": panels, "frozen_tail_gate_passed": bool(all_pass)}
    (RESULTS / "selection_tail_risk_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

