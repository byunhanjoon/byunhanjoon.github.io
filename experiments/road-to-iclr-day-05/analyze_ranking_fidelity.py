"""Summarize complete candidate-ranking fidelity across selection panels."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def load(path: str, panel_map: dict[str, str] | None = None) -> pd.DataFrame:
    frame = pd.read_csv(RESULTS / path)
    if panel_map:
        frame["panel"] = frame.panel.map(panel_map).fillna(frame.panel)
    return frame


def main() -> None:
    frame = pd.concat([
        load("robust_model_selection_draws.csv"),
        load("openml_external_selection_draws.csv", {"openml_external": "openml_external"}),
        load("taskbalanced_model_selection_draws.csv", {"openml_taskbalanced": "openml_taskbalanced"}),
    ], ignore_index=True)
    means = frame.groupby(["panel", "method"], as_index=False).agg(
        validation_rank_spearman=("validation_rank_spearman", "mean"),
        validation_pairwise_order_accuracy=("validation_pairwise_order_accuracy", "mean"),
        draws=("draw", "count"),
    )
    means.to_csv(RESULTS / "ranking_fidelity_means.csv", index=False)
    panels = {}
    passes = 0
    for panel, current in means.groupby("panel"):
        indexed = current.set_index("method")
        clauses = {
            "spearman_above_iid": bool(
                indexed.loc["strength2", "validation_rank_spearman"]
                > indexed.loc["iid16", "validation_rank_spearman"]
            ),
            "pairwise_accuracy_above_iid": bool(
                indexed.loc["strength2", "validation_pairwise_order_accuracy"]
                > indexed.loc["iid16", "validation_pairwise_order_accuracy"]
            ),
        }
        passes += all(clauses.values())
        panels[panel] = {
            "passed": bool(all(clauses.values())), "clauses": clauses,
            "means": indexed.reset_index().to_dict(orient="records"),
        }
    summary = {
        "status": "complete", "panels": panels,
        "panels_strength2_above_iid_both_metrics": int(passes),
        "frozen_descriptive_gate_passed": bool(passes >= 4),
    }
    (RESULTS / "ranking_fidelity_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
