"""Decompose cover and IID cross-score variance into Proposition 19 terms."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_robust_model_selection as RMS


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
ACTION = "strength2_mean16"
CONTROL = "iid_mean16"
COMPONENTS = ("residual_aligned_variance", "covariance_self_interaction_variance")


def main() -> None:
    frame = pd.read_csv(RESULTS / "cross_variance_identity_cells.csv")
    rows = []
    summary: dict[str, object] = {"status": "complete", "panels": {}}
    clauses = {component: 0 for component in COMPONENTS}
    for panel, current in frame.groupby("panel"):
        panel_record = {}
        for component in COMPONENTS:
            pivot = current.pivot(
                index=["dataset", "task", "model"], columns="method", values=component
            )
            difference = pivot[ACTION] - pivot[CONTROL]
            dataset_difference = difference.groupby("dataset").mean()
            interval = RMS.cluster_interval(
                dataset_difference.to_numpy(),
                RMS.stable_seed("cross-variance-components", panel, component),
            )
            action_mean = float(pivot[ACTION].mean())
            control_mean = float(pivot[CONTROL].mean())
            ratio = action_mean / control_mean if control_mean > 0 else np.nan
            passed = bool(action_mean < control_mean)
            clauses[component] += passed
            panel_record[component] = {
                "cover_mean": action_mean, "iid_mean": control_mean,
                "cover_over_iid": ratio,
                "cover_lower_candidate_cells": int((difference < 0).sum()),
                "tied_candidate_cells": int(np.isclose(difference, 0, rtol=0, atol=1e-18).sum()),
                "candidate_cells": int(len(difference)),
                "equal_source_mean_difference": float(dataset_difference.mean()),
                "source_bootstrap_95_interval": interval,
                "source_interval_excludes_zero_favorably": bool(interval[1] < 0),
                "gate_clause": passed,
            }
            for index, value in difference.items():
                rows.append({
                    "panel": panel, "dataset": index[0], "task": index[1],
                    "model": index[2], "component": component,
                    "cover_minus_iid": float(value),
                })
        for method, method_values in current.groupby("method"):
            residual = float(method_values.residual_aligned_variance.sum())
            self_interaction = float(method_values.covariance_self_interaction_variance.sum())
            total = residual + self_interaction
            panel_record.setdefault("component_fractions", {})[method] = {
                "residual_aligned_fraction": residual / total if total > 0 else np.nan,
                "covariance_self_interaction_fraction": self_interaction / total if total > 0 else np.nan,
            }
        summary["panels"][panel] = panel_record
    summary["panels_passing_by_component"] = clauses
    summary["frozen_gate_passed"] = bool(all(value >= 4 for value in clauses.values()))
    pd.DataFrame(rows).to_csv(RESULTS / "cross_variance_component_differences.csv", index=False)
    (RESULTS / "cross_variance_decomposition_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
