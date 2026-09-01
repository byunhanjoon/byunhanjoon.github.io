"""External OpenML strength/computation selection frontier."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import analyze_selection_strength_frontier as FRONTIER


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def main() -> None:
    config = json.loads((HERE / "openml_external_cover_config.json").read_text())
    rows = []
    for dataset in config["datasets"]:
        rows.extend(FRONTIER.analyze_dataset(
            "openml_external", dataset, config["models"], RESULTS / "openml_external_cover"
        ))
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "openml_external_frontier_cells.csv", index=False)
    means = frame.groupby("method").mean(numeric_only=True)
    pivot = frame.pivot(index="dataset", columns="method", values="selected_realized_test_loss")
    difference = pivot.strength2_b16 - pivot.iid_b64
    cover_names = ["strength1_b4", "strength2_b16", "strength3_b64"]
    iid_names = ["iid_b4", "iid_b16", "iid_b64"]
    agreements = means.loc[cover_names, "selection_agreement"]
    regrets = means.loc[cover_names, "validation_quotient_regret"]
    clauses = {
        "cover_agreement_nondecreasing": bool((agreements.diff().dropna() >= -1e-15).all()),
        "cover_regret_nonincreasing": bool((regrets.diff().dropna() <= 1e-15).all()),
        "each_cover_regret_lower_than_same_budget_iid": bool(
            (regrets.to_numpy() < means.loc[iid_names, "validation_quotient_regret"].to_numpy()).all()
        ),
    }
    cross_budget = {
        "mean_strength2_b16_minus_iid_b64_test_brier": float(difference.mean()),
        "datasets_strength2_b16_lower": int((difference < 0).sum()),
        "strength2_b16_agreement": float(means.loc["strength2_b16", "selection_agreement"]),
        "iid_b64_agreement": float(means.loc["iid_b64", "selection_agreement"]),
    }
    cross_budget["frozen_fourfold_compute_gate_passed"] = bool(
        difference.mean() < 0 and (difference < 0).sum() >= 6
        and cross_budget["strength2_b16_agreement"] >= cross_budget["iid_b64_agreement"]
    )
    summary = {
        "status": "complete", "datasets": int(frame.dataset.nunique()),
        "draws_per_dataset": FRONTIER.DRAWS, "hierarchy_clauses": clauses,
        "hierarchy_gate_passed": bool(all(clauses.values())),
        "cross_budget": cross_budget,
        "method_means": means.reset_index().to_dict(orient="records"),
    }
    (RESULTS / "openml_external_frontier_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
