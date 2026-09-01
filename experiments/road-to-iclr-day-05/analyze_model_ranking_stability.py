"""Complete algorithm-ranking recovery under nuisance-cover designs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_robust_model_selection as RMS
from analyze_strength2_cover import proper_loss


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def analyze_dataset(panel: str, dataset: str, models: list[str], directory: Path) -> list[dict]:
    validation, y, shape = [], None, None
    for model in models:
        archive = np.load(directory / f"{dataset}__{model}.npz")
        current = tuple(archive["validation_predictions"].shape[:4])
        shape = current if shape is None else shape
        if current != shape:
            raise AssertionError("factor shape mismatch")
        y = archive["validation_y"] if y is None else y
        validation.append(archive["validation_predictions"].reshape((-1,) + archive["validation_predictions"].shape[-2:]).astype(np.float64))
    assert shape is not None and y is not None
    quotient = np.asarray([proper_loss(y, values.mean(axis=0)) for values in validation])
    target_order = np.argsort(quotient, kind="stable")
    target_rank = np.empty(len(models), dtype=int)
    target_rank[target_order] = np.arange(len(models))
    pairs = [(left, right) for left in range(len(models)) for right in range(left + 1, len(models))]
    actions = RMS.action_ids(shape, RMS.stable_seed("ranking", panel, dataset))
    rows = []
    for method, ids in actions.items():
        losses = np.stack([RMS.batched_losses(y, values, ids) for values in validation], axis=1)
        orders = np.argsort(losses, axis=1, kind="stable")
        ranks = np.empty_like(orders)
        ranks[np.arange(RMS.DRAWS)[:, None], orders] = np.arange(len(models))[None, :]
        pair_agreements = np.mean(np.column_stack([
            (ranks[:, left] < ranks[:, right]) == (target_rank[left] < target_rank[right])
            for left, right in pairs
        ]), axis=1)
        full = np.all(orders == target_order[None, :], axis=1)
        top_two = np.mean(np.isin(orders[:, :2], target_order[:2]), axis=1) == 1
        rows.append({
            "panel": panel, "dataset": dataset, "method": method,
            "pairwise_ranking_agreement": float(pair_agreements.mean()),
            "exact_full_ranking_recovery": float(full.mean()),
            "exact_top_two_set_recovery": float(top_two.mean()),
        })
    return rows


def main() -> None:
    rows = []
    for panel, config_name, directory in RMS.PANELS:
        config = json.loads((HERE / config_name).read_text())
        for dataset in config["datasets"]:
            rows.extend(analyze_dataset(panel, dataset, config["models"], RESULTS / directory))
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "model_ranking_stability_cells.csv", index=False)
    panels = {}
    all_pass = True
    for panel, current in frame.groupby("panel"):
        pivot = current.pivot(index="dataset", columns="method", values="pairwise_ranking_agreement")
        means = current.groupby("method").mean(numeric_only=True)
        clauses = {
            "strength2_mean_higher_all_controls": bool(all(
                means.loc["strength2", "pairwise_ranking_agreement"]
                > means.loc[control, "pairwise_ranking_agreement"] for control in RMS.METHODS[1:]
            )),
            "strength2_higher_iid_at_least_60pct_datasets": bool(
                np.mean(pivot.strength2 > pivot.iid16) >= 0.6
            ),
        }
        all_pass &= all(clauses.values())
        panels[panel] = {
            "datasets": len(pivot), "clauses": clauses,
            "mean_metrics_by_method": means.to_dict(orient="index"),
            "datasets_strength2_pairwise_agreement_higher_iid": int((pivot.strength2 > pivot.iid16).sum()),
        }
    summary = {
        "status": "complete", "draws_per_dataset": RMS.DRAWS, "panels": panels,
        "frozen_ranking_gate_passed": bool(all_pass),
    }
    (RESULTS / "model_ranking_stability_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

