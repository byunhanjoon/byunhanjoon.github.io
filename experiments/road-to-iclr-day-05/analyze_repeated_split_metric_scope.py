"""Aggregate post-primary nonlinear/ranking scope over alternate splits."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SPLITS = (2026082901, 2026082911, 2026082921)
TOLERANCE = 1e-15
COMPARISONS = {
    "pair32": ("disjoint_pair32", "independent_pair32"),
    "pack64": ("mutually_disjoint_pack64", "two_disjoint_pairs64"),
}


def main() -> None:
    frames = []
    for split_seed in SPLITS:
        current = pd.read_csv(RESULTS / f"modern_split_{split_seed}_metric_scope.csv")
        current["split_seed"] = split_seed
        frames.append(current)
    frame = pd.concat(frames, ignore_index=True)
    summaries = {}
    for name, (action, control) in COMPARISONS.items():
        metrics = {}
        for metric in ("brier", "log_loss", "roc_auc", "accuracy"):
            current = frame[
                (frame.metric == metric) & frame.method.isin((action, control))
            ]
            pivot = current.pivot(
                index=["split_seed", "dataset", "model"],
                columns="method", values="rmse",
            )
            difference = pivot[control] - pivot[action]
            nondegenerate = pivot[control] > 1e-12
            difference = difference[nondegenerate]
            loss_index = difference[difference < -TOLERANCE].index
            source = current.groupby(
                ["split_seed", "dataset", "method"]
            ).rmse.mean().unstack()
            metrics[metric] = {
                "nondegenerate_strict_wins": int((difference > TOLERANCE).sum()),
                "nondegenerate_ties": int((difference.abs() <= TOLERANCE).sum()),
                "nondegenerate_losses": int((difference < -TOLERANCE).sum()),
                "loss_cells": [
                    {"split_seed": int(index[0]), "dataset": str(index[1]),
                     "model": str(index[2]), "rmse_difference_action_minus_control":
                     float(-difference.loc[index])}
                    for index in loss_index
                ],
                "nondegenerate_cells": int(len(difference)),
                "mean_nondegenerate_rmse_ratio": float(
                    pivot.loc[nondegenerate, action].mean()
                    / pivot.loc[nondegenerate, control].mean()
                ),
                "dataset_split_mean_nohigher": int(
                    (source[action] <= source[control] + TOLERANCE).sum()
                ),
                "dataset_split_means": int(len(source)),
            }
        summaries[name] = metrics
    summary = {
        "status": "complete", "evidence_status": "post_primary_diagnostic",
        "split_seeds": list(SPLITS), "comparisons": summaries,
        "interpretation": (
            "brier_repeats; one_tiny_pair_log_loss; auc_accuracy_sparse_nonadverse"
        ),
        "formal_gate_imposed": False,
    }
    (RESULTS / "repeated_split_metric_scope_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
