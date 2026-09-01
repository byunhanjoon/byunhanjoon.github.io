"""Relate quotient decision margins to disjoint-packing accuracy RMSE."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

import analyze_cross_quotient_selection as CQS
import analyze_robust_model_selection as RMS


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
THRESHOLDS = (.001, .005, .01, .02, .05)
COMPARISONS = {
    "pair32": ("disjoint_pair32", "independent_pair32"),
    "pack64": ("mutually_disjoint_pack64", "two_disjoint_pairs64"),
}


def permutation_pvalue(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[float, float]:
    observed = float(spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)
    exceed = 0
    ranked_x = rankdata(x) - np.mean(rankdata(x))
    ranked_y = rankdata(y) - np.mean(rankdata(y))
    denominator = np.sqrt(np.sum(ranked_x ** 2) * np.sum(ranked_y ** 2))
    for _ in range(100):
        indices = np.argsort(rng.random((1_000, len(y))), axis=1)
        values = ranked_y[indices] @ ranked_x / denominator
        exceed += int(np.sum(np.abs(values) >= abs(observed)))
    return observed, (exceed + 1) / 100_001


def main() -> None:
    metric = pd.read_csv(RESULTS / "packing_metric_scope_calibration.csv")
    metric = metric[(metric.metric == "accuracy") & (metric.product_cells == 128)]
    base_rows = []
    directory_map = {panel: directory for panel, _, directory in CQS.PANELS}
    for (panel, dataset, model), _ in metric.groupby(["panel", "dataset", "model"]):
        archive = np.load(RESULTS / directory_map[panel] / f"{dataset}__{model}.npz")
        quotient = archive["validation_predictions"].reshape(
            (-1,) + archive["validation_predictions"].shape[-2:]
        ).mean(axis=0)
        margin = np.abs(quotient[:, 1] - quotient[:, 0])
        row = {
            "panel": panel, "dataset": dataset, "model": model,
            "minimum_margin": float(margin.min()), "median_margin": float(np.median(margin)),
        }
        row.update({f"fraction_margin_below_{threshold}": float(np.mean(margin < threshold))
                    for threshold in THRESHOLDS})
        base_rows.append(row)
    base = pd.DataFrame(base_rows)
    output = base.copy()
    summaries = {}
    for name, (action, control) in COMPARISONS.items():
        pivot = metric[metric.method.isin((action, control))].pivot(
            index=["panel", "dataset", "model"], columns="method", values="rmse"
        ).reset_index()
        pivot[f"{name}_rmse_difference"] = pivot[action] - pivot[control]
        pivot[f"{name}_strict_win"] = pivot[action] < pivot[control] - 1e-15
        output = output.merge(
            pivot[["panel", "dataset", "model", f"{name}_rmse_difference", f"{name}_strict_win"]],
            on=["panel", "dataset", "model"],
        )
        records = []
        for threshold in THRESHOLDS:
            column = f"fraction_margin_below_{threshold}"
            merged = base.merge(pivot, on=["panel", "dataset", "model"])
            correlation, pvalue = permutation_pvalue(
                merged[column].to_numpy(), merged[f"{name}_rmse_difference"].to_numpy(),
                RMS.stable_seed("accuracy-margin", name, str(threshold)),
            )
            wins = merged[merged[f"{name}_strict_win"]][column]
            nonwins = merged[~merged[f"{name}_strict_win"]][column]
            records.append({
                "margin_threshold": threshold, "spearman_correlation": correlation,
                "permutation_two_sided_pvalue": pvalue,
                "strict_win_mean_near_tie_fraction": float(wins.mean()),
                "nonwin_mean_near_tie_fraction": float(nonwins.mean()),
            })
        summaries[name] = records
    output.to_csv(RESULTS / "accuracy_margin_diagnostic.csv", index=False)
    summary = {
        "status": "complete", "candidates": int(len(output)),
        "permutations": 100_000, "comparisons": summaries,
        "interpretation": "diagnostic_no_gate",
    }
    (RESULTS / "accuracy_margin_diagnostic_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
