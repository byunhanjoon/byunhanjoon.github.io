"""Source-cluster metric inference over both untouched OpenML blocks."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_robust_model_selection as RMS


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
BOOTSTRAPS = 100_000
COMPARISONS = {
    "pair32": ("disjoint_pair32", "independent_pair32"),
    "pack64": ("mutually_disjoint_pack64", "two_disjoint_pairs64"),
}


def sign_p(wins: int, total: int) -> float:
    tail = min(wins, total - wins)
    return float(min(1.0, 2 * sum(math.comb(total, k) for k in range(tail + 1)) / 2**total))


def main() -> None:
    frame = pd.concat([
        pd.read_csv(RESULTS / "late_source_metric_scope.csv"),
        pd.read_csv(RESULTS / "late_source_b_metric_scope.csv"),
    ], ignore_index=True)
    source_rows, summaries = [], {}
    for comparison, (action, control) in COMPARISONS.items():
        summaries[comparison] = {}
        for metric in ("brier", "log_loss", "roc_auc", "accuracy"):
            current = frame[(frame.metric == metric) & frame.method.isin((action, control))]
            means = current.groupby(["dataset", "method"]).rmse.mean().unstack()
            action_values, control_values = means[action].to_numpy(), means[control].to_numpy()
            reduction_values = np.zeros(len(means), dtype=float)
            positive = control_values > 1e-15
            reduction_values[positive] = 100 * (1 - action_values[positive] / control_values[positive])
            reduction_values[~positive & (action_values > 1e-15)] = -np.inf
            reduction = pd.Series(reduction_values, index=means.index)
            values = reduction.to_numpy()
            rng = np.random.default_rng(RMS.stable_seed("combined-late-metric-source", comparison, metric))
            indices = rng.integers(0, len(values), size=(BOOTSTRAPS, len(values)))
            boot = values[indices].mean(axis=1)
            interval = np.quantile(boot, [.025, .975])
            wins = int(np.sum(values > 0))
            passed = bool(wins >= 7 and interval[0] > 0) if metric in {"brier", "log_loss"} else None
            for dataset, value in reduction.items():
                source_rows.append({
                    "comparison": comparison, "metric": metric,
                    "source": dataset, "percent_reduction": float(value),
                })
            summaries[comparison][metric] = {
                "sources": int(len(values)), "strictly_positive_sources": wins,
                "equal_source_mean_percent_reduction": float(values.mean()),
                "bootstrap_95_interval": [float(interval[0]), float(interval[1])],
                "exact_two_sided_sign_p": sign_p(wins, len(values)),
                "source_scope_passed": passed,
            }
    pd.DataFrame(source_rows).to_csv(RESULTS / "combined_late_metric_source_effects.csv", index=False)
    summary = {
        "status": "complete", "sources": int(frame.dataset.nunique()),
        "bootstrap_resamples": BOOTSTRAPS, "comparisons": summaries,
        "brier_log_source_scope_passed": bool(all(
            summaries[comparison][metric]["source_scope_passed"]
            for comparison in summaries for metric in ("brier", "log_loss")
        )),
    }
    (RESULTS / "combined_late_metric_source_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
