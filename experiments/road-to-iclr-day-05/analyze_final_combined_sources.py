"""Equal-source sensitivity after adding prospective source block C."""

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
    "pair32": ("pair32", "disjoint_pair_mean32", "independent_pair_mean32"),
    "pack64": ("pack64", "mutually_disjoint_pack64", "two_disjoint_pairs64"),
    "unbiased_pair_cross64": (
        "pair_cross64", "disjoint_pair_cross64", "independent_block_u64"
    ),
}


def sign_p(wins: int, total: int) -> float:
    tail = min(wins, total - wins)
    return float(min(
        1.0, 2 * sum(math.comb(total, k) for k in range(tail + 1)) / 2**total
    ))


def main() -> None:
    prior = pd.read_csv(RESULTS / "expanded_model_source_effects.csv")
    added = pd.read_csv(RESULTS / "late_source_c_packing_calibration.csv")
    rows, summaries = [], {}
    for name, (family, action, control) in COMPARISONS.items():
        current = added[
            (added.family == family) & added.method.isin((action, control))
        ]
        means = current.groupby(["dataset", "method"]).score_rmse.mean().unstack()
        means["percent_reduction"] = 100 * (1 - means[action] / means[control])
        appended = means.reset_index().rename(columns={"dataset": "source"})[
            ["source", "percent_reduction"]
        ]
        effects = pd.concat([
            prior[prior.comparison == name][["source", "percent_reduction"]],
            appended,
        ], ignore_index=True).sort_values("source")
        values = effects.percent_reduction.to_numpy(dtype=float)
        rng = np.random.default_rng(RMS.stable_seed("final-combined-source", name))
        indices = rng.integers(0, len(values), size=(BOOTSTRAPS, len(values)))
        interval = np.quantile(values[indices].mean(axis=1), [.025, .975])
        wins = int((values > 0).sum())
        for row in effects.itertuples(index=False):
            rows.append({
                "comparison": name,
                "source": row.source,
                "percent_reduction": float(row.percent_reduction),
            })
        summaries[name] = {
            "sources": int(len(values)),
            "positive_sources": wins,
            "equal_source_mean_percent_reduction": float(values.mean()),
            "bootstrap_95_interval": [float(interval[0]), float(interval[1])],
            "exact_two_sided_sign_p": sign_p(wins, len(values)),
            "all_sources_positive_and_interval_above_zero": bool(
                wins == len(values) and interval[0] > 0
            ),
        }
    pd.DataFrame(rows).to_csv(
        RESULTS / "final_combined_source_effects.csv", index=False
    )
    summary = {
        "status": "complete",
        "evidence_status": "post_outcome_combination_including_prospective_source_c",
        "unique_sources": 15,
        "bootstrap_resamples": BOOTSTRAPS,
        "comparisons": summaries,
        "all_sources_positive": bool(all(
            row["all_sources_positive_and_interval_above_zero"]
            for row in summaries.values()
        )),
    }
    (RESULTS / "final_combined_source_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
