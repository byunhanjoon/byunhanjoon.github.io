"""Source-cluster inference after the untouched four-source extension."""

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
    "pair32": ("disjoint_pair_mean32", "independent_pair_mean32"),
    "pack64": ("mutually_disjoint_pack64", "two_disjoint_pairs64"),
    "unbiased_pair_cross64": ("disjoint_pair_cross64", "independent_block_u64"),
}
PRIOR_FILES = {
    "pair32": "disjoint_pair32_calibration.csv",
    "pack64": "disjoint_pack64_calibration.csv",
    "unbiased_pair_cross64": "disjoint_pair_cross_calibration.csv",
}
LATE_FAMILIES = {
    "pair32": "pair32", "pack64": "pack64", "unbiased_pair_cross64": "pair_cross64"
}


def sign_p(wins: int, total: int) -> float:
    tail = min(wins, total - wins)
    return float(min(1.0, 2 * sum(math.comb(total, k) for k in range(tail + 1)) / 2**total))


def main() -> None:
    late = pd.read_csv(RESULTS / "late_source_packing_calibration.csv")
    late_b = pd.read_csv(RESULTS / "late_source_b_packing_calibration.csv")
    full_keys = pd.read_csv(RESULTS / "disjoint_pack64_calibration.csv")
    full_keys = full_keys[full_keys.product_cells == 128][["panel", "dataset", "model"]].drop_duplicates()
    source_rows, summaries = [], {}
    for name, (action, control) in COMPARISONS.items():
        prior = pd.read_csv(RESULTS / PRIOR_FILES[name])
        if "product_cells" in prior:
            prior = prior[prior.product_cells == 128]
        else:
            prior = prior.merge(full_keys, on=["panel", "dataset", "model"], how="inner")
        current_late = pd.concat([
            late[late.family == LATE_FAMILIES[name]],
            late_b[late_b.family == LATE_FAMILIES[name]],
        ], ignore_index=True)
        combined = pd.concat([
            prior[["dataset", "model", "method", "score_rmse"]],
            current_late[["dataset", "model", "method", "score_rmse"]],
        ], ignore_index=True)
        means = combined[combined.method.isin((action, control))].groupby(
            ["dataset", "method"]
        ).score_rmse.mean().unstack()
        means["percent_reduction"] = 100 * (1 - means[action] / means[control])
        values = means.percent_reduction.to_numpy()
        rng = np.random.default_rng(RMS.stable_seed("combined-packing-source", name))
        indices = rng.integers(0, len(values), size=(BOOTSTRAPS, len(values)))
        boot = values[indices].mean(axis=1)
        interval = np.quantile(boot, [.025, .975])
        wins = int(np.sum(values > 0))
        required_wins = int(math.ceil((6 / 7) * len(values)))
        passed = bool(wins >= required_wins and interval[0] > 0)
        for dataset, row in means.iterrows():
            source_rows.append({
                "comparison": name, "source": dataset,
                "action_rmse": float(row[action]), "control_rmse": float(row[control]),
                "percent_reduction": float(row.percent_reduction),
            })
        summaries[name] = {
            "sources": int(len(values)), "positive_sources": wins,
            "required_positive_sources": required_wins,
            "equal_source_mean_percent_reduction": float(values.mean()),
            "bootstrap_95_interval": [float(interval[0]), float(interval[1])],
            "exact_two_sided_sign_p": sign_p(wins, len(values)),
            "scope_gate_passed": passed,
        }
    frame = pd.DataFrame(source_rows)
    frame.to_csv(RESULTS / "combined_packing_source_effects.csv", index=False)
    summary = {
        "status": "complete", "unique_sources": int(frame.source.nunique()),
        "bootstrap_resamples": BOOTSTRAPS, "comparisons": summaries,
        "all_scope_gates_passed": bool(all(item["scope_gate_passed"] for item in summaries.values())),
    }
    (RESULTS / "combined_packing_source_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
