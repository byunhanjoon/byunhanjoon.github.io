"""Post-hoc source-concentration audit for the combined packing results."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import trim_mean


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def main() -> None:
    frame = pd.read_csv(RESULTS / "combined_packing_source_effects.csv")
    rows: list[dict[str, float | str]] = []
    summaries: dict[str, dict[str, float | int | bool | str | list[float]]] = {}
    for comparison, group in frame.groupby("comparison", sort=True):
        group = group.sort_values("source").reset_index(drop=True)
        values = group.percent_reduction.to_numpy(dtype=float)
        loo = np.array([
            np.delete(values, index).mean() for index in range(len(values))
        ])
        for index, row in group.iterrows():
            rows.append({
                "comparison": comparison,
                "omitted_source": row.source,
                "leave_one_source_out_mean_percent_reduction": float(loo[index]),
            })
        worst = int(np.argmin(loo))
        summaries[comparison] = {
            "sources": int(len(values)),
            "minimum_individual_percent_reduction": float(values.min()),
            "maximum_individual_percent_reduction": float(values.max()),
            "median_percent_reduction": float(np.median(values)),
            "twenty_percent_trimmed_mean_percent_reduction": float(trim_mean(values, 0.2)),
            "leave_one_source_out_mean_range": [float(loo.min()), float(loo.max())],
            "worst_case_omitted_source": str(group.iloc[worst].source),
            "all_individual_and_leave_one_out_effects_positive": bool(
                values.min() > 0 and loo.min() > 0
            ),
        }
    pd.DataFrame(rows).to_csv(
        RESULTS / "combined_packing_source_leave_one_out.csv", index=False
    )
    summary = {
        "status": "complete",
        "evidence_status": "post_hoc_diagnostic",
        "comparisons": summaries,
        "all_sensitivity_checks_positive": bool(all(
            result["all_individual_and_leave_one_out_effects_positive"]
            for result in summaries.values()
        )),
    }
    (RESULTS / "combined_packing_source_sensitivity_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
