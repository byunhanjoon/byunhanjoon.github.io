"""Assemble the unbiased cross-score RMSE frontier at 32, 64, and 128 fits."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def main() -> None:
    c32 = pd.read_csv(RESULTS / "cross_quotient_score_calibration.csv")
    c32 = c32[c32.method == "strength2_cross32"].copy()
    c32["rmse32"] = np.sqrt(
        c32.score_bias ** 2 + (c32.mc_standard_error * np.sqrt(1_024)) ** 2
    )
    c64 = pd.read_csv(RESULTS / "disjoint_pair_cross_calibration.csv")
    c64 = c64[c64.method == "disjoint_pair_cross64"].copy()
    c128 = pd.read_csv(RESULTS / "disjoint_pack_cross128_calibration.csv")
    c128 = c128[(c128.method == "disjoint_pack_cross128") & (c128.product_cells == 128)].copy()
    keys = ["panel", "dataset", "model"]
    frame = c128[keys + ["score_rmse"]].rename(columns={"score_rmse": "rmse128"})
    frame = frame.merge(c64[keys + ["score_rmse"]].rename(columns={"score_rmse": "rmse64"}), on=keys)
    frame = frame.merge(c32[keys + ["rmse32"]], on=keys)
    frame["ratio_64_to_32"] = frame.rmse64 / frame.rmse32
    frame["ratio_128_to_64"] = frame.rmse128 / frame.rmse64
    frame["exponent_32_to_64"] = -np.log2(frame.ratio_64_to_32)
    frame["exponent_64_to_128"] = -np.log2(frame.ratio_128_to_64)
    frame["improves_32_to_64"] = frame.rmse64 < frame.rmse32
    frame["improves_64_to_128"] = frame.rmse128 < frame.rmse64
    frame["rmse128_is_best"] = frame.rmse128 < frame[["rmse32", "rmse64"]].min(axis=1)
    frame.to_csv(RESULTS / "packed_unbiased_frontier.csv", index=False)
    panels = []
    for panel, current in frame.groupby("panel"):
        means = current[["rmse32", "rmse64", "rmse128"]].mean()
        panels.append({
            "panel": panel, "candidates": int(len(current)),
            "mean_rmse32": float(means.rmse32), "mean_rmse64": float(means.rmse64),
            "mean_rmse128": float(means.rmse128),
            "ratio_64_to_32": float(means.rmse64 / means.rmse32),
            "ratio_128_to_64": float(means.rmse128 / means.rmse64),
            "strictly_monotone": bool(means.rmse32 > means.rmse64 > means.rmse128),
        })
    improve64 = int(frame.improves_32_to_64.sum())
    improve128 = int(frame.improves_64_to_128.sum())
    best128 = int(frame.rmse128_is_best.sum())
    summary = {
        "status": "complete", "full_product_candidates": int(len(frame)),
        "candidates_improving_32_to_64": improve64,
        "candidates_improving_64_to_128": improve128,
        "candidates_with_128_best": best128,
        "median_exponent_32_to_64": float(frame.exponent_32_to_64.median()),
        "median_exponent_64_to_128": float(frame.exponent_64_to_128.median()),
        "panels": panels,
        "frozen_gate_passed": bool(
            all(item["strictly_monotone"] for item in panels)
            and improve64 >= 20 and improve128 >= 20 and best128 >= 22
        ),
    }
    (RESULTS / "packed_unbiased_frontier_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
