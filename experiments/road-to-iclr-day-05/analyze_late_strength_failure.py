"""Diagnose the high-order mechanism behind late strength-2 failures."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

import analyze_robust_model_selection as RMS


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PERMUTATIONS = 100_000


def main() -> None:
    frames = []
    for block, name in (("A", "late_source_strength2_cells.csv"), ("B", "late_source_b_strength2_cells.csv")):
        current = pd.read_csv(RESULTS / name)
        current = current[(current.split == "test") & current.material].copy()
        current["block"] = block
        frames.append(current)
    frame = pd.concat(frames, ignore_index=True)
    frame["pair_fraction"] = frame.main_plus_pair_fraction - frame.main_fraction
    frame["higher_fraction"] = 1 - frame.main_plus_pair_fraction
    frame["pair_to_higher_ratio"] = frame.pair_fraction / np.maximum(frame.higher_fraction, 1e-15)
    frame["strength2_log_advantage_vs_strength1"] = np.log(
        frame.four_strength1_residual / frame.strength2_residual
    )
    frame["strength2_beats_strength1"] = frame.strength2_residual < frame.four_strength1_residual
    x = frame.pair_to_higher_ratio.to_numpy()
    y = frame.strength2_log_advantage_vs_strength1.to_numpy()
    observed = float(spearmanr(x, y).statistic)
    rx, ry = rankdata(x), rankdata(y)
    rx -= rx.mean(); ry -= ry.mean()
    denominator = np.sqrt(np.sum(rx ** 2) * np.sum(ry ** 2))
    rng = np.random.default_rng(RMS.stable_seed("late-strength-failure-mechanism"))
    exceed = 0
    for _ in range(100):
        indices = np.argsort(rng.random((1_000, len(y))), axis=1)
        correlations = ry[indices] @ rx / denominator
        exceed += int(np.sum(np.abs(correlations) >= abs(observed)))
    pvalue = (exceed + 1) / (PERMUTATIONS + 1)
    columns = [
        "block", "dataset", "model", "pair_fraction", "higher_fraction",
        "pair_to_higher_ratio", "strength2_log_advantage_vs_strength1",
        "strength2_beats_strength1",
    ]
    frame[columns].to_csv(RESULTS / "late_strength_failure_mechanism.csv", index=False)
    failures = frame[~frame.strength2_beats_strength1]
    summary = {
        "status": "complete", "material_cells": int(len(frame)),
        "strength1_failures": int(len(failures)),
        "spearman_pair_to_higher_vs_log_advantage": observed,
        "permutation_two_sided_pvalue": float(pvalue),
        "failure_cells": failures[columns].to_dict(orient="records"),
        "positive_mechanism_direction": bool(observed > 0),
        "interpretation": "post_failure_interaction_spectrum_diagnostic",
    }
    (RESULTS / "late_strength_failure_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
