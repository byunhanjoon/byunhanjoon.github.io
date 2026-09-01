#!/usr/bin/env python3
"""Frozen hierarchical analysis of CV fold budget."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def unique(pattern: str) -> Path:
    paths = sorted(ROOT.glob(pattern))
    if len(paths) != 1: raise RuntimeError(f"expected one {pattern}, found {paths}")
    return paths[0]


def paired(frame: pd.DataFrame, left: str, right: str) -> dict[str, np.ndarray]:
    return {
        str(name): (pivot[left] - pivot[right]).to_numpy()
        for name, group in frame.groupby("dataset", sort=True)
        for pivot in [group.pivot(index="episode_index", columns="method", values="loss")]
    }


def hierarchical(values: dict[str, np.ndarray], draws: int, seed: int) -> dict:
    names = sorted(values); observed = float(np.mean([values[n].mean() for n in names]))
    rng = np.random.default_rng(seed); samples = np.empty(draws)
    for draw in range(draws):
        chosen = rng.choice(names, len(names), replace=True)
        samples[draw] = np.mean([
            np.mean(rng.choice(values[str(n)], len(values[str(n)]), replace=True)) for n in chosen
        ])
    low, high = np.quantile(samples, [0.025, 0.975])
    return {"gain": observed, "ci_low": float(low), "ci_high": float(high)}


def main() -> None:
    frame = pd.read_csv(unique("results/processed/real_cv_budget_*_cells.csv"))
    metadata = json.loads(unique("results/raw/real_cv_budget_*.metadata.json").read_text())
    config = yaml.safe_load((ROOT / "configs/cv_budget.yaml").read_text())
    draws = int(config["bootstrap_draws"]); comparisons = {}
    for index, fold in enumerate(config["cv_folds"]):
        comparisons[f"{fold}fold_vs_fixed"] = hierarchical(
            paired(frame, "fixed", f"competence_{fold}fold"), draws, 27_000 + index
        )
    two_minus_three = hierarchical(
        paired(frame, "competence_2fold", "competence_3fold"), draws, 27_100
    )
    comparisons["threefold_vs_twofold"] = two_minus_three
    # Positive means three-fold has lower loss; upper bound is the possible two-fold harm.
    gate = (
        comparisons["2fold_vs_fixed"]["ci_low"] > 0
        and two_minus_three["ci_high"] <= float(config["two_fold_harm_margin"])
    )
    audit = {
        "protocol": "CV_BUDGET_PROTOCOL.md", "metadata": metadata,
        "comparisons": comparisons,
        "mean_losses": frame.groupby("method")["loss"].mean().to_dict(),
        "fit_reduction_two_vs_three": 0.25, "low_cost_gate_pass": gate,
    }
    audit_path = ROOT / "results/processed/real_cv_budget_audit_v1.json"
    summary_path = ROOT / "results/processed/real_cv_budget_summary_v1.csv"
    for output in (audit_path, summary_path):
        if output.exists(): raise FileExistsError(f"refusing to overwrite {output}")
    frame.groupby(["dataset", "method"], as_index=False).agg(
        mean_loss=("loss", "mean"), episodes=("episode_index", "nunique")
    ).to_csv(summary_path, index=False)
    audit_path.write_text(json.dumps(audit, indent=2) + "\n"); print(json.dumps(audit, indent=2))


if __name__ == "__main__": main()
