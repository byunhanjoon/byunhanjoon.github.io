#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def main() -> None:
    df = pd.read_csv(RESULTS / "bike_confirmation.csv")
    assert len(df) == 36
    assert df.groupby("model").parameters.nunique().eq(1).all()
    wide = df.pivot(index=["model", "seed"], columns="method", values="test_loss")
    comparisons = {}
    for model in ("mlp", "resnet"):
        x = wide.loc[model]
        comparisons[model] = {}
        for control in ("qple", "periodic", "code_rbf", "mmpe_ring", "mpe_corrupt"):
            gains = 100.0 * (x[control] - x.mpe_ring) / x[control]
            comparisons[model][control] = {
                "control_mean_test_loss": float(x[control].mean()),
                "mpe_mean_test_loss": float(x.mpe_ring.mean()),
                "mpe_mean_gain_pct": float(gains.mean()),
                "mpe_wins": int((x.mpe_ring < x[control]).sum()),
                "cells": 3,
            }
    correct_corrupt_wins = int((wide.mpe_ring < wide.mpe_corrupt).sum())
    practical_gate = (
        all(comparisons[m][c]["mpe_mean_gain_pct"] > 0 for m in ("mlp", "resnet") for c in ("qple", "code_rbf"))
        and correct_corrupt_wins >= 4
    )
    multiscale_promote = all(
        float(wide.loc[m].mmpe_ring.mean()) < float(wide.loc[m].mpe_ring.mean())
        for m in ("mlp", "resnet")
    )
    periodic_better = all(
        float(wide.loc[m].periodic.mean()) < float(wide.loc[m].mpe_ring.mean())
        for m in ("mlp", "resnet")
    )
    summary = {
        "integrity": {"rows": len(df), "equal_parameters_within_backbone": True},
        "comparisons": comparisons,
        "gates": {
            "mpe_practical_gate_vs_qple_code_and_corrupt": practical_gate,
            "correct_vs_corrupt_wins": correct_corrupt_wins,
            "promote_multiscale": multiscale_promote,
            "periodic_better_in_both_backbone_means": periodic_better,
        },
        "decision": "Single-scale ring MPE passes its narrow diagnostic gate but is not the best hour encoder: fixed periodic features have lower mean test loss in both backbones. Do not claim state of the art or universal PLE replacement.",
    }
    (RESULTS / "bike_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

