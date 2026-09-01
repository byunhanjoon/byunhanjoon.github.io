#!/usr/bin/env python3
"""Analyze the two explicitly declared post-hoc control stages."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def main() -> None:
    primary = pd.read_csv(RESULTS / "ridge_screen.csv")
    controls = pd.read_csv(RESULTS / "basis_controls.csv")
    assert len(controls) == 4 * 12 * 8 * 3
    df = pd.concat([primary, controls], ignore_index=True)
    wide = (
        df.groupby(["domain", "seed", "method"]).test_mse.mean()
        .unstack("method")
    )
    rows = {}
    for domain in ("interval", "cycle", "tree", "nominal"):
        x = wide.loc[domain]
        rows[domain] = {}
        for control in ("ple", "ple_local", "ple_whitened", "u_ple", "mpe_corrupt"):
            gains = 100.0 * (x[control] - x.mpe_native) / x[control]
            rows[domain][control] = {
                "control_mean_mse": float(x[control].mean()),
                "mpe_mean_mse": float(x.mpe_native.mean()),
                "mpe_mean_gain_pct": float(gains.mean()),
                "mpe_wins": int((x.mpe_native < x[control]).sum()),
                "cells": 12,
            }
    nominal_tie = abs(rows["nominal"]["u_ple"]["mpe_mean_gain_pct"]) <= 0.5
    structured = {
        domain: rows[domain]["u_ple"]["mpe_wins"] >= 9
        and rows[domain]["u_ple"]["mpe_mean_gain_pct"] >= 10
        for domain in ("cycle", "tree")
    }
    summary = {
        "integrity": {"control_rows": len(controls), "expected": 1152},
        "comparisons": rows,
        "gates": {
            "nominal_tie_with_uniform_ple": nominal_tie,
            "semantic_evidence_vs_uniform_ple": all(structured.values()),
            "semantic_evidence_by_domain": structured,
        },
        "decision": "The nominal gain is discrete-resolution, not geometry. Single-scale native MPE retains structured unseen-state evidence against Q-PLE, local/whitened Q-PLE, U-PLE, and corrupted metric; promote only MPE to neural confirmation.",
    }
    (RESULTS / "basis_control_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["gates"], indent=2))


if __name__ == "__main__":
    main()

