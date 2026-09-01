#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def main() -> None:
    df = pd.read_csv(RESULTS / "neural_confirmation.csv")
    metadata = json.loads((RESULTS / "neural_confirmation.metadata.json").read_text())
    assert len(df) == metadata["expected_rows"] == 144
    assert df.parameter_count.nunique() == 1
    wide = df.pivot(index=["domain", "seed", "schema"], columns="method", values="test_mse")
    summary = {}
    gates = {}
    for domain in ("cycle", "tree"):
        x = wide.loc[domain]
        summary[domain] = {}
        domain_gate = True
        for control in ("ple", "periodic", "code_rbf", "u_ple", "mpe_corrupt"):
            gain = 100.0 * (x[control] - x.mpe_native) / x[control]
            item = {
                "mean_control_mse": float(x[control].mean()),
                "mean_mpe_mse": float(x.mpe_native.mean()),
                "mean_mpe_gain_pct": float(gain.mean()),
                "mpe_wins": int((x.mpe_native < x[control]).sum()),
                "cells": len(x),
            }
            summary[domain][control] = item
            domain_gate = domain_gate and item["mean_mpe_gain_pct"] > 0 and item["mpe_wins"] >= 8
        gates[domain] = domain_gate
    output = {
        "integrity": {"rows": len(df), "parameter_count": int(df.parameter_count.iloc[0])},
        "comparisons": summary,
        "gates": {"by_domain": gates, "passed": all(gates.values())},
    }
    (RESULTS / "neural_summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
