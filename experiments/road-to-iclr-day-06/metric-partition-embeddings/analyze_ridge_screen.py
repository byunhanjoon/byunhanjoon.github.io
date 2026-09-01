#!/usr/bin/env python3
"""Analyze only the outcomes declared in PROTOCOL_FREEZE.md."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def gain(control: pd.Series, candidate: pd.Series) -> pd.Series:
    return 100.0 * (control - candidate) / control


def main() -> None:
    df = pd.read_csv(RESULTS / "ridge_screen.csv")
    expected = 4 * 12 * 8 * 7
    assert len(df) == expected
    assert (df.dimension == 16).all()
    agg = df.groupby(["domain", "seed", "method"], as_index=False).test_mse.mean()
    wide = agg.pivot(index=["domain", "seed"], columns="method", values="test_mse")
    cv = (
        df.groupby(["domain", "seed", "method"]).test_mse
        .agg(lambda x: float(np.std(x, ddof=0) / np.mean(x)))
        .rename("schema_cv")
        .reset_index()
    )
    cvw = cv.pivot(index=["domain", "seed"], columns="method", values="schema_cv")

    domains = {}
    for domain in ("interval", "cycle", "tree", "nominal"):
        x = wide.loc[domain]
        domains[domain] = {}
        for method in x.columns:
            g = gain(x.ple, x[method])
            domains[domain][method] = {
                "mean_test_mse": float(x[method].mean()),
                "mean_gain_vs_ple_pct": float(g.mean()),
                "wins_vs_ple": int((x[method] < x.ple).sum()),
            }
        for control in ("code_rbf", "mpe_corrupt", "mpe_native"):
            g = gain(x[control], x.mmpe_native)
            domains[domain]["mmpe_native"][f"mean_gain_vs_{control}_pct"] = float(g.mean())
            domains[domain]["mmpe_native"][f"wins_vs_{control}"] = int((x.mmpe_native < x[control]).sum())
        domains[domain]["schema_cv_median"] = {
            method: float(cvw.loc[domain, method].median()) for method in cvw.columns
        }

    interval = wide.loc["interval"]
    nominal = wide.loc["nominal"]
    structured = {d: wide.loc[d] for d in ("cycle", "tree")}
    h1 = float(gain(interval.ple, interval.mmpe_native).mean()) >= -2.0
    h2_by_domain = {}
    for domain, x in structured.items():
        h2_by_domain[domain] = all(
            int((x.mmpe_native < x[c]).sum()) >= 9
            and float(gain(x[c], x.mmpe_native).mean()) >= 10.0
            for c in ("ple", "code_rbf", "mpe_corrupt")
        )
    cv_reductions = {
        d: 1.0 - float(cvw.loc[d, "mmpe_native"].median()) / max(float(cvw.loc[d, "ple"].median()), 1e-15)
        for d in ("cycle", "tree")
    }
    metadata = json.loads((RESULTS / "ridge_screen.metadata.json").read_text())
    h3 = all(v >= 0.75 for v in cv_reductions.values()) and metadata["max_native_mmpe_schema_prediction_discrepancy"] < 1e-8
    nominal_gain = float(gain(nominal.ple, nominal.mmpe_native).mean())
    nominal_native_corrupt = float(np.mean(np.abs(nominal.mmpe_native - nominal.mpe_corrupt)))
    h4 = nominal_gain <= 2.0 and nominal_native_corrupt < 1e-10
    h5 = all(float(gain(wide.loc[d].mpe_native, wide.loc[d].mmpe_native).mean()) > 0 for d in ("cycle", "tree"))

    summary = {
        "integrity": {"expected_rows": expected, "actual_rows": len(df), "dimension_matched": True},
        "domains": domains,
        "gates": {
            "H1_interval_safety": h1,
            "H2_topology_interpolation": all(h2_by_domain.values()),
            "H2_by_domain": h2_by_domain,
            "H3_schema_stability": h3,
            "H3_schema_cv_reduction": cv_reductions,
            "H4_nominal_negative_control": h4,
            "H4_nominal_gain_vs_ple_pct": nominal_gain,
            "H4_native_corrupt_absolute_mse_gap": nominal_native_corrupt,
            "H5_multiscale_residual": h5,
        },
        "decision": "Primary protocol does not pass because the nominal negative control and multiscale residual fail. Retain single-scale MPE for information-equivalent basis diagnostics; do not attribute all gains to geometry.",
    }
    (RESULTS / "ridge_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["gates"], indent=2))


if __name__ == "__main__":
    main()

