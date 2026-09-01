#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def main() -> None:
    df = pd.read_csv(RESULTS / "ridge_screen.csv")
    output = {}
    for domain in ("interval", "cycle", "tree", "nominal"):
        output[domain] = {}
        for method in ("linear", "ple", "periodic", "code_rbf"):
            rhos = []
            part = df[(df.domain == domain) & (df.method == method)]
            for _, group in part.groupby("seed"):
                rhos.append(float(spearmanr(group.metric_distortion, group.test_mse).statistic))
            output[domain][method] = {
                "median_within_seed_spearman": float(np.median(rhos)),
                "mean_within_seed_spearman": float(np.mean(rhos)),
            }
    summary = {
        "correlations": output,
        "decision": "The simple global pairwise-distance distortion is predictive for monotone interval charts but not for arbitrary cycle/tree codebooks. Exact MPE invariance survives; T4 as a scalar schema-risk predictor is rejected. A successor needs task-weighted local distortion.",
    }
    (RESULTS / "schema_risk_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
