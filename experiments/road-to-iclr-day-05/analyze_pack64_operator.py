"""Controlled pure-component covariance audit for disjoint four-packs."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_disjoint_pack64 import sample_pack_and_pairs
from analyze_disjoint_pair_cross import graph_theory
from analyze_interaction_phase_diagram import NAMES, SHAPE, pure_component


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
CHUNKS = 256
PER_CHUNK = 1_024
DRAWS = CHUNKS * PER_CHUNK


def main() -> None:
    subsets = list(itertools.combinations(range(4), 3)) + [(0, 1, 2, 3)]
    fields = {": ".join(NAMES[index] for index in subset): pure_component(subset).reshape(-1)
              for subset in subsets}
    # Normalize keys to match component_coefficients.
    fields = {key.replace(": ", ":"): value for key, value in fields.items()}
    sums = {key: 0.0 for key in fields}
    sums2 = {key: 0.0 for key in fields}
    cell_counts = np.zeros(np.prod(SHAPE), dtype=np.int64)
    for chunk in range(CHUNKS):
        pack, _, _ = sample_pack_and_pairs(SHAPE, "operator", str(chunk))
        flat_ids = pack.reshape(PER_CHUNK, -1)
        np.add.at(cell_counts, flat_ids, 1)
        for key, field in fields.items():
            estimates = field[flat_ids].mean(axis=1)
            squared = estimates ** 2
            sums[key] += float(squared.sum())
            sums2[key] += float(np.sum(squared ** 2))

    theory = graph_theory(SHAPE)
    rows = []
    passed_components = 0
    for key in fields:
        mean = sums[key] / DRAWS
        variance = (sums2[key] - DRAWS * mean ** 2) / (DRAWS - 1)
        se = float(np.sqrt(max(variance, 0) / DRAWS))
        control = theory["disjoint_pair_mean_coefficients"][key] / 2
        passed = mean + 2.576 * se < control
        passed_components += int(passed)
        rows.append({
            "component": key, "pack64_coefficient": mean,
            "mc_standard_error": se, "upper_99": mean + 2.576 * se,
            "two_independent_pairs_coefficient": control,
            "pack_to_two_pair_ratio": mean / control, "strict_99_pass": passed,
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "pack64_operator_components.csv", index=False)
    expected = DRAWS / 2
    marginal_z = (cell_counts - expected) / np.sqrt(DRAWS * .5 * .5)
    summary = {
        "status": "complete", "draws": DRAWS,
        "components": frame.to_dict(orient="records"),
        "components_passing_99_percent": passed_components,
        "max_absolute_cell_marginal_z": float(np.max(np.abs(marginal_z))),
        "mean_cell_inclusion_probability": float(cell_counts.mean() / DRAWS),
        "frozen_gate_passed": bool(
            passed_components == len(fields) and np.max(np.abs(marginal_z)) < 4
        ),
    }
    (RESULTS / "pack64_operator_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
