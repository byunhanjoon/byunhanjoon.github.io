"""Declared post-hoc reviewer diagnostics; these do not alter frozen gates."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

import analyze_pilot as ap


HERE = Path(__file__).resolve().parent


def rho(rows, left: str, right: str) -> float:
    return float(spearmanr(
        [float(row[left]) for row in rows], [float(row[right]) for row in rows]
    ).statistic)


def main() -> None:
    config, rows, bundles = ap.load_all()
    h3 = {}
    for interface in ("learned", "native_tuned"):
        current = [
            row for row in rows
            if row["domain"] in ap.STRUCTURED
            and row["regime"] == "category_holdout"
            and row["interface"] == interface
        ]
        h3[interface] = {
            "pooled_cka_vs_held_mse_spearman": rho(current, "final_native_cka", "held_mse"),
            "pooled_cka_vs_orbit_spearman": rho(current, "final_native_cka", "orbit_damage"),
            "domain_cka_vs_held_mse_spearman": {
                domain: rho(
                    [row for row in current if row["domain"] == domain],
                    "final_native_cka",
                    "held_mse",
                )
                for domain in ap.STRUCTURED
            },
        }
    h5 = {}
    for interface in ("learned", "native_tuned"):
        wins = 0
        reductions = []
        original_reductions = []
        for domain in ap.STRUCTURED:
            for seed in config["seeds"]:
                bundle = bundles[(domain, "category_holdout", seed)]
                correct = float(np.mean(ap.held_patch_mse(
                    bundle, interface, "native_transport"
                )))
                controls = {
                    name: float(np.mean(ap.held_patch_mse(bundle, interface, name)))
                    for name in ("original", "mean", "random", "shuffled_transport")
                }
                wins += int(all(correct < value for value in controls.values()))
                best = min(controls.values())
                reductions.append((best - correct) / best)
                original_reductions.append(
                    (controls["original"] - correct) / controls["original"]
                )
        h5[interface] = {
            "beats_every_control_cell_count": wins,
            "median_reduction_vs_best_control": float(np.median(reductions)),
            "median_reduction_vs_original": float(np.median(original_reductions)),
        }
    correct_corrupt = []
    nominal_cells = []
    for domain in ap.STRUCTURED:
        for seed in config["seeds"]:
            native = ap.chart_mean(
                rows, domain, "category_holdout", seed, "native_fixed", "held_mse"
            )
            corrupt = ap.chart_mean(
                rows, domain, "category_holdout", seed, "corrupt_fixed", "held_mse"
            )
            correct_corrupt.append((corrupt - native) / corrupt)
    for seed in config["seeds"]:
        nominal_cells.append({
            interface: ap.chart_mean(
                rows, "nominal16", "category_holdout", seed, interface, "held_mse"
            )
            for interface in config["interfaces"]
        })
    summary = {
        "status": "posthoc_not_gate_changing",
        "h3_within_interface": h3,
        "h5_both_interfaces": h5,
        "native_vs_corrupt_structured": {
            "win_count": int(sum(value > 0 for value in correct_corrupt)),
            "median_relative_reduction": float(np.median(correct_corrupt)),
        },
        "nominal_cell_held_mse": nominal_cells,
    }
    path = HERE / "results" / "posthoc_diagnostics.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

