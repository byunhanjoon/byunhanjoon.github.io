"""Calibrate the finite-population/Gaussian antithetic distinction."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

import analyze_robust_model_selection as RMS


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
H = 16
DIMENSION = 7
REPLICATES = 100_000


def normalized_trace_covariance(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean(axis=0)
    y = y - y.mean(axis=0)
    return float(np.mean(np.sum(x * y, axis=-1)) / np.mean(np.sum(x * x, axis=-1)))


def main() -> None:
    rng = np.random.default_rng(RMS.stable_seed("antithetic-operator-boundary"))
    population = rng.normal(size=(H, DIMENSION))
    population -= population.mean(axis=0)
    population_energy = float(np.mean(np.sum(population ** 2, axis=1)))
    finite, gaussian = {}, {}
    for k in (2, 4, 8, 16):
        packs = np.asarray(list(itertools.combinations(range(H), k)), dtype=int)
        means = population[packs].mean(axis=1)
        pack_risk = float(np.mean(np.sum(means ** 2, axis=1)))
        independent_mean_risk = population_energy / k
        # Every unordered pair occurs equally often across the enumerated
        # packs; average directly over the H-choose-2 population pairs.
        pairs = np.asarray(list(itertools.combinations(range(H), 2)), dtype=int)
        first = population[pairs[:, 0]]
        second = population[pairs[:, 1]]
        pair_coefficient = float(
            np.mean(np.sum(first * second, axis=1)) / population_energy
        )
        theoretical_pair = -1 / (H - 1)
        theoretical_ratio = (H - k) / (H - 1)
        finite[str(k)] = {
            "packs_enumerated": int(len(packs)),
            "pair_covariance_coefficient": pair_coefficient,
            "theoretical_pair_covariance_coefficient": theoretical_pair,
            "pair_coefficient_absolute_error": abs(pair_coefficient - theoretical_pair),
            "mean_risk_ratio_vs_independent": pack_risk / independent_mean_risk,
            "theoretical_mean_risk_ratio": theoretical_ratio,
            "risk_ratio_absolute_error": abs(pack_risk / independent_mean_risk - theoretical_ratio),
            "pack_mean_closes": bool(pack_risk < 1e-28),
        }

        raw = rng.normal(size=(REPLICATES, k, DIMENSION))
        antithetic = np.sqrt(k / (k - 1)) * (raw - raw.mean(axis=1, keepdims=True))
        coefficient = normalized_trace_covariance(antithetic[:, 0], antithetic[:, 1])
        expected = -1 / (k - 1)
        gaussian[str(k)] = {
            "replicates": REPLICATES,
            "pair_covariance_coefficient": coefficient,
            "theoretical_pair_covariance_coefficient": expected,
            "pair_coefficient_absolute_error": abs(coefficient - expected),
            "maximum_zero_sum_coordinate_error": float(
                np.max(np.abs(antithetic.sum(axis=1)))
            ),
        }
    summary = {
        "status": "complete", "evidence_status": "controlled_literature_boundary",
        "finite_population_size": H, "dimension": DIMENSION,
        "finite_cover_packs": finite, "gaussian_antithetic": gaussian,
        "finite_exact_identity_passed": bool(all(
            row["pair_coefficient_absolute_error"] < 1e-12
            and row["risk_ratio_absolute_error"] < 1e-12
            for row in finite.values()
        )),
        "gaussian_calibration_passed": bool(all(
            row["pair_coefficient_absolute_error"] < .015
            and row["maximum_zero_sum_coordinate_error"] < 1e-13
            for row in gaussian.values()
        )),
        "same_pair_coefficient_only_at_full_resolution": bool(
            all(abs(finite[str(k)]["theoretical_pair_covariance_coefficient"]
                    - gaussian[str(k)]["theoretical_pair_covariance_coefficient"]) > 1e-12
                for k in (2, 4, 8))
            and abs(finite["16"]["theoretical_pair_covariance_coefficient"]
                    - gaussian["16"]["theoretical_pair_covariance_coefficient"]) < 1e-12
        ),
        "interpretation": "distinct_antithetic_operators_and_estimands",
    }
    (RESULTS / "antithetic_operator_boundary_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
