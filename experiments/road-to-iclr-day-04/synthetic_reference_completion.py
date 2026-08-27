#!/usr/bin/env python3
"""Controlled off-support sanity check for reference-mass completion.

This is deliberately not benchmark evidence: the data-generating residual is
smooth in the declared path geometry.  The experiment checks that the
implementation can recover that known mechanism across an unobserved interval
and that the full-space controls falsify it when the geometry is randomized.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from spatial_product_riesz_pilot import (
    exact_isospectral_operator,
    generalized_median_scale,
)
from support_heat_pilot import HERE, hat_basis


METHODS = (
    "empirical_mass",
    "empirical_correct",
    "completed_mass",
    "completed_correct",
    "completed_wrong",
    "completed_isospectral",
)


def true_residual(values: np.ndarray) -> np.ndarray:
    return np.sin(2 * np.pi * values) + 0.35 * np.sin(4 * np.pi * values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--samples", type=int, default=120)
    parser.add_argument("--nodes", type=int, default=25)
    parser.add_argument("--noise", type=float, default=0.25)
    parser.add_argument("--rho", type=float, default=0.01)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "results/synthetic_reference_completion.csv",
    )
    args = parser.parse_args()

    nodes = np.linspace(0.0, 1.0, args.nodes)
    grid = np.linspace(0.0, 1.0, 1001)
    grid_basis = hat_basis(grid, nodes)
    reference_mass = grid_basis.T @ grid_basis / len(grid)
    difference = np.zeros((args.nodes - 1, args.nodes))
    difference[np.arange(args.nodes - 1), np.arange(args.nodes - 1)] = -1
    difference[np.arange(args.nodes - 1), np.arange(1, args.nodes)] = 1
    raw_stiffness = difference.T @ difference / (args.nodes - 1)
    truth = true_residual(grid)
    gap = (grid > 0.4) & (grid < 0.6)

    rows: list[dict[str, object]] = []
    rank_rows: list[dict[str, int]] = []
    for seed in range(args.repetitions):
        rng = np.random.default_rng(seed)
        left = rng.uniform(0.0, 0.4, args.samples // 2)
        right = rng.uniform(0.6, 1.0, args.samples - len(left))
        train_x = np.concatenate([left, right])
        rng.shuffle(train_x)
        train_y = true_residual(train_x) + args.noise * rng.normal(
            size=args.samples
        )
        train_basis = hat_basis(train_x, nodes)
        empirical_mass = train_basis.T @ train_basis / len(train_basis)
        completed_mass = (
            (1.0 - args.rho) * empirical_mass + args.rho * reference_mass
        )
        covector = train_basis.T @ train_y / len(train_basis)
        scale = generalized_median_scale(completed_mass, raw_stiffness)
        stiffness = raw_stiffness / scale
        permutation = rng.permutation(args.nodes)
        wrong_stiffness = stiffness[np.ix_(permutation, permutation)]
        operators = {
            "empirical_mass": empirical_mass,
            "empirical_correct": empirical_mass + args.strength * stiffness,
            "completed_mass": completed_mass,
            "completed_correct": completed_mass + args.strength * stiffness,
            "completed_wrong": completed_mass + args.strength * wrong_stiffness,
            "completed_isospectral": exact_isospectral_operator(
                completed_mass, stiffness, args.strength, 10_000 + seed
            ),
        }
        rank_rows.append(
            {
                "empirical_mass_rank": int(np.linalg.matrix_rank(empirical_mass)),
                "reference_mass_rank": int(np.linalg.matrix_rank(reference_mass)),
                "completed_mass_rank": int(np.linalg.matrix_rank(completed_mass)),
            }
        )
        for method, operator in operators.items():
            coefficients = np.linalg.pinv(operator, rcond=1e-12) @ covector
            prediction = grid_basis @ coefficients
            for region, mask in (("whole", np.ones(len(grid), dtype=bool)),
                                 ("unobserved-gap", gap)):
                rows.append(
                    {
                        "seed": seed,
                        "method": method,
                        "region": region,
                        "mse": float(np.mean((prediction[mask] - truth[mask]) ** 2)),
                    }
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    correct = "completed_correct"
    summary: dict[str, object] = {
        "design": {
            "repetitions": args.repetitions,
            "samples": args.samples,
            "nodes": args.nodes,
            "noise": args.noise,
            "rho": args.rho,
            "strength": args.strength,
            "train_support": "Uniform([0,0.4]) union Uniform([0.6,1])",
        },
        "ranks": {
            key: sorted({row[key] for row in rank_rows})
            for key in rank_rows[0]
        },
        "regions": {},
    }
    for region in ("whole", "unobserved-gap"):
        region_rows = [row for row in rows if row["region"] == region]
        by_method = {
            method: np.array(
                [row["mse"] for row in region_rows if row["method"] == method]
            )
            for method in METHODS
        }
        correct_values = by_method[correct]
        summary["regions"][region] = {
            method: {
                "mean_mse": float(values.mean()),
                "correct_wins": int(np.sum(correct_values < values)),
                "pairs": int(len(values)),
                "mean_correct_gain_pct": (
                    0.0
                    if method == correct
                    else float(np.mean((values - correct_values) / values) * 100)
                ),
            }
            for method, values in by_method.items()
        }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
