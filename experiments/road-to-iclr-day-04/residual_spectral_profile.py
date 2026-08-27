#!/usr/bin/env python3
"""Semantic spectral-retention curves with node and M-isospectral controls.

This is a diagnostic, not a predictive benchmark.  It uses the shared RAPLE
out-of-fold residual covector and asks whether declared smooth modes retain
more projected residual energy than preregistered false operators over a range
of strengths, without choosing the best strength on test data.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from residual_riesz_pilot import field_forms, fit_shared_raple
from support_heat_pilot import (
    HERE,
    clean_numeric,
    hat_basis,
    load_dataset,
    load_tabred,
    support_nodes,
)


DEFAULT_STRENGTHS = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)


def energy_curve(
    mass: np.ndarray,
    stiffness: np.ndarray,
    covector: np.ndarray,
    strengths: np.ndarray,
) -> np.ndarray:
    """Evaluate c^T(M+tau S)^dagger c through generalized eigenmodes."""
    values, vectors = np.linalg.eigh(mass)
    threshold = max(float(values[-1]), 0.0) * 1e-10
    keep = values > threshold
    if not np.any(keep):
        return np.zeros_like(strengths)
    whitener = vectors[:, keep] / np.sqrt(values[keep])[None, :]
    operator = whitener.T @ stiffness @ whitener
    lambdas, modes = np.linalg.eigh(0.5 * (operator + operator.T))
    lambdas = np.maximum(lambdas, 0.0)
    weights = (modes.T @ (whitener.T @ covector)) ** 2
    return np.sum(
        weights[None, :] / (1.0 + strengths[:, None] * lambdas[None, :]),
        axis=1,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "california",
            "tabred-cooking-time",
            "tabred-delivery-eta",
            "tabred-maps-routing",
            "tabred-weather",
        ],
    )
    parser.add_argument("--bins", type=int, default=32)
    parser.add_argument(
        "--strengths", nargs="+", type=float, default=list(DEFAULT_STRENGTHS)
    )
    parser.add_argument(
        "--control-seeds", nargs="+", type=int,
        default=[1201, 1202, 1203, 1204, 1205],
    )
    parser.add_argument("--max-train-rows", type=int, default=50000)
    parser.add_argument("--max-eval-rows", type=int, default=15000)
    parser.add_argument(
        "--output", type=Path,
        default=HERE / "results/residual_spectral_profile.csv",
    )
    parser.add_argument(
        "--summary-output", type=Path,
        default=HERE / "results/residual_spectral_profile_summary.csv",
    )
    args = parser.parse_args()
    strengths = np.asarray(args.strengths, dtype=np.float64)
    if np.any(strengths <= 0):
        raise ValueError("all strengths must be positive")

    rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for name in args.datasets:
        if name.startswith("tabred-"):
            dataset = load_tabred(
                name.removeprefix("tabred-"),
                args.max_train_rows,
                args.max_eval_rows,
                20260826,
            )
        else:
            dataset = load_dataset(
                name,
                max_train_rows=args.max_train_rows,
                max_eval_rows=args.max_eval_rows,
                sample_seed=20260826,
            )
        _, _, residual, _, _ = fit_shared_raple(dataset, 20260826)
        clean = clean_numeric(dataset.x_num)
        dataset_curves: list[dict[str, np.ndarray | float]] = []
        for column in range(clean["train"].shape[1]):
            nodes = support_nodes(clean["train"][:, column], args.bins, 0.35)
            hats = hat_basis(clean["train"][:, column], nodes)
            phi = hats - hats.mean(axis=0)
            covector = phi.T @ residual / len(residual)
            mass, correct, _, _ = field_forms(
                phi, nodes, column, 1.0, args.control_seeds[0]
            )
            semantic = energy_curve(
                mass, correct - mass, covector, strengths
            )
            node_controls = []
            iso_controls = []
            for control_seed in args.control_seeds:
                _, _, node, isospectral = field_forms(
                    phi, nodes, column, 1.0, control_seed
                )
                node_controls.append(
                    energy_curve(mass, node - mass, covector, strengths)
                )
                iso_controls.append(
                    energy_curve(
                        mass, isospectral - mass, covector, strengths
                    )
                )
            node_array = np.asarray(node_controls)
            iso_array = np.asarray(iso_controls)
            energy_zero = float(
                covector @ (np.linalg.pinv(mass, rcond=1e-10) @ covector)
            )
            denominator = max(energy_zero, 1e-30)
            dataset_curves.append(
                {
                    "energy_zero": energy_zero,
                    "semantic": semantic,
                    "node": node_array.mean(axis=0),
                    "isospectral": iso_array.mean(axis=0),
                }
            )
            for index, strength in enumerate(strengths):
                rows.append(
                    {
                        "dataset": name,
                        "column": column,
                        "nodes": len(nodes),
                        "strength": float(strength),
                        "energy_zero": energy_zero,
                        "semantic_retention": float(semantic[index] / denominator),
                        "node_control_retention_mean": float(
                            node_array[:, index].mean() / denominator
                        ),
                        "node_control_retention_sd": float(
                            node_array[:, index].std(ddof=1) / denominator
                        ),
                        "isospectral_retention_mean": float(
                            iso_array[:, index].mean() / denominator
                        ),
                        "isospectral_retention_sd": float(
                            iso_array[:, index].std(ddof=1) / denominator
                        ),
                        "semantic_minus_node": float(
                            (semantic[index] - node_array[:, index].mean())
                            / denominator
                        ),
                        "semantic_minus_isospectral": float(
                            (semantic[index] - iso_array[:, index].mean())
                            / denominator
                        ),
                    }
                )

        total_zero = sum(float(curve["energy_zero"]) for curve in dataset_curves)
        total_zero = max(total_zero, 1e-30)
        for index, strength in enumerate(strengths):
            semantic_total = sum(
                float(np.asarray(curve["semantic"])[index])
                for curve in dataset_curves
            )
            node_total = sum(
                float(np.asarray(curve["node"])[index])
                for curve in dataset_curves
            )
            iso_total = sum(
                float(np.asarray(curve["isospectral"])[index])
                for curve in dataset_curves
            )
            summary_rows.append(
                {
                    "dataset": name,
                    "fields": len(dataset_curves),
                    "strength": float(strength),
                    "semantic_retention": semantic_total / total_zero,
                    "node_control_retention": node_total / total_zero,
                    "isospectral_retention": iso_total / total_zero,
                    "semantic_minus_node": (semantic_total - node_total) / total_zero,
                    "semantic_minus_isospectral": (
                        semantic_total - iso_total
                    ) / total_zero,
                }
            )

    for path, output_rows in (
        (args.output, rows),
        (args.summary_output, summary_rows),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
            writer.writeheader()
            writer.writerows(output_rows)


if __name__ == "__main__":
    main()
