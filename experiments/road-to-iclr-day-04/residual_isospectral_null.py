#!/usr/bin/env python3
"""Haar-isospectral reference distribution for residual spectral retention.

For a fixed generalized spectrum, a random orientation sends the normalized
residual covector to a uniform point on the sphere. Its squared modal weights
are Dirichlet(1/2, ..., 1/2), so thousands of exact-spectrum controls can be
sampled without building or diagonalizing thousands of matrices.

The reported upper-tail value is an isospectral reference percentile. It is
not a design-based p-value unless random orientation is adopted as the null.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from king_county_spatial_pilot import load_king_county
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


def modal_problem(
    mass: np.ndarray, stiffness: np.ndarray, covector: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    values, vectors = np.linalg.eigh(mass)
    keep = values > max(float(values[-1]), 0.0) * 1e-10
    if not np.any(keep):
        return np.zeros(0), np.zeros(0)
    whitener = vectors[:, keep] / np.sqrt(values[keep])[None, :]
    operator = whitener.T @ stiffness @ whitener
    lambdas, modes = np.linalg.eigh(0.5 * (operator + operator.T))
    q = modes.T @ (whitener.T @ covector)
    return np.maximum(lambdas, 0.0), q


def bh_selected(p_values: np.ndarray, level: float) -> np.ndarray:
    order = np.argsort(p_values)
    accepted = 0
    for rank, index in enumerate(order, start=1):
        if p_values[index] <= level * rank / len(order):
            accepted = rank
    selected = np.zeros(len(p_values), dtype=bool)
    if accepted:
        selected[order[:accepted]] = True
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets", nargs="+", default=["california", "tabred-weather"]
    )
    parser.add_argument("--bins", type=int, default=32)
    parser.add_argument(
        "--strengths", nargs="+", type=float, default=list(DEFAULT_STRENGTHS)
    )
    parser.add_argument("--samples", type=int, default=9999)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--fdr", type=float, default=0.1)
    parser.add_argument("--max-train-rows", type=int, default=50000)
    parser.add_argument("--max-eval-rows", type=int, default=15000)
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "results/residual_isospectral_null.csv",
    )
    args = parser.parse_args()
    strengths = np.asarray(args.strengths, dtype=np.float64)
    if np.any(strengths <= 0):
        raise ValueError("strengths must be positive")
    if args.samples < 1:
        raise ValueError("samples must be positive")

    rows: list[dict[str, object]] = []
    for dataset_offset, name in enumerate(args.datasets):
        if name == "king-county-sales":
            dataset, source = load_king_county()
            names = source["feature_names"]
        elif name.startswith("tabred-"):
            dataset = load_tabred(
                name.removeprefix("tabred-"),
                args.max_train_rows,
                args.max_eval_rows,
                20260826,
            )
            names = None
        else:
            dataset = load_dataset(
                name,
                max_train_rows=args.max_train_rows,
                max_eval_rows=args.max_eval_rows,
                sample_seed=20260826,
            )
            names = None
        _, _, residual, _, _ = fit_shared_raple(dataset, 20260826)
        clean = clean_numeric(dataset.x_num)
        dataset_rows: list[dict[str, object]] = []
        for column in range(clean["train"].shape[1]):
            nodes = support_nodes(clean["train"][:, column], args.bins, 0.35)
            hats = hat_basis(clean["train"][:, column], nodes)
            phi = hats - hats.mean(axis=0)
            covector = phi.T @ residual / len(residual)
            mass, correct, _, _ = field_forms(phi, nodes, column, 1.0)
            lambdas, q = modal_problem(mass, correct - mass, covector)
            q_norm2 = float(q @ q)
            if not len(q) or q_norm2 <= 1e-30:
                semantic_auc = null_mean = null_sd = 0.0
                semantic_z = 0.0
                upper_tail = 1.0
                tau1_retention = tau1_null_mean = 0.0
                rank = len(q)
            else:
                attenuation = 1.0 / (
                    1.0 + strengths[:, None] * lambdas[None, :]
                )
                weights = q * q / q_norm2
                semantic_curve = attenuation @ weights
                semantic_auc = float(semantic_curve.mean())
                rng = np.random.default_rng(
                    args.seed + 100_003 * dataset_offset + column
                )
                sphere = rng.normal(size=(args.samples, len(q)))
                sphere /= np.linalg.norm(sphere, axis=1, keepdims=True)
                null_curves = (sphere * sphere) @ attenuation.T
                null_auc_values = null_curves.mean(axis=1)
                null_mean = float(null_auc_values.mean())
                null_sd = float(null_auc_values.std(ddof=1))
                if null_sd <= 1e-12:
                    semantic_z = 0.0
                    upper_tail = 1.0
                else:
                    semantic_z = (semantic_auc - null_mean) / null_sd
                    upper_tail = float(
                        (1 + np.count_nonzero(null_auc_values >= semantic_auc))
                        / (args.samples + 1)
                    )
                tau1_index = int(np.argmin(np.abs(strengths - 1.0)))
                tau1_retention = float(semantic_curve[tau1_index])
                tau1_null_mean = float(null_curves[:, tau1_index].mean())
                rank = len(q)
            dataset_rows.append(
                {
                    "dataset": name,
                    "column": column,
                    "field_name": names[column] if names is not None else "",
                    "nodes": len(nodes),
                    "rank": rank,
                    "energy_zero": q_norm2,
                    "semantic_auc_retention": semantic_auc,
                    "isospectral_null_auc_mean": null_mean,
                    "isospectral_null_auc_sd": null_sd,
                    "semantic_auc_z": semantic_z,
                    "isospectral_upper_tail": upper_tail,
                    "semantic_tau1_retention": tau1_retention,
                    "isospectral_tau1_mean": tau1_null_mean,
                    "samples": args.samples,
                }
            )
        selected = bh_selected(
            np.asarray(
                [row["isospectral_upper_tail"] for row in dataset_rows],
                dtype=np.float64,
            ),
            args.fdr,
        )
        for row, keep in zip(dataset_rows, selected):
            row["reference_bh_selected"] = bool(keep)
            rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
