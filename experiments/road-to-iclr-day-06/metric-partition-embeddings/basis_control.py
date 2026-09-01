#!/usr/bin/env python3
"""Post-hoc information-equivalent PLE controls declared in the adjacent note."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

import metric_partition_benchmark as mpb


METHODS = ("ple_local", "ple_whitened", "u_ple")


def local_basis(cumulative: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = {}
    for part, p in cumulative.items():
        h = np.empty_like(p)
        h[:, 0] = 1.0 - p[:, 0]
        h[:, 1:] = p[:, :-1] - p[:, 1:]
        out[part] = h
    return out


def whiten_basis(cumulative: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    train = cumulative["train"]
    mean = train.mean(axis=0)
    centered = train - mean
    covariance = centered.T @ centered / len(train)
    values, vectors = np.linalg.eigh(covariance)
    keep = values > 1e-10
    transform = np.zeros_like(covariance)
    transform[:, keep] = vectors[:, keep] / np.sqrt(values[keep])[None, :]
    return {part: (x - mean) @ transform for part, x in cumulative.items()}


def uniform_ple(values: dict[str, np.ndarray], dim: int = 16) -> dict[str, np.ndarray]:
    lo = float(np.min(values["train"]))
    hi = float(np.max(values["train"]))
    knots = np.linspace(lo, hi, dim + 1)
    out = {}
    for part, x in values.items():
        z = np.empty((len(x), dim), dtype=np.float64)
        for j in range(dim):
            z[:, j] = np.clip((x - knots[j]) / max(knots[j + 1] - knots[j], 1e-12), 0.0, 1.0)
        out[part] = z
    return out


def run(output: Path) -> None:
    rows = []
    for domain_name in mpb.DOMAINS:
        for seed in range(20260880, 20260892):
            domain = mpb.make_domain(domain_name, seed)
            y = mpb.standardize_target(domain)
            for schema in range(8):
                stored = mpb.stored_values(domain, schema, seed)
                cumulative = mpb.ple_fit_transform(stored["train"], stored)
                features = {
                    "ple_local": local_basis(cumulative),
                    "ple_whitened": whiten_basis(cumulative),
                    "u_ple": uniform_ple(stored),
                }
                for method in METHODS:
                    result = mpb.ridge_select(features[method], y)
                    result.pop("predictions")
                    rows.append({
                        "domain": domain_name,
                        "seed": seed,
                        "schema": schema,
                        "method": method,
                        "dimension": 16,
                        **result,
                    })
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    run(mpb.RESULTS / "basis_controls.csv")
