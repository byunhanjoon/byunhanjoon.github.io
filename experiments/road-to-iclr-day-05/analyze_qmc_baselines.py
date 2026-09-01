"""Scrambled Sobol/LHS equal-budget baselines on exact nuisance tensors."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

from analyze_strength2_cover import expected_residual, incidence_covariance, strength2_family


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
INPUT = RESULTS / "tier1_confirmation"
DESIGNS = 4096
SHAPE = (4, 4, 2, 4)


def seed(kind: str, index: int) -> int:
    return int.from_bytes(hashlib.sha256(f"qmc:{kind}:{index}".encode()).digest()[:4], "little")


def mapped(points: np.ndarray, shape: tuple[int, ...] = SHAPE) -> np.ndarray:
    levels = np.asarray(shape)
    return np.minimum((points * levels[None]).astype(int), levels - 1)


def design_family(kind: str, shape: tuple[int, ...] = SHAPE) -> np.ndarray:
    output = np.empty((DESIGNS, 16, 4), dtype=np.int8)
    for index in range(DESIGNS):
        if kind == "sobol16":
            points = qmc.Sobol(d=4, scramble=True, seed=seed(kind, index)).random_base2(4)
        elif kind == "lhs16":
            points = qmc.LatinHypercube(d=4, scramble=True, seed=seed(kind, index)).random(16)
        else:
            raise ValueError(kind)
        output[index] = mapped(points, shape)
    return output


def pair_balance_fraction(family: np.ndarray, shape: tuple[int, ...] = SHAPE) -> float:
    balanced = np.ones(len(family), dtype=bool)
    for left in range(4):
        for right in range(left + 1, 4):
            expected = len(family[0]) / (shape[left] * shape[right])
            if expected != int(expected):
                continue
            for index, design in enumerate(family):
                counts = np.zeros((shape[left], shape[right]), dtype=int)
                np.add.at(counts, (design[:, left], design[:, right]), 1)
                balanced[index] &= np.all(counts == expected)
    return float(balanced.mean())


def covariance(family: np.ndarray, shape: tuple[int, ...] = SHAPE) -> np.ndarray:
    return incidence_covariance(family, shape)


def main() -> None:
    screened = pd.read_csv(RESULTS / "validation_screened_cover_cells.csv")
    material = screened[screened.study == "strength2_confirmation"][["dataset", "model"]]
    caches = {}
    rows = []
    for cell in material.itertuples(index=False):
        archive = np.load(INPUT / f"{cell.dataset}__{cell.model}.npz")
        predictions = archive["test_predictions"].astype(np.float64)
        shape = tuple(int(value) for value in predictions.shape[:4])
        if shape not in caches:
            families = {kind: design_family(kind, shape) for kind in ("sobol16", "lhs16")}
            covariances = {kind: covariance(family, shape) for kind, family in families.items()}
            covariance_s2 = covariance(strength2_family(*shape[1:]), shape)
            caches[shape] = (families, covariances, covariance_s2)
        families, covariances, covariance_s2 = caches[shape]
        flat = predictions.reshape((-1,) + predictions.shape[-2:])
        quotient = flat.mean(axis=0)
        joint = float(np.mean(np.sum((flat - quotient) ** 2, axis=-1)))
        record = {"dataset": cell.dataset, "model": cell.model, "iid16": joint / 16,
                  "strength2": expected_residual(predictions, covariance_s2)}
        for kind, current in covariances.items():
            record[kind] = expected_residual(predictions, current)
        rows.append(record)
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "qmc_baseline_cells.csv", index=False)
    comparisons = {}
    for control in ("sobol16", "lhs16"):
        comparisons[control] = {
            "cells_strength2_lower": int((frame.strength2 < frame[control]).sum()),
            "pooled_reduction": float(1 - frame.strength2.mean() / frame[control].mean()),
            "control_pairwise_balanced_design_fraction_by_shape": {
                "x".join(map(str, shape)): pair_balance_fraction(values[0][control], shape)
                for shape, values in caches.items()
            },
        }
    summary = {
        "status": "complete", "cells": len(frame), "designs_per_qmc_family": DESIGNS,
        "comparisons": comparisons,
        "frozen_qmc_gate_passed": bool(all(
            value["cells_strength2_lower"] >= 20 and value["pooled_reduction"] > 0
            for value in comparisons.values()
        )),
    }
    (RESULTS / "qmc_baseline_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
