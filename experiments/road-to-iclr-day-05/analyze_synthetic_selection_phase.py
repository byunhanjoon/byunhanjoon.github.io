"""Synthetic model-selection phase across pure fANOVA interaction orders."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_interaction_phase_diagram import SHAPE, pure_component
from analyze_strength2_cover import strength2_family


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DRAWS = 65_536
AMPLITUDE = .1
BASE_QUOTIENT = .2
MARGINS = (.002, .005, .010, .020)


def design_ids(family: np.ndarray) -> np.ndarray:
    return np.stack([
        np.ravel_multi_index(design.T, SHAPE) for design in family
    ])


def candidate_scores(
    field: np.ndarray, quotient: float, rng: np.random.Generator,
    family_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    flat = field.reshape(-1)
    left = family_ids[rng.integers(0, len(family_ids), size=DRAWS)]
    right = family_ids[rng.integers(0, len(family_ids), size=DRAWS)]
    cover_a = quotient + AMPLITUDE * flat[left].mean(axis=1)
    cover_b = quotient + AMPLITUDE * flat[right].mean(axis=1)
    cover_score = cover_a * cover_b

    iid_ids = rng.integers(0, len(flat), size=(DRAWS, 32))
    residual = -(quotient + AMPLITUDE * flat[iid_ids])
    iid_score = (
        residual.sum(axis=1) ** 2 - np.sum(residual ** 2, axis=1)
    ) / (32 * 31)
    return cover_score, iid_score


def main() -> None:
    family_ids = design_ids(strength2_family(4, 2))
    rows = []
    for order in range(1, 5):
        field = pure_component(tuple(range(order)))
        for margin in MARGINS:
            quotients = (BASE_QUOTIENT, float(np.sqrt(BASE_QUOTIENT ** 2 + margin)))
            scores = {"strength2_cross32": [], "iid_u32": []}
            for candidate, quotient in enumerate(quotients):
                rng = np.random.default_rng(2026082800 + 10_000 * order + 100 * int(margin * 1000) + candidate)
                cover, iid = candidate_scores(field, quotient, rng, family_ids)
                scores["strength2_cross32"].append(cover)
                scores["iid_u32"].append(iid)
            for method, values in scores.items():
                matrix = np.stack(values, axis=1)
                inversion = matrix[:, 1] < matrix[:, 0]
                rows.append({
                    "interaction_order": order, "quotient_loss_margin": margin,
                    "method": method, "draws": DRAWS,
                    "inversion_rate": float(inversion.mean()),
                    "mean_validation_regret": float(margin * inversion.mean()),
                    "inversions": int(inversion.sum()),
                })
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "synthetic_selection_phase_cells.csv", index=False)
    pivot = frame.pivot(
        index=["interaction_order", "quotient_loss_margin"],
        columns="method", values="inversion_rate"
    )
    low = pivot.loc[[1, 2]]
    triple = pivot.loc[3]
    four = pivot.loc[4]
    clauses = {
        "strength2_zero_inversions_orders_1_2": bool((low.strength2_cross32 == 0).all()),
        "strength2_worse_at_least_3_of_4_triple_margins": bool(
            (triple.strength2_cross32 > triple.iid_u32).sum() >= 3
        ),
        "strength2_better_at_least_3_of_4_fourway_margins": bool(
            (four.strength2_cross32 < four.iid_u32).sum() >= 3
        ),
    }
    summary = {
        "status": "complete", "draws_per_cell": DRAWS,
        "amplitude": AMPLITUDE, "margins": list(MARGINS),
        "clauses": clauses, "frozen_gate_passed": bool(all(clauses.values())),
        "cells": frame.to_dict(orient="records"),
    }
    (RESULTS / "synthetic_selection_phase_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
