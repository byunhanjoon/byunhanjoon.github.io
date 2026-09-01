"""Exact favorable/adverse interaction-order regions for randomized covers."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_strength2_cover import (
    component_coefficients, decompose, expected_residual, incidence_covariance,
    strength1_family, strength2_family,
)
from analyze_strength3_cover import strength3_family


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SHAPE = (4, 4, 2, 4)
NAMES = ("feature", "category", "class", "seed")


def families() -> dict[str, tuple[np.ndarray, int]]:
    return {
        "strength1": (strength1_family(4, 2), 4),
        "strength2": (strength2_family(4, 2), 16),
        "strength3": (strength3_family(4, 2), 64),
    }


def pure_component(subset: tuple[int, ...]) -> np.ndarray:
    vectors = []
    for index, levels in enumerate(SHAPE):
        if index in subset:
            vector = np.arange(levels, dtype=float) - (levels - 1) / 2
            vector /= np.sqrt(np.mean(vector**2))
        else:
            vector = np.ones(levels)
        vectors.append(vector)
    tensor = vectors[0]
    for vector in vectors[1:]:
        tensor = np.multiply.outer(tensor, vector)
    return tensor


def exhaustive_mse(family: np.ndarray, tensor: np.ndarray) -> float:
    estimates = np.empty(len(family), dtype=float)
    for index, design in enumerate(family):
        estimates[index] = np.mean(tensor[tuple(design.T)])
    return float(np.mean(estimates**2))


def empirical_panel(study: str, input_dir: Path, s2_path: Path, s3_path: Path) -> pd.DataFrame:
    selected = pd.read_csv(RESULTS / "validation_screened_cover_cells.csv")
    chosen = selected[selected.study == study][["dataset", "model"]]
    s2 = pd.read_csv(s2_path)
    s3 = pd.read_csv(s3_path)
    s2 = s2[s2.split == "test"].set_index(["dataset", "model"])
    s3 = s3[s3.split == "test"].set_index(["dataset", "model"])
    caches = {}
    rows = []
    for cell in chosen.itertuples(index=False):
        key = (cell.dataset, cell.model)
        predictions = np.load(input_dir / f"{cell.dataset}__{cell.model}.npz")["test_predictions"].astype(np.float64)
        shape = tuple(map(int, predictions.shape[:4]))
        if shape not in caches:
            cov2 = incidence_covariance(strength2_family(*shape[1:]), shape)
            cov3 = incidence_covariance(strength3_family(*shape[1:]), shape)
            caches[shape] = component_coefficients(cov2, shape), component_coefficients(cov3, shape)
        coefficients2, coefficients3 = caches[shape]
        components = {
            name: float(value) for name, value in decompose(predictions, NAMES).items()
            if name in coefficients2
        }
        joint = float(sum(components.values()))
        energies = {
            order: float(sum(value for name, value in components.items() if name.count(":") + 1 == order))
            for order in range(1, 5)
        }
        predicted2 = float(sum(coefficients2[name] * value for name, value in components.items()))
        predicted3 = float(sum(coefficients3[name] * value for name, value in components.items()))
        rows.append({
            "panel": study, "dataset": cell.dataset, "model": cell.model,
            "shape": "x".join(map(str, shape)),
            "low_order_1_2_fraction": (energies[1] + energies[2]) / joint,
            "order3_fraction": energies[3] / joint, "order4_fraction": energies[4] / joint,
            "predicted_strength2_vs_iid16_ratio": predicted2 / (joint / 16),
            "actual_strength2_vs_iid16_ratio": float(s2.loc[key, "strength2_residual"] / s2.loc[key, "iid16_residual"]),
            "predicted_strength3_vs_iid64_ratio": predicted3 / (joint / 64),
            "actual_strength3_vs_iid64_ratio": float(s3.loc[key, "strength3_residual"] / s3.loc[key, "iid64_residual"]),
        })
    return pd.DataFrame(rows)


def main() -> None:
    coefficient_rows = []
    verification_rows = []
    max_error = 0.0
    all_annihilate = True
    current_families = families()
    for method, (family, budget) in current_families.items():
        covariance = incidence_covariance(family, SHAPE)
        coefficients = component_coefficients(covariance, SHAPE)
        strength = int(method[-1])
        for order in range(1, 5):
            names = [name for name in coefficients if name.count(":") + 1 == order]
            values = [coefficients[name] for name in names]
            coefficient_rows.append({
                "method": method, "budget": budget, "order": order,
                "coefficient": float(np.mean(values)),
                "ratio_to_equal_budget_iid": float(budget * np.mean(values)),
            })
            if order <= strength:
                all_annihilate &= max(map(abs, values)) < 1e-12
        for order in range(1, 5):
            for subset in itertools.combinations(range(4), order):
                tensor = pure_component(subset)
                energy = float(np.mean(tensor**2))
                name = ":".join(NAMES[index] for index in subset)
                observed = exhaustive_mse(family, tensor)
                predicted = coefficients[name] * energy
                error = abs(observed - predicted)
                max_error = max(max_error, error)
                verification_rows.append({
                    "method": method, "component": name, "energy": energy,
                    "observed_mse": observed, "predicted_mse": predicted, "absolute_error": error,
                })
    coefficients_frame = pd.DataFrame(coefficient_rows)
    verification = pd.DataFrame(verification_rows)
    empirical = pd.concat([
        empirical_panel(
            "strength2_confirmation", RESULTS / "tier1_confirmation",
            RESULTS / "strength2_confirmation_cells.csv",
            RESULTS / "strength3_confirmation_cells.csv",
        ),
        empirical_panel(
            "strength2_openml_external", RESULTS / "openml_external_cover",
            RESULTS / "strength2_openml_external_cells.csv",
            RESULTS / "strength3_openml_external_cells.csv",
        ),
        empirical_panel(
            "strength2_openml_taskbalanced", RESULTS / "openml_taskbalanced_cover",
            RESULTS / "strength2_openml_taskbalanced_cells.csv",
            RESULTS / "strength3_openml_taskbalanced_cells.csv",
        ),
        empirical_panel(
            "strength2_openml_multiclass", RESULTS / "openml_multiclass_cover",
            RESULTS / "strength2_openml_multiclass_cells.csv",
            RESULTS / "strength3_openml_multiclass_cells.csv",
        ),
    ], ignore_index=True)
    empirical_error = float(max(
        np.max(np.abs(empirical.predicted_strength2_vs_iid16_ratio - empirical.actual_strength2_vs_iid16_ratio)),
        np.max(np.abs(empirical.predicted_strength3_vs_iid64_ratio - empirical.actual_strength3_vs_iid64_ratio)),
    ))
    s2_pure_triple = float(coefficients_frame.query("method == 'strength2' and order == 3").ratio_to_equal_budget_iid.iloc[0])
    s3_pure_four = float(coefficients_frame.query("method == 'strength3' and order == 4").ratio_to_equal_budget_iid.iloc[0])
    summary = {
        "status": "complete",
        "maximum_exhaustive_spectral_error": max_error,
        "maximum_empirical_gram_vs_phase_error": empirical_error,
        "all_declared_low_orders_annihilated": bool(all_annihilate),
        "adverse_corners": {
            "strength2_pure_triple_ratio_vs_iid16": s2_pure_triple,
            "strength3_pure_fourway_ratio_vs_iid64": s3_pure_four,
        },
        "favorable_region_thresholds": {
            "strength2": "16*(E3/9 + E4/27) < 1 for total energy 1",
            "strength3": "E4 < 27/64 of total energy",
        },
        "empirical_cells": len(empirical),
        "empirical_cells_by_panel": {
            name: len(current) for name, current in empirical.groupby("panel")
        },
        "empirical_cells_strength2_in_favorable_region": int((empirical.predicted_strength2_vs_iid16_ratio < 1).sum()),
        "empirical_cells_strength3_in_favorable_region": int((empirical.predicted_strength3_vs_iid64_ratio < 1).sum()),
        "empirical_strength2_ratio_q50_q90_max": [
            float(value) for value in np.quantile(
                empirical.predicted_strength2_vs_iid16_ratio, [.5, .9, 1]
            )
        ],
        "empirical_strength3_ratio_q50_q90_max": [
            float(value) for value in np.quantile(
                empirical.predicted_strength3_vs_iid64_ratio, [.5, .9, 1]
            )
        ],
        "posthoc_scope_extension_panels": [
            "strength2_openml_multiclass", "strength2_openml_taskbalanced"
        ],
    }
    summary["frozen_phase_diagram_gate_passed"] = bool(
        max_error < 1e-10 and empirical_error < 1e-10 and all_annihilate
        and s2_pure_triple > 1 and s3_pure_four > 1
    )
    coefficients_frame.to_csv(RESULTS / "interaction_phase_coefficients.csv", index=False)
    verification.to_csv(RESULTS / "interaction_phase_verification.csv", index=False)
    empirical.to_csv(RESULTS / "interaction_phase_empirical_cells.csv", index=False)
    (RESULTS / "interaction_phase_diagram_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
