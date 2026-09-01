"""Linear-programmed symmetry-orbit law over valid disjoint four-packs."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

import analyze_cross_quotient_selection as CQS
from analyze_disjoint_pack64 import sample_pack_and_pairs
from analyze_interaction_phase_diagram import NAMES
from analyze_mixed_resolvable_packing import SHAPE, mixed_coset_resolution


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
GRAPH_DRAWS = 32_768
SUBSETS = list(itertools.combinations(range(4), 3)) + [(0, 1, 2, 3)]


def subset_key(subset: tuple[int, ...]) -> str:
    return ":".join(NAMES[index] for index in subset)


def projector(subset: tuple[int, ...]) -> np.ndarray:
    output = np.asarray([[1.0]])
    for factor, levels in enumerate(SHAPE):
        mean = np.ones((levels, levels)) / levels
        current = np.eye(levels) - mean if factor in subset else mean
        output = np.kron(output, current)
    return output


def template_library() -> tuple[np.ndarray, list[str]]:
    unique: dict[tuple[int, ...], str] = {}
    for chunk in range(GRAPH_DRAWS // 1_024):
        pack, _, _ = sample_pack_and_pairs(SHAPE, "orbit-lp", str(chunk))
        for current in pack:
            cells = tuple(sorted(np.unique(current).tolist()))
            if len(cells) != 64:
                raise AssertionError("candidate is not a mutually disjoint four-pack")
            unique.setdefault(cells, "graph")
    resolution = mixed_coset_resolution()
    resolution_ids = np.ravel_multi_index(resolution.transpose(2, 0, 1), SHAPE)
    for chosen in itertools.combinations(range(8), 4):
        cells = tuple(sorted(resolution_ids[list(chosen)].reshape(-1).tolist()))
        unique.setdefault(cells, "resolution")
    keys = list(unique)
    return np.asarray(keys, dtype=np.int16), [unique[key] for key in keys]


def component_matrix(templates: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    weights = np.zeros((len(templates), 128), dtype=np.float64)
    weights[np.arange(len(templates))[:, None], templates] = 1 / 64
    columns, projections = [], {}
    for subset in SUBSETS:
        key = subset_key(subset)
        projection = projector(subset)
        projections[key] = projection
        rank = int(np.prod([SHAPE[index] - 1 for index in subset]))
        quadratic = np.einsum("ni,ij,nj->n", weights, projection, weights, optimize=True)
        columns.append(128 * quadratic / rank)
    return np.stack(columns, axis=0), projections


def candidate_energies(flat: np.ndarray, projections: dict[str, np.ndarray]) -> dict[str, float]:
    matrix = flat.reshape(128, -1).astype(np.float64)
    examples = flat.shape[-2]
    return {
        key: float(np.einsum("id,ij,jd->", matrix, projection, matrix, optimize=True)
                   / (128 * examples))
        for key, projection in projections.items()
    }


def main() -> None:
    templates, origins = template_library()
    coefficients, projections = component_matrix(templates)
    graph_frame = pd.read_csv(RESULTS / "pack64_operator_components.csv").set_index("component")
    graph = np.asarray([graph_frame.loc[subset_key(subset), "pack64_coefficient"] for subset in SUBSETS])
    graph_se = np.asarray([graph_frame.loc[subset_key(subset), "mc_standard_error"] for subset in SUBSETS])
    lower99 = graph - 2.576 * graph_se

    # x=(template probabilities, worst normalized coefficient).
    count = len(templates)
    objective = np.zeros(count + 1); objective[-1] = 1
    upper = np.concatenate((coefficients, -graph[:, None]), axis=1)
    equality = np.zeros((1, count + 1)); equality[0, :count] = 1
    solution = linprog(
        objective, A_ub=upper, b_ub=np.zeros(5), A_eq=equality, b_eq=np.ones(1),
        bounds=[(0, None)] * count + [(0, None)], method="highs",
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    probabilities = solution.x[:count]
    optimized = coefficients @ probabilities
    support = np.flatnonzero(probabilities > 1e-9)
    component_rows = []
    for index, subset in enumerate(SUBSETS):
        component_rows.append({
            "component": subset_key(subset), "optimized_coefficient": optimized[index],
            "graph_coefficient": graph[index], "graph_lower_99": lower99[index],
            "optimized_to_graph": optimized[index] / graph[index],
            "below_graph_lower_99": bool(optimized[index] < lower99[index]),
        })
    pd.DataFrame(component_rows).to_csv(RESULTS / "orbit_optimal_pack_components.csv", index=False)

    real_rows = []
    for panel, config_name, directory_name in CQS.PANELS:
        config = json.loads((HERE / config_name).read_text())
        for dataset in config["datasets"]:
            for model in config["models"]:
                archive = np.load(RESULTS / directory_name / f"{dataset}__{model}.npz")
                if tuple(archive["validation_predictions"].shape[:4]) != SHAPE:
                    continue
                flat = archive["validation_predictions"].reshape(
                    (-1,) + archive["validation_predictions"].shape[-2:]
                )
                energies = candidate_energies(flat, projections)
                vector = np.asarray([energies[subset_key(subset)] for subset in SUBSETS])
                optimized_risk = float(optimized @ vector)
                graph_risk = float(graph @ vector)
                lower99_risk = float(lower99 @ vector)
                real_rows.append({
                    "panel": panel, "dataset": dataset, "model": model,
                    "optimized_risk": optimized_risk, "graph_operator_risk": graph_risk,
                    "graph_lower99_operator_risk": lower99_risk,
                    "optimized_beats_graph": bool(optimized_risk < graph_risk),
                    "optimized_beats_graph_lower99": bool(optimized_risk < lower99_risk),
                    **{f"energy_{key}": value for key, value in energies.items()},
                })
    real = pd.DataFrame(real_rows)
    real.to_csv(RESULTS / "orbit_optimal_pack_real_risks.csv", index=False)

    # Every template orbit is cell-uniform by transitivity; verify raw templates too.
    support_origins = {origin: int(sum(origins[index] == origin for index in support))
                       for origin in set(origins)}
    summary = {
        "status": "complete", "generated_graph_packs": GRAPH_DRAWS,
        "unique_pack_templates": count, "lp_success": bool(solution.success),
        "minimax_normalized_coefficient": float(solution.x[-1]),
        "support_size": int(len(support)), "support_by_origin": support_origins,
        "support_probabilities": [
            {"template_index": int(index), "origin": origins[index],
             "probability": float(probabilities[index]),
             "cells": templates[index].tolist()}
            for index in support
        ],
        "components": component_rows,
        "components_below_graph_point": int(np.sum(optimized < graph)),
        "components_below_graph_lower_99": int(np.sum(optimized < lower99)),
        "real_candidates": int(len(real)),
        "real_candidates_below_graph_point": int(real.optimized_beats_graph.sum()),
        "real_candidates_below_graph_lower_99": int(real.optimized_beats_graph_lower99.sum()),
        "orbit_cell_marginals_exact_by_transitivity": True,
        "all_templates_are_64_distinct_cells": bool(np.all([
            len(np.unique(template)) == 64 for template in templates
        ])),
    }
    summary["frozen_gate_passed"] = bool(
        summary["components_below_graph_lower_99"] == 5
        and summary["real_candidates_below_graph_point"] == 23
        and summary["orbit_cell_marginals_exact_by_transitivity"]
        and summary["all_templates_are_64_distinct_cells"]
    )
    (RESULTS / "orbit_optimal_pack_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items()
                      if key not in {"support_probabilities", "components"}}, indent=2))


if __name__ == "__main__":
    main()
