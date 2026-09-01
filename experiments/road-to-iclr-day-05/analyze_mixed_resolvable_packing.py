"""Resolvable strength-2 coset packing on the observed 4x4x2x4 product."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_cross_quotient_selection as CQS
from analyze_strength2_cover import PERMS4, assert_strength, component_coefficients, strength2_base


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SHAPE = (4, 4, 2, 4)
RESOLUTIONS = 4_096
KS = (1, 2, 4, 8)
PERMS2 = np.asarray(((0, 1), (1, 0)), dtype=int)


def mixed_coset_resolution() -> np.ndarray:
    base = strength2_base(4, 2, 4)
    unique: dict[tuple[int, ...], np.ndarray] = {}
    for shift in itertools.product(range(4), range(4), range(2), range(4)):
        current = base ^ np.asarray(shift, dtype=int)
        ids = tuple(sorted(np.ravel_multi_index(current.T, SHAPE).tolist()))
        unique.setdefault(ids, current)
    return np.stack(list(unique.values()))


def randomized_single_cover_covariance(resolution: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(2026082845)
    cells = int(np.prod(SHAPE))
    second = np.zeros((cells, cells), dtype=np.float64)
    for _ in range(RESOLUTIONS):
        perms = (PERMS4[rng.integers(0, 24)], PERMS4[rng.integers(0, 24)],
                 PERMS2[rng.integers(0, 2)], PERMS4[rng.integers(0, 24)])
        randomized = np.empty_like(resolution)
        for factor, perm in enumerate(perms):
            randomized[:, :, factor] = np.asarray(perm)[resolution[:, :, factor]]
        ids = np.ravel_multi_index(randomized.transpose(2, 0, 1), SHAPE)
        membership = np.zeros((8, cells), dtype=np.float64)
        membership[np.arange(8)[:, None], ids] = 1 / 16
        second += membership.T @ membership / 8
    second /= RESOLUTIONS
    uniform = np.full(cells, 1 / cells)
    return second - np.outer(uniform, uniform)


def covariance_risk(flat: np.ndarray, covariance: np.ndarray) -> float:
    centered = flat.reshape(len(flat), -1).astype(np.float64)
    # Sum over classes and average over evaluation examples, matching direct residual.
    examples = flat.shape[-2]
    return float(np.einsum("id,ij,jd->", centered, covariance, centered, optimize=True) / examples)


def pure_mixed_field(subset: tuple[int, ...]) -> np.ndarray:
    vectors = []
    for factor, levels in enumerate(SHAPE):
        if factor in subset:
            value = np.arange(levels, dtype=float) - (levels - 1) / 2
            value /= np.sqrt(np.mean(value ** 2))
        else:
            value = np.ones(levels)
        vectors.append(value)
    output = vectors[0]
    for value in vectors[1:]:
        output = np.multiply.outer(output, value)
    return output


def main() -> None:
    resolution = mixed_coset_resolution()
    strength = resolution.shape == (8, 16, 4)
    if strength:
        for design in resolution:
            try:
                assert_strength(design, SHAPE, 2)
            except AssertionError:
                strength = False
    all_ids = np.ravel_multi_index(resolution.transpose(2, 0, 1), SHAPE)
    partition = bool(np.array_equal(np.sort(all_ids.reshape(-1)), np.arange(128)))
    covariance = randomized_single_cover_covariance(resolution)

    # At K=4, Proposition 29 multiplies the one-cover covariance by 1/7.
    resolvable_components = component_coefficients(covariance / 7, SHAPE)
    graph_components = pd.read_csv(RESULTS / "pack64_operator_components.csv")
    component_rows = []
    for key, resolvable_value in resolvable_components.items():
        if key.count(":") < 2:
            continue
        graph_value = float(
            graph_components.loc[graph_components.component == key, "pack64_coefficient"].iloc[0]
        )
        component_rows.append({
            "component": key, "resolvable_pack64_coefficient": resolvable_value,
            "graph_pack64_coefficient": graph_value,
            "resolvable_minus_graph": resolvable_value - graph_value,
            "preferred_operator": "resolvable" if resolvable_value < graph_value else "graph",
        })
    component_frame = pd.DataFrame(component_rows)
    component_frame.to_csv(RESULTS / "mixed_resolvable_component_comparison.csv", index=False)

    controlled_rows = []
    max_formula_error = 0.0
    for subset in list(itertools.combinations(range(4), 3)) + [(0, 1, 2, 3)]:
        field = pure_mixed_field(subset)
        single = covariance_risk(field.reshape(128, 1, 1), covariance)
        for k in KS:
            independent = single / k
            packed = independent * (8 - k) / 7
            ratio = packed / independent if independent else 0.0
            predicted = (8 - k) / 7
            max_formula_error = max(max_formula_error, abs(ratio - predicted))
            controlled_rows.append({
                "component": ":".join(map(str, subset)), "covers": k, "fits": 16 * k,
                "independent_risk": independent, "resolvable_pack_risk": packed,
                "ratio": ratio, "predicted_ratio": predicted,
            })
    pd.DataFrame(controlled_rows).to_csv(RESULTS / "mixed_resolvable_controlled.csv", index=False)

    graph = pd.read_csv(RESULTS / "disjoint_pack64_calibration.csv")
    graph = graph[(graph.method == "mutually_disjoint_pack64") & (graph.product_cells == 128)]
    rows = []
    for panel, config_name, directory_name in CQS.PANELS:
        config = json.loads((HERE / config_name).read_text())
        for dataset in config["datasets"]:
            for model in config["models"]:
                archive = np.load(RESULTS / directory_name / f"{dataset}__{model}.npz")
                shape = tuple(int(value) for value in archive["validation_predictions"].shape[:4])
                if shape != SHAPE:
                    continue
                flat = archive["validation_predictions"].reshape(
                    (-1,) + archive["validation_predictions"].shape[-2:]
                )
                single = covariance_risk(flat, covariance)
                match = graph[(graph.panel == panel) & (graph.dataset == dataset) & (graph.model == model)]
                assert len(match) == 1
                graph_risk = float(match.prediction_residual.iloc[0])
                for k in KS:
                    independent = single / k
                    packed = independent * (8 - k) / 7
                    ratio = packed / independent if independent else 0.0
                    predicted = (8 - k) / 7
                    max_formula_error = max(max_formula_error, abs(ratio - predicted))
                    rows.append({
                        "panel": panel, "dataset": dataset, "model": model,
                        "covers": k, "fits": 16 * k, "single_cover_risk": single,
                        "independent_cover_risk": independent,
                        "resolvable_pack_risk": packed, "ratio": ratio,
                        "predicted_ratio": predicted,
                        "graph_pack64_risk": graph_risk if k == 4 else np.nan,
                        "resolvable_beats_graph64": bool(packed < graph_risk) if k == 4 else np.nan,
                    })
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "mixed_resolvable_real_frontier.csv", index=False)
    four = frame[frame.covers == 4]
    panel_comparison = four.groupby("panel").agg(
        candidates=("model", "size"),
        resolvable_risk=("resolvable_pack_risk", "mean"),
        graph_risk=("graph_pack64_risk", "mean"),
        candidate_wins=("resolvable_beats_graph64", "sum"),
    )
    panel_comparison["resolvable_beats_graph"] = panel_comparison.resolvable_risk < panel_comparison.graph_risk
    construction_pass = bool(
        strength and partition and max_formula_error < 1e-10
        and frame[frame.covers == 8].resolvable_pack_risk.max() < 1e-15
    )
    empirical_pass = bool(
        panel_comparison.resolvable_beats_graph.sum() >= 3
        and four.resolvable_beats_graph64.sum() >= 16
    )
    summary = {
        "status": "complete", "randomized_resolutions": RESOLUTIONS,
        "cosets": int(len(resolution)), "each_coset_strength2": bool(strength),
        "cosets_partition_full_product": partition,
        "max_ratio_formula_error": max_formula_error,
        "real_full_product_candidates": int(len(four)),
        "full_pack_max_risk": float(frame[frame.covers == 8].resolvable_pack_risk.max()),
        "ratios_by_covers": frame.groupby("covers").ratio.mean().to_dict(),
        "panel_graph_comparison": panel_comparison.reset_index().to_dict(orient="records"),
        "candidate_wins_vs_graph64": int(four.resolvable_beats_graph64.sum()),
        "component_comparison": component_frame.to_dict(orient="records"),
        "components_preferring_graph": int((component_frame.preferred_operator == "graph").sum()),
        "components_preferring_resolution": int((component_frame.preferred_operator == "resolvable").sum()),
        "construction_gate_passed": construction_pass,
        "stronger_empirical_gate_passed": empirical_pass,
    }
    (RESULTS / "mixed_resolvable_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
