#!/usr/bin/env python3
"""Synthetic theorem validation for the prospectively frozen MPE program."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import mpe


HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"


@dataclass(frozen=True)
class Space:
    name: str
    distance: np.ndarray
    coordinates: np.ndarray | None = None


def adjacency_distance(graph: nx.Graph, *, weighted: bool = False) -> np.ndarray:
    nodes = sorted(graph.nodes())
    array = nx.to_numpy_array(graph, nodelist=nodes, weight="weight" if weighted else None)
    return mpe.graph_shortest_path(array, weighted=weighted)


def make_spaces(seed: int) -> list[Space]:
    n = 64
    interval_coordinates = np.linspace(0.0, 1.0, n)[:, None]
    interval = np.abs(interval_coordinates - interval_coordinates.T)

    cycle = mpe.cycle_distance(n)
    path = mpe.path_distance(n)

    balanced_graph = nx.balanced_tree(2, 5)
    balanced_graph.add_node(63)
    balanced_graph.add_edge(31, 63)
    balanced = adjacency_distance(balanced_graph)

    unbalanced_graph = nx.Graph()
    unbalanced_graph.add_nodes_from(range(n))
    for node in range(1, 48):
        unbalanced_graph.add_edge(node - 1, node)
    for node in range(48, n):
        unbalanced_graph.add_edge((node - 48) * 3, node)
    unbalanced = adjacency_distance(unbalanced_graph)

    grid_graph = nx.grid_2d_graph(8, 8)
    grid_nodes = sorted(grid_graph.nodes())
    grid_map = {node: index for index, node in enumerate(grid_nodes)}
    grid_graph = nx.relabel_nodes(grid_graph, grid_map)
    grid_coordinates = np.asarray(grid_nodes, dtype=np.float64) / 7.0
    grid = adjacency_distance(grid_graph)

    rng = np.random.default_rng(mpe.stable_seed("geometric", seed))
    geometric_coordinates = rng.uniform(0.0, 1.0, size=(n, 2))
    radius = 0.19
    while True:
        geometric_graph = nx.random_geometric_graph(n, radius, pos={i: geometric_coordinates[i] for i in range(n)})
        if nx.is_connected(geometric_graph):
            break
        radius += 0.02
    for left, right in geometric_graph.edges:
        geometric_graph[left][right]["weight"] = float(
            np.linalg.norm(geometric_coordinates[left] - geometric_coordinates[right])
        )
    geometric = adjacency_distance(geometric_graph, weighted=True)

    small_world_graph = nx.watts_strogatz_graph(n, 4, 0.15, seed=mpe.stable_seed("small-world", seed))
    if not nx.is_connected(small_world_graph):
        components = [sorted(component) for component in nx.connected_components(small_world_graph)]
        for left, right in zip(components[:-1], components[1:]):
            small_world_graph.add_edge(left[0], right[0])
    small_world = adjacency_distance(small_world_graph)

    equality = mpe.equality_distance(n)
    spaces = [
        Space("interval", interval, interval_coordinates),
        Space("cycle", cycle),
        Space("path_graph", path),
        Space("balanced_tree", balanced),
        Space("unbalanced_tree", unbalanced),
        Space("grid_2d", grid, grid_coordinates),
        Space("random_geometric_graph", geometric, geometric_coordinates),
        Space("small_world_graph", small_world),
        Space("nominal_equality", equality),
    ]
    for space in spaces:
        mpe.validate_distance_matrix(space.distance)
    return spaces


def standardized(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return (values - values.mean()) / max(float(values.std()), 1e-12)


def make_targets(space: Space, seed: int) -> dict[str, tuple[np.ndarray, bool]]:
    d = space.distance
    diameter = max(float(np.max(d)), 1e-12)
    anchor_a, anchor_b = 7, 49
    radial_a = d[:, anchor_a] / diameter
    radial_b = d[:, anchor_b] / diameter
    rng = np.random.default_rng(mpe.stable_seed("targets", space.name, seed))
    smooth = np.exp(-3.0 * radial_a) - 0.7 * np.exp(-4.0 * radial_b)
    piecewise = np.maximum(0.0, 1.0 - 3.0 * radial_a) - 0.5 * np.maximum(0.0, 1.0 - 4.0 * radial_b)
    high_frequency = np.sin(12.0 * radial_a) + 0.25 * np.cos(16.0 * radial_b)
    bump = np.exp(-45.0 * radial_a**2)
    discontinuous = (radial_a < np.median(radial_a)).astype(np.float64)
    random_labels = rng.normal(size=len(d))
    permutation = rng.permutation(len(d))
    metric_misaligned = smooth[permutation]
    return {
        "lipschitz_smooth": (standardized(smooth), True),
        "piecewise_smooth": (standardized(piecewise), True),
        "high_frequency": (standardized(high_frequency), True),
        "localized_bump": (standardized(bump), True),
        "discontinuous": (standardized(discontinuous), False),
        "random_labels": (standardized(random_labels), False),
        "metric_misaligned": (standardized(metric_misaligned), False),
    }


def empirical_lipschitz(target: np.ndarray, distance: np.ndarray) -> float:
    difference = np.abs(target[:, None] - target[None, :])
    mask = distance > 1e-12
    return float(np.max(difference[mask] / distance[mask])) if np.any(mask) else math.inf


def training_states(n_states: int, seed: int) -> np.ndarray:
    order = mpe.stable_order(list(range(n_states)), seed)
    return np.sort(order[: int(math.floor(0.6 * n_states))])


def theorem_rows(spaces: list[Space], seeds: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    interpolation: list[dict[str, object]] = []
    coverage: list[dict[str, object]] = []
    for seed in seeds:
        for space in spaces:
            train = training_states(len(space.distance), mpe.stable_seed("train", space.name, seed))
            test = np.setdiff1d(np.arange(len(space.distance)), train)
            grid = mpe.bandwidth_grid(space.distance, train)
            bandwidth = float(grid[min(2, len(grid) - 1)])
            targets = make_targets(space, seed)
            for landmark_budget in (8, 16, 32):
                landmarks = mpe.farthest_point_landmarks(space.distance, train, landmark_budget)
                weights = mpe.state_weight_table(space.distance, landmarks, bandwidth)
                radius = mpe.weighted_metric_radius(weights, space.distance[:, landmarks])
                cover = mpe.cover_radius(space.distance, np.arange(len(space.distance)), landmarks)
                for target_name, (target, intended_smooth) in targets.items():
                    estimate = weights @ target[landmarks]
                    error = np.abs(estimate - target)
                    lipschitz = empirical_lipschitz(target, space.distance)
                    bound = lipschitz * radius
                    violation = error - bound
                    interpolation.append(
                        {
                            "space": space.name,
                            "target": target_name,
                            "intended_smooth": intended_smooth,
                            "seed": seed,
                            "landmarks": len(landmarks),
                            "bandwidth": bandwidth,
                            "cover_radius": cover,
                            "mean_weighted_radius_test": float(np.mean(radius[test])),
                            "max_weighted_radius_test": float(np.max(radius[test])),
                            "empirical_lipschitz": lipschitz,
                            "mean_absolute_error_test": float(np.mean(error[test])),
                            "max_absolute_error_test": float(np.max(error[test])),
                            "max_bound_violation": float(np.max(violation[test])),
                            "bound_pass": bool(np.max(violation[test]) <= 1e-10),
                        }
                    )
                coverage.append(
                    {
                        "space": space.name,
                        "seed": seed,
                        "states": len(space.distance),
                        "landmarks": len(landmarks),
                        "cover_radius": cover,
                        "mean_weighted_radius": float(np.mean(radius[test])),
                    }
                )
    return pd.DataFrame(interpolation), pd.DataFrame(coverage)


def invariance_rows(spaces: list[Space]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    tokens = np.random.default_rng(17).normal(size=(16, 32))
    for space in spaces:
        train = np.arange(0, len(space.distance), 2)
        landmarks = mpe.farthest_point_landmarks(space.distance, train, 16)
        bandwidth = float(mpe.bandwidth_grid(space.distance, train)[0])
        reference_weights = mpe.state_weight_table(space.distance, landmarks, bandwidth)
        reference_token = reference_weights @ tokens
        for relabeling in range(32):
            new_of_old = mpe.codebook_permutation(len(space.distance), mpe.stable_seed(space.name, relabeling))
            transported = np.empty_like(space.distance)
            transported[np.ix_(new_of_old, new_of_old)] = space.distance
            candidate_weights = mpe.state_weight_table(transported, new_of_old[landmarks], bandwidth)[new_of_old]
            candidate_token = candidate_weights @ tokens
            rows.append(
                {
                    "space": space.name,
                    "relabeling": relabeling,
                    "max_weight_difference": float(np.max(np.abs(reference_weights - candidate_weights))),
                    "max_representation_difference": float(np.max(np.abs(reference_token - candidate_token))),
                }
            )
    return pd.DataFrame(rows)


def support_gap_rows(spaces: list[Space], seed: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for space in spaces:
        target = make_targets(space, seed)["lipschitz_smooth"][0]
        distance_values = np.unique(space.distance[space.distance > 1e-12])
        if len(distance_values) == 0:
            continue
        thresholds = np.unique(np.quantile(distance_values, [0.0, 0.1, 0.25, 0.5, 0.75]))
        for query in mpe.stable_order(list(range(len(space.distance))), seed)[:16]:
            for threshold in thresholds:
                train = np.flatnonzero(space.distance[query] >= threshold - 1e-12)
                train = train[train != query]
                if len(train) < 8:
                    continue
                landmarks = mpe.farthest_point_landmarks(space.distance, train, min(32, len(train)))
                bandwidth = float(mpe.bandwidth_grid(space.distance, train)[min(2, len(mpe.bandwidth_grid(space.distance, train)) - 1)])
                weights = mpe.state_weight_table(space.distance, landmarks, bandwidth)
                prediction = float(weights[query] @ target[landmarks])
                unknown_prediction = float(np.mean(target[train]))
                code_prediction = float(np.interp(query, np.sort(train), target[np.sort(train)]))
                nearest = float(np.min(space.distance[query, train]))
                radius = float(mpe.weighted_metric_radius(weights[[query]], space.distance[[query]][:, landmarks])[0])
                error = abs(prediction - target[query])
                best_identity_error = min(abs(unknown_prediction - target[query]), abs(code_prediction - target[query]))
                rows.append(
                    {
                        "space": space.name,
                        "query_state": int(query),
                        "requested_gap": float(threshold),
                        "nearest_training_support": nearest,
                        "weighted_metric_radius": radius,
                        "mpe_absolute_error": error,
                        "identity_ple_best_error": best_identity_error,
                        "mpe_advantage": best_identity_error - error,
                    }
                )
    return pd.DataFrame(rows)


def corruption_rows(spaces: list[Space], seeds: list[int]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    levels = (0.0, 0.05, 0.10, 0.25, 0.50, 1.0)
    for seed in seeds:
        for space in spaces:
            target = make_targets(space, seed)["lipschitz_smooth"][0]
            train = training_states(len(space.distance), mpe.stable_seed("corrupt-train", space.name, seed))
            test = np.setdiff1d(np.arange(len(space.distance)), train)
            for level in levels:
                permutation = mpe.partial_permutation(
                    len(space.distance), level, mpe.stable_seed("corruption", space.name, seed, level)
                )
                metric = mpe.corrupt_state_association(space.distance, permutation)
                landmarks = mpe.farthest_point_landmarks(metric, train, 32)
                bandwidth_grid = mpe.bandwidth_grid(metric, train)
                bandwidth = float(bandwidth_grid[min(2, len(bandwidth_grid) - 1)])
                weights = mpe.state_weight_table(metric, landmarks, bandwidth)
                prediction = weights @ target[landmarks]
                rows.append(
                    {
                        "space": space.name,
                        "seed": seed,
                        "corruption": level,
                        "test_mse": float(np.mean((prediction[test] - target[test]) ** 2)),
                        "test_mae": float(np.mean(np.abs(prediction[test] - target[test]))),
                        "distance_distribution_preserved": bool(
                            np.array_equal(np.sort(metric.ravel()), np.sort(space.distance.ravel()))
                        ),
                    }
                )
    return pd.DataFrame(rows)


def perturbation_rows(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for delta in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2):
        distances = rng.uniform(0.0, 4.0, size=(256, 32))
        changed = np.maximum(0.0, distances + rng.uniform(-delta, delta, size=distances.shape))
        tokens = rng.normal(size=(32, 32))
        norms = np.linalg.norm(tokens, axis=1, keepdims=True)
        tokens /= np.maximum(norms, 1e-12)
        h = 0.8
        affinity = mpe.kernel_affinity(distances / h, "gaussian")
        z0 = float(np.min(affinity.sum(axis=1)))
        original = mpe.mpe_weights(distances, h) @ tokens
        alternative = mpe.mpe_weights(changed, h) @ tokens
        observed = np.linalg.norm(original - alternative, axis=1)
        bound = 2.0 * 32.0 * math.exp(-0.5) * delta / (h * z0)
        rows.append(
            {
                "delta": delta,
                "max_representation_change": float(np.max(observed)),
                "mean_representation_change": float(np.mean(observed)),
                "bound": bound,
                "max_bound_ratio": float(np.max(observed) / bound),
                "bound_pass": bool(float(np.max(observed)) <= bound + 1e-12),
            }
        )
    return pd.DataFrame(rows)


def realizability_rows(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for desired_rank in (1, 4, 8, 16, 32, 40):
        left = rng.normal(size=(48, desired_rank))
        right = rng.normal(size=(desired_rank, 40))
        target = left @ right
        u, singular, vt = np.linalg.svd(target, full_matrices=False)
        dimension = 32
        use = min(dimension, len(singular))
        reconstructed = (u[:, :use] * singular[:use][None, :]) @ vt[:use]
        rank = int(np.linalg.matrix_rank(target, tol=1e-9))
        relative_error = float(np.linalg.norm(reconstructed - target) / max(np.linalg.norm(target), 1e-12))
        rows.append(
            {
                "desired_rank": desired_rank,
                "actual_rank": rank,
                "token_dimension": dimension,
                "relative_error": relative_error,
                "predicted_exact": rank <= dimension,
                "exact_pass": bool((relative_error < 1e-10) == (rank <= dimension)),
            }
        )
    return pd.DataFrame(rows)


def interval_special_case() -> dict[str, float | bool]:
    landmarks = np.linspace(-2.0, 2.0, 17)
    queries = np.linspace(-2.0, 2.0, 10001)
    gap = float(landmarks[1] - landmarks[0])
    distances = np.abs(queries[:, None] - landmarks[None, :])
    weights = mpe.mpe_weights(distances, gap * (1.0 + 1e-14), kernel="triangular")
    values = np.sin(landmarks) + 0.2 * landmarks
    difference = np.abs(weights @ values - np.interp(queries, landmarks, values))
    return {"max_difference": float(np.max(difference)), "passed": bool(np.max(difference) < 1e-12)}


def equality_special_case() -> dict[str, float | bool]:
    distance = mpe.equality_distance(64)
    landmarks = np.arange(32)
    weights = mpe.state_weight_table(distance, landmarks, 1.0)
    difference = float(np.max(np.abs(weights[32:] - weights[32])))
    return {"max_unseen_weight_difference": difference, "passed": difference < 1e-12}


def summarize(
    interpolation: pd.DataFrame,
    invariance: pd.DataFrame,
    support: pd.DataFrame,
    corruption: pd.DataFrame,
    perturbation: pd.DataFrame,
    realizability: pd.DataFrame,
    interval: dict[str, float | bool],
    equality: dict[str, float | bool],
) -> dict[str, object]:
    smooth = interpolation[interpolation["intended_smooth"]]
    support_rho = spearmanr(support["weighted_metric_radius"], support["mpe_absolute_error"]).statistic
    degradation = []
    for (_, group) in corruption.groupby(["space", "seed"]):
        ordered = group.sort_values("corruption")
        degradation.append(float(ordered.iloc[-1]["test_mse"] - ordered.iloc[0]["test_mse"]))
    return {
        "theorem_1": {
            "relabelings": int(len(invariance)),
            "max_representation_difference": float(invariance["max_representation_difference"].max()),
            "passed": bool(invariance["max_representation_difference"].max() < 1e-7),
        },
        "theorem_2": {
            "smooth_cells": int(len(smooth)),
            "max_bound_violation": float(smooth["max_bound_violation"].max()),
            "passed": bool(smooth["bound_pass"].all()),
            "support_radius_error_spearman": float(support_rho),
        },
        "theorem_3": {
            "cases": int(len(realizability)),
            "passed": bool(realizability["exact_pass"].all()),
        },
        "theorem_4": equality,
        "theorem_5": {
            "max_bound_ratio": float(perturbation["max_bound_ratio"].max()),
            "passed": bool(perturbation["bound_pass"].all()),
            "corruption_cells_with_worse_100pct_than_correct": int(np.sum(np.asarray(degradation) > 0)),
            "corruption_cells": int(len(degradation)),
        },
        "theorem_6": {
            "coverage_cells": int(len(interpolation)),
            "all_finite": bool(np.isfinite(interpolation["cover_radius"]).all()),
        },
        "proposition_7": interval,
    }


def write_frame(frame: pd.DataFrame, stem: str) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    frame.to_csv(RAW / f"{stem}.csv", index=False)
    frame.to_parquet(RAW / f"{stem}.parquet", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[20261401, 20261402, 20261403])
    args = parser.parse_args()
    started = time.time()
    spaces = make_spaces(args.seeds[0])
    interpolation, coverage = theorem_rows(spaces, args.seeds)
    invariance = invariance_rows(spaces)
    support = support_gap_rows(spaces, args.seeds[0])
    corruption = corruption_rows(spaces, args.seeds)
    perturbation = perturbation_rows(args.seeds[0])
    realizability = realizability_rows(args.seeds[0])
    interval = interval_special_case()
    equality = equality_special_case()
    summary = summarize(
        interpolation, invariance, support, corruption, perturbation, realizability, interval, equality
    )
    summary["elapsed_seconds"] = time.time() - started
    summary["spaces"] = [space.name for space in spaces]
    summary["targets"] = list(make_targets(spaces[0], args.seeds[0]))
    for frame, stem in (
        (interpolation, "theory_interpolation"),
        (coverage, "theory_coverage"),
        (invariance, "theory_invariance"),
        (support, "synthetic_support_gap"),
        (corruption, "synthetic_metric_corruption"),
        (perturbation, "theory_metric_perturbation"),
        (realizability, "theory_linear_realizability"),
    ):
        write_frame(frame, stem)
    (RAW / "theory_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
