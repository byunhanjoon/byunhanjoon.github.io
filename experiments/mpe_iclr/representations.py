#!/usr/bin/env python3
"""Target-independent state representations for the frozen real-data panel."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import networkx as nx
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.sparse import csr_matrix

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from mpe import (  # noqa: E402
    bandwidth_grid,
    codebook_permutation,
    coordinates_to_unit_sphere,
    corrupt_state_association,
    equality_distance,
    farthest_point_landmarks,
    kernel_affinity,
    nystrom_features,
    state_weight_table,
)


@dataclass
class TaskData:
    name: str
    rows: pd.DataFrame
    states: pd.DataFrame
    distance: np.ndarray
    splits: dict[str, Any]
    manifest: dict[str, Any]
    arrays: dict[str, np.ndarray]

    @property
    def state_ids(self) -> list[str]:
        return self.states["state_id"].astype(str).tolist()

    @property
    def state_to_index(self) -> dict[str, int]:
        return {state: index for index, state in enumerate(self.state_ids)}

    def row_state_indices(self) -> np.ndarray:
        lookup = self.state_to_index
        return np.asarray([lookup[value] for value in self.rows["field_state"].astype(str)], dtype=np.int64)


def load_task(name: str, root: Path = HERE / "processed") -> TaskData:
    folder = root / name
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest["status"] != "RUN":
        raise RuntimeError(f"{name}: {manifest['status']} — {manifest.get('reason', '')}")
    arrays = {}
    for path in folder.glob("*.npy"):
        if path.name != "distance_primary.npy":
            arrays[path.stem] = np.load(path)
    return TaskData(
        name=name,
        rows=pd.read_parquet(folder / "rows.parquet"),
        states=pd.read_parquet(folder / "states.parquet"),
        distance=np.asarray(np.load(folder / "distance_primary.npy"), dtype=np.float64),
        splits=json.loads((folder / "splits.json").read_text()),
        manifest=manifest,
        arrays=arrays,
    )


def split_state_indices(task: TaskData, split_index: int) -> dict[str, np.ndarray]:
    lookup = task.state_to_index
    payload = task.splits[str(split_index)]
    return {
        part: np.asarray([lookup[str(state)] for state in payload[part]], dtype=np.int64)
        for part in ("train", "validation", "test")
    }


def split_row_indices(task: TaskData, split_index: int) -> dict[str, np.ndarray]:
    parts = split_state_indices(task, split_index)
    row_states = task.row_state_indices()
    return {part: np.flatnonzero(np.isin(row_states, states)) for part, states in parts.items()}


def piecewise_linear_table(values: np.ndarray, knots: np.ndarray, width: int = 32) -> np.ndarray:
    """Classical cumulative PLE coordinates at a frozen set of ordered knots."""
    x = np.asarray(values, dtype=np.float64)
    k = np.unique(np.asarray(knots, dtype=np.float64))
    if len(k) <= 1:
        return np.zeros((len(x), width), dtype=np.float32)
    result = np.zeros((len(x), len(k) - 1), dtype=np.float64)
    for index in range(len(k) - 1):
        denominator = max(k[index + 1] - k[index], 1e-12)
        result[:, index] = np.clip((x - k[index]) / denominator, 0.0, 1.0)
    if result.shape[1] >= width:
        return result[:, :width].astype(np.float32)
    return np.pad(result, ((0, 0), (0, width - result.shape[1]))).astype(np.float32)


def categorical_unknown_table(n_states: int, train_states: np.ndarray, width: int | None = None) -> np.ndarray:
    train = np.asarray(train_states, dtype=np.int64)
    dimension = len(train) + 1
    table = np.zeros((n_states, dimension), dtype=np.float32)
    table[:, -1] = 1.0
    for column, state in enumerate(train):
        table[state] = 0.0
        table[state, column] = 1.0
    if width is None or width == dimension:
        return table
    if dimension < width:
        return np.pad(table, ((0, 0), (0, width - dimension)))
    return table


def graph_from_paths(states: pd.DataFrame) -> tuple[nx.Graph, list[str], dict[str, list[str]]]:
    graph = nx.Graph()
    paths: dict[str, list[str]] = {}
    state_nodes: list[str] = []
    for row in states.itertuples(index=False):
        state_id = str(row.state_id)
        path = list(json.loads(row.path_json))
        paths[state_id] = path
        graph.add_edges_from(zip(path[:-1], path[1:]))
        state_nodes.append(path[-1])
    return graph, state_nodes, paths


def spectral_coordinates(adjacency: np.ndarray, dimension: int = 32) -> np.ndarray:
    a = np.asarray(adjacency, dtype=np.float64)
    degree = a.sum(axis=1)
    inv = np.divide(1.0, np.sqrt(degree), out=np.zeros_like(degree), where=degree > 0)
    laplacian = np.eye(len(a)) - inv[:, None] * a * inv[None, :]
    values, vectors = eigh((laplacian + laplacian.T) / 2.0)
    nontrivial = np.flatnonzero(values > 1e-10)[:dimension]
    result = vectors[:, nontrivial]
    for column in range(result.shape[1]):
        pivot = int(np.argmax(np.abs(result[:, column])))
        if result[pivot, column] < 0:
            result[:, column] *= -1
    if result.shape[1] < dimension:
        result = np.pad(result, ((0, 0), (0, dimension - result.shape[1])))
    return result.astype(np.float32)


def hierarchy_specialists(task: TaskData, landmarks: np.ndarray, dimension: int = 32) -> dict[str, np.ndarray]:
    graph, state_nodes, paths = graph_from_paths(task.states)
    all_nodes = sorted(graph.nodes, key=str)
    node_index = {node: index for index, node in enumerate(all_nodes)}
    ancestor = np.zeros((len(task.states), len(all_nodes) + 1), dtype=np.float32)
    weighted = np.zeros_like(ancestor)
    depths = np.zeros(len(task.states), dtype=np.float64)
    for row, state in enumerate(task.state_ids):
        path = paths[state]
        depths[row] = len(path)
        for depth, node in enumerate(path, 1):
            ancestor[row, node_index[node]] = 1.0
            weighted[row, node_index[node]] = 1.0 / depth
        ancestor[row, -1] = len(path) / max(map(len, paths.values()))
        weighted[row, -1] = ancestor[row, -1]

    adjacency = nx.to_numpy_array(graph, nodelist=all_nodes, dtype=np.float64)
    spectral_all = spectral_coordinates(adjacency, dimension)
    spectral = spectral_all[[node_index[node] for node in state_nodes]]

    # A deterministic DeepWalk-style topology embedding: truncated SVD of a
    # five-step random-walk co-occurrence matrix. It uses no labels.
    degree = adjacency.sum(axis=1)
    transition = np.divide(adjacency, degree[:, None], out=np.zeros_like(adjacency), where=degree[:, None] > 0)
    cooccurrence = np.zeros_like(transition)
    power = np.eye(len(transition))
    for _ in range(5):
        power = power @ transition
        cooccurrence += power
    symmetric = (cooccurrence + cooccurrence.T) / 2.0
    values, vectors = eigh(symmetric)
    chosen = np.argsort(np.abs(values))[::-1][:dimension]
    deepwalk_all = vectors[:, chosen] * np.sqrt(np.abs(values[chosen]))[None, :]
    deepwalk = deepwalk_all[[node_index[node] for node in state_nodes]]
    if deepwalk.shape[1] < dimension:
        deepwalk = np.pad(deepwalk, ((0, 0), (0, dimension - deepwalk.shape[1])))

    wu = np.zeros((len(task.states), len(landmarks)), dtype=np.float64)
    lch = np.zeros_like(wu)
    max_depth = max(map(len, paths.values()))
    landmark_states = [task.state_ids[index] for index in landmarks]
    for row, state in enumerate(task.state_ids):
        left = paths[state]
        for column, landmark_state in enumerate(landmark_states):
            right = paths[landmark_state]
            common = 0
            for left_node, right_node in zip(left, right):
                if left_node != right_node:
                    break
                common += 1
            wu[row, column] = 2.0 * common / max(1.0, len(left) + len(right))
            path_length = nx.shortest_path_length(graph, left[-1], right[-1])
            lch[row, column] = -math.log((path_length + 1.0) / (2.0 * max_depth + 1.0))
    return {
        "ancestor_multihot": ancestor,
        "path_to_root": weighted,
        "wu_palmer": wu.astype(np.float32),
        "lch_path": lch.astype(np.float32),
        "laplacian": spectral,
        "node2vec": deepwalk.astype(np.float32),
    }


def geographic_specialists(
    task: TaskData, train_states: np.ndarray, landmarks: np.ndarray, bandwidth: float, dimension: int = 32
) -> dict[str, np.ndarray]:
    coordinates = np.asarray(task.arrays["coordinates"], dtype=np.float64)
    sphere = coordinates_to_unit_sphere(coordinates)
    result: dict[str, np.ndarray] = {
        "raw_coordinates": sphere.astype(np.float32),
        "raw_latlon": coordinates.astype(np.float32),
    }
    low = coordinates[train_states].min(axis=0)
    high = coordinates[train_states].max(axis=0)
    normalized = 2.0 * (coordinates - low) / np.where(high > low, high - low, 1.0) - 1.0
    fourier_blocks = []
    frequency = 1.0
    while sum(block.shape[1] for block in fourier_blocks) < dimension:
        fourier_blocks.extend([np.sin(math.pi * frequency * normalized), np.cos(math.pi * frequency * normalized)])
        frequency *= 2.0
    result["coordinate_fourier"] = np.concatenate(fourier_blocks, axis=1)[:, :dimension].astype(np.float32)
    center = sphere[landmarks]
    chord = np.linalg.norm(sphere[:, None, :] - center[None, :, :], axis=2)
    # Convert the chosen arc-length bandwidth to a unit-sphere chord scale.
    chord_h = max(2.0 * math.sin((bandwidth / 6371.0088) / 2.0), 1e-8)
    result["spatial_rbf"] = np.exp(-0.5 * (chord / chord_h) ** 2).astype(np.float32)
    if "adjacency" in task.arrays:
        adjacency = np.asarray(task.arrays["adjacency"], dtype=np.float64)
        result["graph_laplacian"] = spectral_coordinates(adjacency, dimension)
        # Same deterministic topology-only walk embedding convention as above.
        degree = adjacency.sum(axis=1)
        transition = np.divide(adjacency, degree[:, None], out=np.zeros_like(adjacency), where=degree[:, None] > 0)
        cooccurrence = sum(np.linalg.matrix_power(transition, power) for power in range(1, 6))
        values, vectors = eigh((cooccurrence + cooccurrence.T) / 2.0)
        chosen = np.argsort(np.abs(values))[::-1][:dimension]
        result["node2vec"] = (vectors[:, chosen] * np.sqrt(np.abs(values[chosen]))[None, :]).astype(np.float32)
    return result


def string_hash_table(states: Sequence[str], dimension: int = 128) -> np.ndarray:
    table = np.zeros((len(states), dimension), dtype=np.float32)
    for row, value in enumerate(states):
        text = f"  {str(value)}  "
        for index in range(max(1, len(text) - 2)):
            gram = text[index : index + 3]
            digest = hashlib.sha256(gram.encode()).digest()
            column = int.from_bytes(digest[:4], "little") % dimension
            sign = 1.0 if digest[4] & 1 else -1.0
            table[row, column] += sign
        norm = np.linalg.norm(table[row])
        if norm:
            table[row] /= norm
    return table


def representation_tables(
    task: TaskData,
    split_index: int,
    bandwidth: float,
    *,
    dimension: int = 32,
    corruption_index: int | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Return one feature row per state; never reads ``task.rows.target``."""
    state_parts = split_state_indices(task, split_index)
    training_states = state_parts["train"]
    landmarks = farthest_point_landmarks(
        task.distance, training_states, dimension, state_ids=task.state_ids
    )
    weights = state_weight_table(task.distance, landmarks, bandwidth, kernel="gaussian", normalization="partition")
    affinities = kernel_affinity(task.distance[:, landmarks] / bandwidth, "gaussian")
    tables: dict[str, np.ndarray] = {
        "mpe": weights.astype(np.float32),
        "similarity_same_metric": weights.astype(np.float32),
        "similarity_unnormalized": affinities.astype(np.float32),
        "rbf_normalized": weights.astype(np.float32),
        "rbf_unnormalized": affinities.astype(np.float32),
    }
    nystrom, rank = nystrom_features(task.distance, np.arange(len(task.states)), landmarks, bandwidth)
    tables["nystrom"] = nystrom.astype(np.float32)
    tables["unknown_embedding"] = categorical_unknown_table(len(task.states), training_states)
    tables["support_complete_categorical"] = tables["unknown_embedding"].copy()

    codes = np.arange(len(task.states), dtype=np.float64)
    row_states = task.row_state_indices()
    training_row_codes = codes[row_states[np.isin(row_states, training_states)]]
    quantile_knots = np.quantile(training_row_codes, np.linspace(0.0, 1.0, dimension + 1))
    uniform_knots = np.linspace(training_row_codes.min(), training_row_codes.max(), dimension + 1)
    tables["q_ple"] = piecewise_linear_table(codes, quantile_knots, dimension)
    tables["uniform_ple"] = piecewise_linear_table(codes, uniform_knots, dimension)

    eq = equality_distance(len(task.states))
    eq_landmarks = farthest_point_landmarks(eq, training_states, dimension, state_ids=task.state_ids)
    tables["mpe_equality"] = state_weight_table(eq, eq_landmarks, 1.0).astype(np.float32)
    if corruption_index is not None:
        permutation = codebook_permutation(len(task.states), stable_corruption_seed(task.name, split_index, corruption_index))
        corrupt = corrupt_state_association(task.distance, permutation)
        corrupt_landmarks = farthest_point_landmarks(corrupt, training_states, dimension, state_ids=task.state_ids)
        tables[f"mpe_corrupt_{corruption_index}"] = state_weight_table(
            corrupt, corrupt_landmarks, bandwidth, kernel="gaussian", normalization="partition"
        ).astype(np.float32)

    if "path_json" in task.states.columns:
        tables["hierarchy_shortest_path_similarity"] = weights.astype(np.float32)
        tables["tree_rbf"] = affinities.astype(np.float32)
        tables.update(hierarchy_specialists(task, landmarks, dimension))
    if "coordinates" in task.arrays:
        tables.update(geographic_specialists(task, training_states, landmarks, bandwidth, dimension))
    if task.manifest["source_unit"] == "STRING_BENCHMARK":
        tables["character_3gram_hash"] = string_hash_table(task.state_ids)
    metadata = {
        "landmark_indices": landmarks.tolist(),
        "landmark_state_ids": [task.state_ids[index] for index in landmarks],
        "bandwidth": float(bandwidth),
        "nystrom_effective_rank": rank,
        "dimensions": {name: int(value.shape[1]) for name, value in tables.items()},
    }
    return tables, metadata


def corrupted_mpe_table(
    task: TaskData,
    split_index: int,
    bandwidth: float,
    corruption_index: int,
    dimension: int = 32,
) -> np.ndarray:
    training_states = split_state_indices(task, split_index)["train"]
    permutation = codebook_permutation(
        len(task.states), stable_corruption_seed(task.name, split_index, corruption_index)
    )
    corrupt = corrupt_state_association(task.distance, permutation)
    landmarks = farthest_point_landmarks(
        corrupt, training_states, dimension, state_ids=task.state_ids
    )
    return state_weight_table(
        corrupt, landmarks, bandwidth, kernel="gaussian", normalization="partition"
    ).astype(np.float32)


def stable_corruption_seed(task: str, split_index: int, corruption_index: int) -> int:
    digest = hashlib.sha256(f"20260829|{task}|{split_index}|corrupt|{corruption_index}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**32 - 1)


def candidate_bandwidths(task: TaskData, split_index: int) -> np.ndarray:
    return bandwidth_grid(task.distance, split_state_indices(task, split_index)["train"])
