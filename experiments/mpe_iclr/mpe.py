#!/usr/bin/env python3
"""Core target-independent metric representations for the frozen MPE program."""

from __future__ import annotations

import hashlib
import math
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.linalg import eigh
from scipy.sparse import csgraph


EPS = 1e-12
EARTH_RADIUS_KM = 6371.0088


def stable_seed(*parts: object) -> int:
    payload = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32 - 1)


def stable_order(values: Sequence[object], seed: int) -> np.ndarray:
    keys = [hashlib.sha256(f"{seed}|{value}".encode("utf-8")).digest() for value in values]
    return np.asarray(sorted(range(len(values)), key=lambda i: (keys[i], str(values[i]))), dtype=np.int64)


def validate_distance_matrix(
    distance: np.ndarray,
    *,
    triangle: bool = True,
    atol: float = 1e-10,
    max_triangle_states: int = 512,
) -> dict[str, float | bool | int]:
    d = np.asarray(distance, dtype=np.float64)
    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise AssertionError(f"distance must be square, got {d.shape}")
    if not np.isfinite(d).all():
        raise AssertionError("distance contains non-finite values")
    if float(d.min()) < -atol:
        raise AssertionError("distance contains negative values")
    symmetry_error = float(np.max(np.abs(d - d.T)))
    diagonal_error = float(np.max(np.abs(np.diag(d))))
    if symmetry_error > atol:
        raise AssertionError(f"metric symmetry error {symmetry_error}")
    if diagonal_error > atol:
        raise AssertionError(f"metric diagonal error {diagonal_error}")
    triangle_violation = 0.0
    checked = 0
    if triangle:
        if len(d) <= max_triangle_states:
            indices = np.arange(len(d))
        else:
            indices = stable_order(list(range(len(d))), stable_seed("triangle", len(d)))[:max_triangle_states]
        sub = d[np.ix_(indices, indices)]
        for k in range(len(sub)):
            violation = sub - (sub[:, k, None] + sub[k, None, :])
            triangle_violation = max(triangle_violation, float(np.max(violation)))
            checked += int(violation.size)
        if triangle_violation > atol:
            raise AssertionError(f"triangle inequality violation {triangle_violation}")
    return {
        "states": int(len(d)),
        "symmetry_error": symmetry_error,
        "diagonal_error": diagonal_error,
        "max_triangle_violation": triangle_violation,
        "triangle_entries_checked": checked,
        "passed": True,
    }


def equality_distance(n_states: int) -> np.ndarray:
    return np.ones((n_states, n_states), dtype=np.float64) - np.eye(n_states, dtype=np.float64)


def path_distance(n_states: int) -> np.ndarray:
    x = np.arange(n_states, dtype=np.float64)
    return np.abs(x[:, None] - x[None, :])


def cycle_distance(n_states: int) -> np.ndarray:
    path = path_distance(n_states)
    return np.minimum(path, n_states - path)


def graph_shortest_path(adjacency: np.ndarray, *, weighted: bool = False) -> np.ndarray:
    graph = np.asarray(adjacency, dtype=np.float64)
    d = csgraph.shortest_path(graph, directed=False, unweighted=not weighted)
    if not np.isfinite(d).all():
        raise AssertionError("graph metric is disconnected")
    return np.asarray(d, dtype=np.float64)


def haversine_distance(coordinates_degrees: np.ndarray) -> np.ndarray:
    coordinates = np.asarray(coordinates_degrees, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("coordinates must have columns [latitude, longitude]")
    latitude = np.deg2rad(coordinates[:, 0])
    longitude = np.deg2rad(coordinates[:, 1])
    dlat = latitude[:, None] - latitude[None, :]
    dlon = longitude[:, None] - longitude[None, :]
    a = np.sin(dlat / 2.0) ** 2 + (
        np.cos(latitude[:, None]) * np.cos(latitude[None, :]) * np.sin(dlon / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def coordinates_to_unit_sphere(coordinates_degrees: np.ndarray) -> np.ndarray:
    coordinates = np.asarray(coordinates_degrees, dtype=np.float64)
    latitude = np.deg2rad(coordinates[:, 0])
    longitude = np.deg2rad(coordinates[:, 1])
    return np.column_stack(
        [np.cos(latitude) * np.cos(longitude), np.cos(latitude) * np.sin(longitude), np.sin(latitude)]
    )


def farthest_point_landmarks(
    distance: np.ndarray,
    candidates: Sequence[int],
    m: int,
    *,
    state_ids: Sequence[object] | None = None,
) -> np.ndarray:
    d = np.asarray(distance, dtype=np.float64)
    candidate = np.unique(np.asarray(candidates, dtype=np.int64))
    if len(candidate) == 0:
        raise ValueError("at least one training-state candidate is required")
    count = min(int(m), len(candidate))
    identifiers = np.asarray(state_ids if state_ids is not None else np.arange(len(d)), dtype=object)
    within = d[np.ix_(candidate, candidate)]
    sums = within.sum(axis=1)
    first_options = np.flatnonzero(np.isclose(sums, sums.min(), rtol=0.0, atol=1e-14))
    first_pos = min(first_options, key=lambda position: str(identifiers[candidate[position]]))
    selected = [int(candidate[first_pos])]
    while len(selected) < count:
        nearest = d[np.ix_(candidate, np.asarray(selected, dtype=np.int64))].min(axis=1)
        nearest[np.isin(candidate, selected)] = -np.inf
        maximum = float(np.max(nearest))
        options = np.flatnonzero(np.isclose(nearest, maximum, rtol=0.0, atol=1e-14))
        position = min(options, key=lambda pos: str(identifiers[candidate[pos]]))
        selected.append(int(candidate[position]))
    result = np.asarray(selected, dtype=np.int64)
    if len(np.unique(result)) != len(result):
        raise AssertionError("duplicate landmark selected")
    if not np.isin(result, candidate).all():
        raise AssertionError("landmark outside training states")
    return result


def k_medoids_landmarks(
    distance: np.ndarray,
    candidates: Sequence[int],
    m: int,
    *,
    max_iter: int = 100,
) -> np.ndarray:
    """Deterministic PAM-style medoids initialized by farthest-point traversal."""
    d = np.asarray(distance, dtype=np.float64)
    candidate = np.unique(np.asarray(candidates, dtype=np.int64))
    medoids = farthest_point_landmarks(d, candidate, m)
    for _ in range(max_iter):
        assignment = np.argmin(d[np.ix_(candidate, medoids)], axis=1)
        updated: list[int] = []
        for cluster in range(len(medoids)):
            members = candidate[assignment == cluster]
            if len(members) == 0:
                updated.append(int(medoids[cluster]))
                continue
            costs = d[np.ix_(members, members)].sum(axis=1)
            choices = members[np.isclose(costs, costs.min(), rtol=0.0, atol=1e-14)]
            updated.append(int(np.min(choices)))
        new = np.asarray(updated, dtype=np.int64)
        if len(np.unique(new)) != len(new):
            new = farthest_point_landmarks(d, candidate, len(medoids))
        if np.array_equal(new, medoids):
            break
        medoids = new
    return medoids


def random_landmarks(candidates: Sequence[int], m: int, seed: int) -> np.ndarray:
    candidate = np.unique(np.asarray(candidates, dtype=np.int64))
    order = stable_order(candidate.tolist(), seed)
    return candidate[order[: min(m, len(candidate))]]


def frequency_landmarks(
    candidates: Sequence[int], frequencies: Mapping[int, int], m: int
) -> np.ndarray:
    candidate = np.unique(np.asarray(candidates, dtype=np.int64))
    ordered = sorted(candidate.tolist(), key=lambda value: (-int(frequencies.get(int(value), 0)), int(value)))
    return np.asarray(ordered[: min(m, len(ordered))], dtype=np.int64)


def cover_radius(distance: np.ndarray, support: Sequence[int], landmarks: Sequence[int]) -> float:
    support_array = np.asarray(support, dtype=np.int64)
    landmark_array = np.asarray(landmarks, dtype=np.int64)
    return float(np.max(np.min(np.asarray(distance)[np.ix_(support_array, landmark_array)], axis=1)))


def bandwidth_grid(distance: np.ndarray, training_states: Sequence[int]) -> np.ndarray:
    d = np.asarray(distance, dtype=np.float64)
    states = np.unique(np.asarray(training_states, dtype=np.int64))
    if len(states) < 2:
        return np.asarray([1.0], dtype=np.float64)
    within = d[np.ix_(states, states)].copy()
    np.fill_diagonal(within, np.inf)
    nearest = np.min(within, axis=1)
    nearest = nearest[np.isfinite(nearest) & (nearest > EPS)]
    upper = within[np.triu_indices(len(states), 1)]
    upper = upper[np.isfinite(upper) & (upper > EPS)]
    base = float(np.median(nearest)) if len(nearest) else float(np.median(upper))
    candidates = [0.5 * base, base, 2.0 * base, 4.0 * base]
    if len(upper):
        candidates.extend(np.quantile(upper, [0.25, 0.5, 0.75]).tolist())
    values = np.asarray(sorted({round(float(value), 14) for value in candidates if value > EPS}), dtype=np.float64)
    return values if len(values) else np.asarray([1.0], dtype=np.float64)


def kernel_affinity(scaled_distance: np.ndarray, kernel: str) -> np.ndarray:
    r = np.asarray(scaled_distance, dtype=np.float64)
    if np.any(r < -EPS):
        raise ValueError("scaled distances must be nonnegative")
    if kernel == "gaussian":
        return np.exp(-0.5 * r**2)
    if kernel == "laplacian":
        return np.exp(-r)
    if kernel == "triangular":
        return np.maximum(0.0, 1.0 - r)
    if kernel == "inverse_distance":
        return 1.0 / (1.0 + r)
    raise KeyError(kernel)


def mpe_weights(
    distances_to_landmarks: np.ndarray,
    bandwidth: float,
    *,
    kernel: str = "gaussian",
    normalization: str = "partition",
    sparse_k: int | None = None,
) -> np.ndarray:
    distances = np.asarray(distances_to_landmarks, dtype=np.float64)
    if distances.ndim != 2 or distances.shape[1] == 0:
        raise ValueError("distance-to-landmark matrix must be nonempty and 2-D")
    if not np.isfinite(distances).all() or float(np.min(distances)) < -EPS:
        raise ValueError("invalid distances")
    h = float(bandwidth)
    if not math.isfinite(h) or h <= 0:
        raise ValueError("bandwidth must be finite and positive")
    if normalization == "softmax_distance":
        logits = -distances / h
        if sparse_k is not None and sparse_k < distances.shape[1]:
            keep = np.argpartition(distances, sparse_k - 1, axis=1)[:, :sparse_k]
            mask = np.zeros_like(distances, dtype=bool)
            np.put_along_axis(mask, keep, True, axis=1)
            logits = np.where(mask, logits, -np.inf)
        logits -= np.max(logits, axis=1, keepdims=True)
        weights = np.exp(logits)
        weights /= weights.sum(axis=1, keepdims=True)
        return weights
    # Gaussian and Laplacian kernels are mathematically strictly positive, but
    # their raw floating-point affinities can all underflow for a state far
    # from every landmark (for example, a remote airport).  Normalize these
    # two kernels in log space.  Subtracting a rowwise constant leaves the
    # partition-of-unity weights exactly unchanged.
    if normalization == "partition" and kernel in {"gaussian", "laplacian"}:
        scaled = distances / h
        logits = -0.5 * scaled**2 if kernel == "gaussian" else -scaled
        if sparse_k is not None and sparse_k < distances.shape[1]:
            keep = np.argpartition(distances, sparse_k - 1, axis=1)[:, :sparse_k]
            mask = np.zeros_like(distances, dtype=bool)
            np.put_along_axis(mask, keep, True, axis=1)
            logits = np.where(mask, logits, -np.inf)
        logits -= np.max(logits, axis=1, keepdims=True)
        weights = np.exp(logits)
        denominator = weights.sum(axis=1, keepdims=True)
        if np.any(denominator <= 0):
            raise FloatingPointError("kernel partition has zero normalizer")
        weights /= denominator
        if not np.allclose(weights.sum(axis=1), 1.0, atol=1e-12):
            raise AssertionError("partition weights do not sum to one")
        return weights
    affinities = kernel_affinity(distances / h, kernel)
    if sparse_k is not None and sparse_k < distances.shape[1]:
        keep = np.argpartition(distances, sparse_k - 1, axis=1)[:, :sparse_k]
        mask = np.zeros_like(distances, dtype=bool)
        np.put_along_axis(mask, keep, True, axis=1)
        affinities = np.where(mask, affinities, 0.0)
    if normalization == "unnormalized":
        return affinities
    if normalization != "partition":
        raise KeyError(normalization)
    denominator = affinities.sum(axis=1, keepdims=True)
    if np.any(denominator <= 0):
        raise FloatingPointError("kernel partition has zero normalizer")
    weights = affinities / denominator
    if not np.allclose(weights.sum(axis=1), 1.0, atol=1e-12):
        raise AssertionError("partition weights do not sum to one")
    return weights


def state_weight_table(
    distance: np.ndarray,
    landmarks: Sequence[int],
    bandwidth: float,
    **kwargs: object,
) -> np.ndarray:
    d = np.asarray(distance, dtype=np.float64)
    landmark_array = np.asarray(landmarks, dtype=np.int64)
    return mpe_weights(d[:, landmark_array], bandwidth, **kwargs)


def nystrom_features(
    distance: np.ndarray,
    states: Sequence[int],
    landmarks: Sequence[int],
    bandwidth: float,
    *,
    eigen_floor: float = 1e-8,
) -> tuple[np.ndarray, int]:
    d = np.asarray(distance, dtype=np.float64)
    state_array = np.asarray(states, dtype=np.int64)
    landmark_array = np.asarray(landmarks, dtype=np.int64)
    k_xl = kernel_affinity(d[np.ix_(state_array, landmark_array)] / bandwidth, "gaussian")
    k_ll = kernel_affinity(d[np.ix_(landmark_array, landmark_array)] / bandwidth, "gaussian")
    values, vectors = eigh((k_ll + k_ll.T) / 2.0)
    maximum = max(float(np.max(values)), EPS)
    keep = values > eigen_floor * maximum
    inverse_root = (vectors[:, keep] / np.sqrt(values[keep])[None, :]) @ vectors[:, keep].T
    return k_xl @ inverse_root, int(np.sum(keep))


def weighted_metric_radius(weights: np.ndarray, distances_to_landmarks: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    d = np.asarray(distances_to_landmarks, dtype=np.float64)
    if w.shape != d.shape:
        raise ValueError("weights and distances must have the same shape")
    return np.sum(w * d, axis=1)


def nearest_support_distance(
    distance: np.ndarray, query_states: Sequence[int], training_states: Sequence[int]
) -> np.ndarray:
    return np.min(
        np.asarray(distance)[np.ix_(np.asarray(query_states, dtype=np.int64), np.asarray(training_states, dtype=np.int64))],
        axis=1,
    )


def corrupt_state_association(distance: np.ndarray, permutation: Sequence[int]) -> np.ndarray:
    d = np.asarray(distance, dtype=np.float64)
    permutation_array = np.asarray(permutation, dtype=np.int64)
    if sorted(permutation_array.tolist()) != list(range(len(d))):
        raise ValueError("corruption must be a bijection")
    result = d[np.ix_(permutation_array, permutation_array)]
    if not np.allclose(np.sort(d.ravel()), np.sort(result.ravel()), atol=0.0):
        raise AssertionError("corruption did not preserve distance distribution")
    return result


def partial_permutation(n_states: int, fraction: float, seed: int) -> np.ndarray:
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must lie in [0,1]")
    permutation = np.arange(n_states, dtype=np.int64)
    count = min(n_states, max(0, int(round(fraction * n_states))))
    if count <= 1:
        return permutation
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(n_states, size=count, replace=False))
    shuffled = selected.copy()
    for _ in range(100):
        rng.shuffle(shuffled)
        if not np.array_equal(shuffled, selected):
            break
    permutation[selected] = shuffled
    return permutation


def codebook_permutation(n_states: int, seed: int) -> np.ndarray:
    permutation = np.arange(n_states, dtype=np.int64)
    np.random.default_rng(seed).shuffle(permutation)
    return permutation


def assert_disjoint_states(parts: Mapping[str, Iterable[object]]) -> None:
    sets = {name: set(values) for name, values in parts.items()}
    names = list(sets)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlap = sets[left] & sets[right]
            if overlap:
                raise AssertionError(f"state leakage between {left} and {right}: {list(overlap)[:5]}")


def make_state_partition(
    states: Sequence[object],
    seed: int,
    proportions: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> dict[str, np.ndarray]:
    unique = np.asarray(sorted(set(states), key=str), dtype=object)
    if len(unique) < 15:
        raise ValueError("at least 15 eligible states are required for a 5/5/5 split")
    order = stable_order(unique.tolist(), seed)
    shuffled = unique[order]
    n_train = max(5, int(math.floor(proportions[0] * len(unique))))
    n_val = max(5, int(math.floor(proportions[1] * len(unique))))
    if n_train + n_val > len(unique) - 5:
        n_train = len(unique) - n_val - 5
    parts = {
        "train": shuffled[:n_train],
        "validation": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }
    assert_disjoint_states(parts)
    return parts


def state_balanced_mean(losses: np.ndarray, states: Sequence[object]) -> float:
    values = np.asarray(losses, dtype=np.float64)
    state_array = np.asarray(states, dtype=object)
    if len(values) != len(state_array):
        raise ValueError("loss and state lengths differ")
    means = [float(np.mean(values[state_array == state])) for state in sorted(set(state_array), key=str)]
    return float(np.mean(means))


def state_loss_table(losses: np.ndarray, states: Sequence[object]) -> dict[object, float]:
    values = np.asarray(losses, dtype=np.float64)
    state_array = np.asarray(states, dtype=object)
    return {state: float(np.mean(values[state_array == state])) for state in sorted(set(state_array), key=str)}


def normalize_string(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value)).lower()
    return " ".join(text.split())


def character_ngrams(value: object, n: int = 3) -> set[str]:
    text = f"  {normalize_string(value)}  "
    return {text[index : index + n] for index in range(max(1, len(text) - n + 1))}


def trigram_jaccard_distance(values: Sequence[object]) -> np.ndarray:
    grams = [character_ngrams(value, 3) for value in values]
    d = np.zeros((len(grams), len(grams)), dtype=np.float64)
    for i in range(len(grams)):
        for j in range(i + 1, len(grams)):
            union = grams[i] | grams[j]
            similarity = len(grams[i] & grams[j]) / max(1, len(union))
            d[i, j] = d[j, i] = 1.0 - similarity
    return d


def levenshtein_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, left_character in enumerate(left, 1):
        current = [i]
        for j, right_character in enumerate(right, 1):
            current.append(
                min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left_character != right_character))
            )
        previous = current
    return previous[-1]


def normalized_levenshtein_distance(values: Sequence[object]) -> np.ndarray:
    text = [normalize_string(value) for value in values]
    d = np.zeros((len(text), len(text)), dtype=np.float64)
    for i in range(len(text)):
        for j in range(i + 1, len(text)):
            d[i, j] = d[j, i] = levenshtein_distance(text[i], text[j]) / max(1, len(text[i]), len(text[j]))
    return d


@dataclass(frozen=True)
class MetricField:
    name: str
    state_ids: tuple[object, ...]
    distance: np.ndarray
    coordinates: np.ndarray | None = None
    adjacency: np.ndarray | None = None

    def validate(self, *, triangle: bool = True) -> dict[str, float | bool | int]:
        if len(self.state_ids) != len(self.distance):
            raise AssertionError("state IDs and distance matrix disagree")
        if len(set(self.state_ids)) != len(self.state_ids):
            raise AssertionError("duplicate state IDs")
        return validate_distance_matrix(self.distance, triangle=triangle)

    @property
    def state_to_index(self) -> dict[object, int]:
        return {state: index for index, state in enumerate(self.state_ids)}
