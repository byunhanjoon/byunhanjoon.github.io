from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans


def rbf_kernel(a: np.ndarray, b: np.ndarray | None = None, gamma: float | None = None) -> np.ndarray:
    if b is None:
        b = a
    sq = cdist(a, b, metric="sqeuclidean")
    if gamma is None:
        positive = sq[sq > 0]
        gamma = 1.0 / max(float(np.median(positive)), 1e-8)
    return np.exp(-gamma * sq)


def cosine_similarity(a: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(a, axis=1, keepdims=True)
    unit = a / np.maximum(norm, 1e-8)
    return unit @ unit.T


def enforce_coverage(order: np.ndarray, y: np.ndarray, k: int, classification: bool) -> np.ndarray:
    order = np.asarray(order, dtype=int)
    if not classification:
        return np.sort(order[:k])
    chosen: list[int] = []
    for cls in np.unique(y):
        matches = order[y[order] == cls]
        if len(matches) and len(chosen) < k:
            chosen.append(int(matches[0]))
    chosen_set = set(chosen)
    chosen.extend(int(i) for i in order if i not in chosen_set and len(chosen) < k)
    return np.sort(np.asarray(chosen, dtype=int))


def topk(scores: np.ndarray, y: np.ndarray, k: int, classification: bool) -> np.ndarray:
    return enforce_coverage(np.argsort(-scores), y, k, classification)


def kcenter(z: np.ndarray, k: int, y: np.ndarray, classification: bool) -> np.ndarray:
    center = z.mean(axis=0)
    first = int(np.argmin(np.linalg.norm(z - center, axis=1)))
    chosen = [first]
    minimum = cdist(z, z[[first]]).ravel()
    while len(chosen) < k:
        candidate = int(np.argmax(minimum))
        chosen.append(candidate)
        minimum = np.minimum(minimum, cdist(z, z[[candidate]]).ravel())
    # Coverage correction uses farthest-point order, preserving as much geometry as possible.
    return enforce_coverage(np.asarray(chosen + [i for i in np.argsort(-minimum) if i not in chosen]), y, k, classification)


def kmedoids_like(z: np.ndarray, k: int, y: np.ndarray, classification: bool, seed: int = 0) -> np.ndarray:
    # Lloyd-style alternate optimization with actual observed medoids. This is
    # an approximate k-medoids baseline, not a nearest-to-k-means-centroid proxy.
    distance = cdist(z, z)
    rng = np.random.default_rng(seed)
    medoids = [int(rng.integers(len(z)))]
    nearest = distance[:, medoids[0]].copy()
    while len(medoids) < k:
        candidate = int(np.argmax(nearest))
        medoids.append(candidate)
        nearest = np.minimum(nearest, distance[:, candidate])
    for _ in range(20):
        assignment = np.argmin(distance[:, medoids], axis=1)
        updated: list[int] = []
        used: set[int] = set()
        for cluster in range(k):
            members = np.flatnonzero(assignment == cluster)
            if len(members):
                costs = distance[np.ix_(members, members)].sum(axis=1)
                candidate = int(members[np.argmin(costs)])
            else:
                available = np.asarray([i for i in range(len(z)) if i not in used], dtype=int)
                candidate = int(available[np.argmax(distance[np.ix_(available, list(used))].min(axis=1))]) if used else int(available[0])
            if candidate in used:
                available = np.asarray([i for i in range(len(z)) if i not in used], dtype=int)
                candidate = int(available[np.argmax(distance[np.ix_(available, medoids)].min(axis=1))])
            updated.append(candidate)
            used.add(candidate)
        if set(updated) == set(medoids):
            medoids = updated
            break
        medoids = updated
    remaining = [int(i) for i in np.argsort(distance[:, medoids].min(axis=1)) if int(i) not in set(medoids)]
    return enforce_coverage(np.asarray(medoids + remaining), y, k, classification)


def latent_medoid_like(
    candidate_z: np.ndarray,
    query_z: np.ndarray,
    k: int,
    y: np.ndarray,
    classification: bool,
    seed: int = 0,
) -> np.ndarray:
    centers = KMeans(n_clusters=min(k, len(query_z)), n_init=10, random_state=seed).fit(query_z).cluster_centers_
    distance = cdist(centers, candidate_z)
    order: list[int] = []
    used: set[int] = set()
    for row in distance:
        for idx in np.argsort(row):
            if int(idx) not in used:
                order.append(int(idx))
                used.add(int(idx))
                break
    order.extend(int(i) for i in np.argsort(np.min(distance, axis=0)) if int(i) not in used)
    return enforce_coverage(np.asarray(order), y, k, classification)


def nearest_query_cluster(
    candidate_z: np.ndarray,
    query_z: np.ndarray,
    k: int,
    y: np.ndarray,
    classification: bool,
    seed: int = 0,
) -> np.ndarray:
    n_clusters = min(max(2, int(np.sqrt(k))), len(query_z))
    fitted = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit(query_z)
    per_cluster = np.bincount(fitted.labels_, minlength=n_clusters)
    quota = np.maximum(1, np.floor(k * per_cluster / per_cluster.sum()).astype(int))
    while quota.sum() > k:
        quota[np.argmax(quota)] -= 1
    while quota.sum() < k:
        quota[np.argmax(per_cluster / np.maximum(quota, 1))] += 1
    order: list[int] = []
    used: set[int] = set()
    distances = cdist(fitted.cluster_centers_, candidate_z)
    for cluster, count in enumerate(quota):
        added = 0
        for idx in np.argsort(distances[cluster]):
            if int(idx) not in used:
                order.append(int(idx))
                used.add(int(idx))
                added += 1
                if added >= count:
                    break
    order.extend(int(i) for i in np.argsort(np.min(distances, axis=0)) if int(i) not in used)
    return enforce_coverage(np.asarray(order), y, k, classification)


def mmd_crumb_like(
    candidate_z: np.ndarray,
    query_z: np.ndarray,
    k: int,
    y: np.ndarray,
    classification: bool,
) -> np.ndarray:
    kcc = rbf_kernel(candidate_z)
    kcq = rbf_kernel(candidate_z, query_z)
    target = kcq.mean(axis=1)
    selected: list[int] = []
    available = np.ones(len(candidate_z), dtype=bool)
    running = np.zeros(len(candidate_z))
    for step in range(k):
        # Kernel herding minimizes MMD to the selector-query distribution.
        score = target - running / max(step, 1)
        score[~available] = -np.inf
        idx = int(np.argmax(score))
        selected.append(idx)
        available[idx] = False
        running += kcc[:, idx]
    order = np.asarray(selected + [int(i) for i in np.argsort(-target) if available[int(i)]])
    return enforce_coverage(order, y, k, classification)


def dpp_logdet(z: np.ndarray, k: int, y: np.ndarray, classification: bool) -> np.ndarray:
    kernel = rbf_kernel(z) + np.eye(len(z)) * 1e-6
    selected: list[int] = []
    remaining = set(range(len(z)))
    # Exact marginal log-determinants are cheap at n=256, k<=64 and robust for a kill experiment.
    current_logdet = 0.0
    for _ in range(k):
        best, best_gain = None, -np.inf
        for idx in remaining:
            proposed = selected + [idx]
            sign, value = np.linalg.slogdet(kernel[np.ix_(proposed, proposed)])
            gain = value - current_logdet if sign > 0 else -np.inf
            if gain > best_gain:
                best, best_gain = idx, gain
        assert best is not None
        selected.append(best)
        remaining.remove(best)
        current_logdet += best_gain
    order = np.asarray(selected + list(remaining), dtype=int)
    return enforce_coverage(order, y, k, classification)


def pairwise_objective(indices: np.ndarray, additive: np.ndarray, pair: np.ndarray) -> float:
    indices = np.asarray(indices, dtype=int)
    return float(additive[indices].sum() + np.triu(pair[np.ix_(indices, indices)], 1).sum())


def pairwise_greedy(
    additive: np.ndarray,
    pair: np.ndarray,
    k: int,
    y: np.ndarray,
    classification: bool,
) -> np.ndarray:
    selected: list[int] = []
    remaining = set(range(len(additive)))
    if classification:
        for cls in np.unique(y):
            candidates = np.flatnonzero(y == cls)
            score = additive[candidates]
            if selected:
                score = score + pair[np.ix_(candidates, selected)].sum(axis=1)
            idx = int(candidates[np.argmax(score)])
            selected.append(idx)
            remaining.remove(idx)
    while len(selected) < k:
        candidates = np.fromiter(remaining, dtype=int)
        score = additive[candidates]
        if selected:
            score = score + pair[np.ix_(candidates, selected)].sum(axis=1)
        idx = int(candidates[np.argmax(score)])
        selected.append(idx)
        remaining.remove(idx)
    return np.sort(np.asarray(selected, dtype=int))


def one_swap(
    initial: np.ndarray,
    additive: np.ndarray,
    pair: np.ndarray,
    y: np.ndarray,
    classification: bool,
    max_rounds: int = 50,
) -> np.ndarray:
    selected = set(map(int, initial))
    all_indices = set(range(len(additive)))
    current = pairwise_objective(np.fromiter(selected, dtype=int), additive, pair)
    for _ in range(max_rounds):
        best_gain, best_swap = 1e-10, None
        for old in tuple(selected):
            if classification and np.sum(y[np.fromiter(selected, dtype=int)] == y[old]) <= 1:
                continue
            without = selected - {old}
            for new in all_indices - selected:
                proposal = np.fromiter(without | {new}, dtype=int)
                gain = pairwise_objective(proposal, additive, pair) - current
                if gain > best_gain:
                    best_gain, best_swap = gain, (old, new)
        if best_swap is None:
            break
        selected.remove(best_swap[0])
        selected.add(best_swap[1])
        current += best_gain
    return np.sort(np.fromiter(selected, dtype=int))


def geometry_pair_matrices(z: np.ndarray) -> dict[str, np.ndarray]:
    cosine = cosine_similarity(z)
    rbf = rbf_kernel(z)
    distance = cdist(z, z)
    positive = distance[distance > 0]
    threshold = float(np.quantile(positive, 0.1))
    neighbor = (distance <= threshold).astype(float)
    np.fill_diagonal(neighbor, 0.0)
    return {
        "cosine_diversity": -cosine,
        "rbf_diversity": -rbf,
        "euclidean_neighbor_diversity": -neighbor,
    }


def complementarity_matrix(target_bins: np.ndarray) -> np.ndarray:
    bins = np.asarray(target_bins)
    if len(np.unique(bins)) <= 2:
        pair = (bins[:, None] != bins[None, :]).astype(float) - (bins[:, None] == bins[None, :]).astype(float)
    else:
        scale = max(float(np.ptp(bins)), 1.0)
        pair = np.abs(bins[:, None] - bins[None, :]) / scale
        pair -= (bins[:, None] == bins[None, :]).astype(float) * 0.25
    np.fill_diagonal(pair, 0.0)
    return pair


def aggregate_pair_feature(X: np.ndarray, pair: np.ndarray) -> np.ndarray:
    return 0.5 * ((X @ pair) * X).sum(axis=1)
