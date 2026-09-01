"""Target-free invariant and partially invariant feature interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .common import bd, ensure_finite


EPS = 1e-12


@dataclass
class BlockTransform:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    metadata: dict[str, Any]


def relative_error(reference: np.ndarray, candidate: np.ndarray) -> float:
    return float(np.linalg.norm(reference - candidate) / max(np.linalg.norm(reference), EPS))


def gram_pivot_indices(values: np.ndarray, count: int) -> tuple[np.ndarray, int]:
    """Select row anchors by deterministic pivoted Cholesky of ``ZZ^T``.

    The implementation forms one Gram column per pivot instead of materializing
    the full n-by-n Gram matrix.  Every decision depends only on inner products,
    making the selected row identities basis invariant.
    """

    z = np.asarray(values, dtype=np.float64)
    n = len(z)
    count = min(int(count), n)
    diagonal = np.einsum("ij,ij->i", z, z)
    residual = diagonal.copy()
    factors = np.zeros((n, count), dtype=np.float64)
    chosen: list[int] = []
    available = np.ones(n, dtype=bool)
    # RBF blocks can be nominally full-rank while their final Cholesky
    # residuals are at floating-point noise scale.  Stop before those pivots
    # and use deterministic row-index padding.  Likewise, treat nearly tied
    # residuals as ties and select the lowest row index.  Both rules depend
    # only on the invariant Gram diagonal/residual, but are robust to the
    # O(1e-15) roundoff introduced by Z -> ZQ.
    tolerance = max(float(diagonal.max(initial=0.0)) * 1e-10, 1e-13)
    rank = 0
    for column in range(count):
        scores = np.where(available, residual, -np.inf)
        provisional = int(np.argmax(scores))
        if not np.isfinite(scores[provisional]):
            break
        maximum = float(scores[provisional])
        if maximum <= tolerance:
            remaining = np.flatnonzero(available)
            chosen.extend(int(index) for index in remaining[: count - len(chosen)])
            break
        tie_tolerance = max(tolerance, abs(maximum) * 1e-10)
        tied = np.flatnonzero(available & (residual >= maximum - tie_tolerance))
        pivot = int(tied[0])
        chosen.append(pivot)
        available[pivot] = False
        correction = factors[:, :column] @ factors[pivot, :column] if column else 0.0
        factors[:, column] = (z @ z[pivot] - correction) / np.sqrt(max(residual[pivot], tolerance))
        residual = np.maximum(residual - factors[:, column] ** 2, 0.0)
        rank += 1
    if len(chosen) < count:
        remaining = [int(index) for index in np.flatnonzero(available) if int(index) not in chosen]
        chosen.extend(remaining[: count - len(chosen)])
    return np.asarray(chosen, dtype=np.int64), rank


def random_anchor_indices(n: int, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    count = int(count)
    return rng.choice(n, size=count, replace=n < count).astype(np.int64)


def _standardize(
    train: np.ndarray, validation: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    mean = np.asarray(train, dtype=np.float64).mean(axis=0)
    scale = np.asarray(train, dtype=np.float64).std(axis=0)
    zero = scale < 1e-12
    scale[zero] = 1.0
    result = tuple((np.asarray(values, dtype=np.float64) - mean) / scale for values in (train, validation, test))
    return result[0], result[1], result[2], {
        "standardized": True,
        "constant_coordinate_count": int(zero.sum()),
    }


def _anchor_indices(
    train: np.ndarray,
    count: int,
    selection: str,
    seed: int,
) -> tuple[np.ndarray, int]:
    if selection == "gram_pivot":
        return gram_pivot_indices(train, count)
    if selection == "random_index":
        indices = random_anchor_indices(len(train), count, seed)
        return indices, int(np.linalg.matrix_rank(train[indices]))
    raise ValueError(f"unknown anchor selection {selection}")


def gram_anchor_block(
    splits: tuple[np.ndarray, np.ndarray, np.ndarray],
    count: int,
    selection: str,
    seed: int,
    normalize: bool = True,
) -> BlockTransform:
    train = np.asarray(splits[0], dtype=np.float64)
    indices, pivot_rank = _anchor_indices(train, count, selection, seed)
    anchors = train[indices]
    norm = np.linalg.norm(anchors, axis=1)
    denominator = np.maximum(norm, EPS) if normalize else np.ones_like(norm)
    outputs = tuple(np.asarray(values, dtype=np.float64) @ anchors.T / denominator for values in splits)
    standardized = _standardize(*outputs)
    return BlockTransform(
        standardized[0], standardized[1], standardized[2],
        {
            "anchor_indices": indices.tolist(),
            "anchor_count": int(count),
            "selection": selection,
            "pivot_or_anchor_rank": int(pivot_rank),
            "empirical_rank": int(np.linalg.matrix_rank(train)),
            "normalized_inner_products": bool(normalize),
            **standardized[3],
        },
    )


def gram_distance_block(
    splits: tuple[np.ndarray, np.ndarray, np.ndarray],
    count: int,
    selection: str,
    seed: int,
    kernel: str,
) -> BlockTransform:
    train = np.asarray(splits[0], dtype=np.float64)
    indices, pivot_rank = _anchor_indices(train, count, selection, seed)
    anchors = train[indices]

    def squared(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        return np.maximum(
            np.sum(values**2, axis=1, keepdims=True)
            + np.sum(anchors**2, axis=1)[None, :]
            - 2.0 * values @ anchors.T,
            0.0,
        )

    train_squared = squared(train)
    positive = train_squared[train_squared > EPS]
    median_distance = float(np.median(positive)) if len(positive) else 1.0
    raw = tuple(squared(values) for values in splits)
    if kernel == "rbf":
        raw = tuple(np.exp(-values / max(median_distance, EPS)) for values in raw)
    elif kernel != "squared":
        raise ValueError(f"unknown distance kernel {kernel}")
    standardized = _standardize(*raw)
    return BlockTransform(
        standardized[0], standardized[1], standardized[2],
        {
            "anchor_indices": indices.tolist(),
            "anchor_count": int(count),
            "selection": selection,
            "pivot_or_anchor_rank": int(pivot_rank),
            "empirical_rank": int(np.linalg.matrix_rank(train)),
            "kernel": kernel,
            "median_positive_squared_distance": median_distance,
            **standardized[3],
        },
    )


def _nystrom_from_cross_kernel(
    kernels: tuple[np.ndarray, np.ndarray, np.ndarray],
    anchor_kernel: np.ndarray,
    energy: float,
    max_rank: int,
    floor_fraction: float,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], dict[str, Any]]:
    # Equivalent orthogonal bases can change a Gram entry at ~1e-15 solely
    # through accumulation order.  Quantization far below the 1e-6 protocol
    # tolerance keeps the eigensystem input identical without altering a
    # scientifically meaningful kernel coordinate.
    stable_anchor_kernel = np.round(np.asarray(anchor_kernel, dtype=np.float64), decimals=12)
    stable_anchor_kernel = 0.5 * (stable_anchor_kernel + stable_anchor_kernel.T)
    eigenvalues, eigenvectors = np.linalg.eigh(stable_anchor_kernel)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    largest = max(float(eigenvalues[0]) if len(eigenvalues) else 0.0, EPS)
    floor = floor_fraction * largest
    positive = np.maximum(eigenvalues, 0.0)
    total = float(positive.sum())
    if total <= EPS:
        rank = 1
    else:
        cumulative = np.cumsum(positive) / total
        rank = int(np.searchsorted(cumulative, energy) + 1)
    rank = max(1, min(rank, int(max_rank), len(eigenvalues)))
    # Do not truncate through an eigenvalue multiplicity.  With the primary
    # k=8 blocks the full positive rank is at most max_rank, so extending to
    # the end of a tied group remains inside the preregistered cap.
    while (
        rank < min(int(max_rank), len(eigenvalues))
        and abs(eigenvalues[rank - 1] - eigenvalues[rank])
        <= 1e-8 * max(abs(eigenvalues[rank - 1]), EPS)
    ):
        rank += 1
    values = np.maximum(eigenvalues[:rank], floor)
    vectors = eigenvectors[:, :rank].copy()
    # Eigensolver coordinates are arbitrary inside repeated-eigenvalue
    # subspaces.  Canonicalize each such subspace using projections of the
    # invariant anchor-index axes, and use the group's mean eigenvalue.  This
    # retains the Nyström construction while making its interface deterministic
    # under exact/near-exact spectral degeneracy.
    group_start = 0
    while group_start < rank:
        group_end = group_start + 1
        while (
            group_end < rank
            and abs(values[group_end - 1] - values[group_end])
            <= 1e-8 * max(abs(values[group_end - 1]), EPS)
        ):
            group_end += 1
        if group_end - group_start > 1:
            projector = vectors[:, group_start:group_end] @ vectors[:, group_start:group_end].T
            canonical: list[np.ndarray] = []
            for axis in range(projector.shape[0]):
                candidate = projector[:, axis].copy()
                for existing in canonical:
                    candidate -= existing * float(existing @ candidate)
                norm = float(np.linalg.norm(candidate))
                if norm > 1e-9:
                    canonical.append(candidate / norm)
                if len(canonical) == group_end - group_start:
                    break
            if len(canonical) != group_end - group_start:
                raise RuntimeError("failed to canonicalize degenerate Nyström eigenspace")
            vectors[:, group_start:group_end] = np.column_stack(canonical)
            values[group_start:group_end] = float(np.mean(values[group_start:group_end]))
        group_start = group_end
    outputs = tuple(kernel @ vectors / np.sqrt(values)[None, :] for kernel in kernels)
    train_scores = outputs[0].copy()
    for component in range(rank):
        row = int(np.argmax(np.abs(train_scores[:, component])))
        if train_scores[row, component] < 0:
            vectors[:, component] *= -1
    outputs = tuple(kernel @ vectors / np.sqrt(values)[None, :] for kernel in kernels)
    standardized = _standardize(*outputs)
    return (standardized[0], standardized[1], standardized[2]), {
        "rank": rank,
        "eigenvalues": eigenvalues.tolist(),
        "lambda_floor": floor,
        "captured_energy": float(positive[:rank].sum() / max(total, EPS)),
        **standardized[3],
    }


def nystrom_gram_block(
    splits: tuple[np.ndarray, np.ndarray, np.ndarray],
    count: int,
    selection: str,
    seed: int,
    energy: float = 0.99,
    max_rank: int = 8,
    floor_fraction: float = 1e-6,
) -> BlockTransform:
    train = np.asarray(splits[0], dtype=np.float64)
    indices, pivot_rank = _anchor_indices(train, count, selection, seed)
    anchors = train[indices]
    kernels = tuple(np.asarray(values, dtype=np.float64) @ anchors.T for values in splits)
    coordinates, metadata = _nystrom_from_cross_kernel(
        kernels, anchors @ anchors.T, energy, max_rank, floor_fraction
    )
    return BlockTransform(
        coordinates[0], coordinates[1], coordinates[2],
        {
            "anchor_indices": indices.tolist(),
            "anchor_count": int(count),
            "selection": selection,
            "pivot_rank": int(pivot_rank),
            "empirical_rank": int(np.linalg.matrix_rank(train)),
            **metadata,
        },
    )


def pca_block(splits: tuple[np.ndarray, np.ndarray, np.ndarray]) -> BlockTransform:
    train = np.asarray(splits[0], dtype=np.float64)
    mean = train.mean(axis=0)
    _, singular, vt = np.linalg.svd(train - mean, full_matrices=False)
    vectors = vt.T
    train_scores = (train - mean) @ vectors
    for component in range(vectors.shape[1]):
        row = int(np.argmax(np.abs(train_scores[:, component])))
        if train_scores[row, component] < 0:
            vectors[:, component] *= -1
    outputs = tuple((np.asarray(values, dtype=np.float64) - mean) @ vectors for values in splits)
    ratios = singular[1:] / np.maximum(singular[:-1], EPS)
    return BlockTransform(
        outputs[0], outputs[1], outputs[2],
        {
            "singular_values": singular.tolist(),
            "empirical_rank": int(np.linalg.matrix_rank(train - mean)),
            "degenerate": bool(np.any(ratios > 0.99)),
            "closest_adjacent_ratio": float(ratios.max()) if len(ratios) else 0.0,
        },
    )


def hybrid_spectral_block(
    splits: tuple[np.ndarray, np.ndarray, np.ndarray],
    tau: float,
    anchor_count: int,
    seed: int,
    floor_fraction: float = 1e-6,
) -> BlockTransform:
    """Canonicalize separated eigendirections and Gram-map degenerate groups."""

    train = np.asarray(splits[0], dtype=np.float64)
    mean = train.mean(axis=0)
    centered = tuple(np.asarray(values, dtype=np.float64) - mean for values in splits)
    covariance = centered[0].T @ centered[0] / max(len(train), 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    boundaries = [0]
    for index in range(len(eigenvalues) - 1):
        gap = (eigenvalues[index] - eigenvalues[index + 1]) / max(abs(eigenvalues[index]), EPS)
        if gap > tau:
            boundaries.append(index + 1)
    boundaries.append(len(eigenvalues))
    groups = [(boundaries[index], boundaries[index + 1]) for index in range(len(boundaries) - 1)]
    pieces: list[list[np.ndarray]] = [[], [], []]
    records = []
    for group_index, (start, end) in enumerate(groups):
        vectors = eigenvectors[:, start:end]
        projected = tuple(values @ vectors for values in centered)
        if end - start == 1:
            sign = 1.0
            row = int(np.argmax(np.abs(projected[0][:, 0])))
            if projected[0][row, 0] < 0:
                sign = -1.0
            for split_index in range(3):
                pieces[split_index].append(projected[split_index] * sign)
            records.append({"start": start, "end": end, "kind": "canonical_singleton"})
            continue
        group_train = projected[0]
        indices, pivot_rank = gram_pivot_indices(group_train, min(anchor_count, len(group_train)))
        anchors = group_train[indices]
        kernels = tuple(values @ anchors.T for values in projected)
        coordinates, metadata = _nystrom_from_cross_kernel(
            kernels,
            anchors @ anchors.T,
            energy=0.99,
            max_rank=end - start,
            floor_fraction=floor_fraction,
        )
        for split_index in range(3):
            pieces[split_index].append(coordinates[split_index])
        records.append(
            {
                "start": start,
                "end": end,
                "kind": "invariant_gram_subspace",
                "anchor_indices": indices.tolist(),
                "pivot_rank": pivot_rank,
                **metadata,
            }
        )
    outputs = tuple(np.concatenate(parts, axis=1) for parts in pieces)
    return BlockTransform(
        outputs[0], outputs[1], outputs[2],
        {
            "tau": float(tau),
            "eigenvalues": eigenvalues.tolist(),
            "groups": records,
            "output_dimension": int(outputs[0].shape[1]),
        },
    )


def mahalanobis_gram_block(
    splits: tuple[np.ndarray, np.ndarray, np.ndarray],
    count: int,
    seed: int,
    ridge: float,
) -> BlockTransform:
    train = np.asarray(splits[0], dtype=np.float64)
    indices, pivot_rank = gram_pivot_indices(train, min(count, len(train)))
    anchors = train[indices]
    covariance = train.T @ train / max(len(train), 1)
    inverse = np.linalg.pinv(covariance + float(ridge) * np.eye(covariance.shape[0]))
    raw = tuple(np.asarray(values, dtype=np.float64) @ inverse @ anchors.T for values in splits)
    standardized = _standardize(*raw)
    return BlockTransform(
        standardized[0], standardized[1], standardized[2],
        {
            "anchor_indices": indices.tolist(),
            "pivot_rank": pivot_rank,
            "ridge": float(ridge),
            "empirical_rank": int(np.linalg.matrix_rank(train)),
            "covariance_condition": float(np.linalg.cond(covariance)),
            **standardized[3],
        },
    )


def _all_blocks(rep: Any) -> list[tuple[str, list[int], str]]:
    records: list[tuple[str, list[int], str]] = []
    for kind, mapping in (
        ("continuous", rep.feature_blocks),
        ("categorical", rep.categorical_blocks),
    ):
        records.extend((str(name), list(indices), kind) for name, indices in mapping.items())
    records.sort(key=lambda item: min(item[1]))
    used: set[int] = set()
    for name, indices, _ in records:
        overlap = used.intersection(indices)
        if overlap:
            raise RuntimeError(f"overlapping block {name}: {sorted(overlap)}")
        used.update(indices)
    # Low-cardinality numeric passthrough blocks are not retained on Representation,
    # so contiguous uncovered columns are copied verbatim below.
    return records


def build_interface(
    rep: Any,
    method: str,
    dataset_key: str,
    **parameters: Any,
) -> Any:
    """Fit a target-free interface on the representation's training partition."""

    if method == "raw":
        return rep
    output: list[list[np.ndarray]] = [[], [], []]
    columns: list[str] = []
    feature_blocks: dict[str, list[int]] = {}
    categorical_blocks: dict[str, list[int]] = {}
    audits: dict[str, Any] = {}
    cursor = 0
    records = _all_blocks(rep)
    splits_all = (rep.X_train, rep.X_validation, rep.X_test)
    for name, indices, kind in records:
        start, end = min(indices), max(indices) + 1
        if indices != list(range(start, end)):
            raise RuntimeError(f"non-contiguous block {name}")
        if cursor < start:
            for split_index, values in enumerate(splits_all):
                output[split_index].append(values[:, cursor:start])
            columns.extend(rep.columns[cursor:start])
        block_splits = tuple(values[:, start:end] for values in splits_all)
        seed = bd.stable_seed(dataset_key, method, name, parameters)
        if method == "gram_anchor":
            transformed = gram_anchor_block(
                block_splits,
                int(parameters.get("anchors", 16)),
                str(parameters.get("selection", "gram_pivot")),
                seed,
                bool(parameters.get("normalize", True)),
            )
        elif method == "gram_distance":
            transformed = gram_distance_block(
                block_splits,
                int(parameters.get("anchors", 16)),
                str(parameters.get("selection", "gram_pivot")),
                seed,
                str(parameters.get("kernel", "rbf")),
            )
        elif method == "nystrom_gram":
            transformed = nystrom_gram_block(
                block_splits,
                int(parameters.get("anchors", 16)),
                str(parameters.get("selection", "gram_pivot")),
                seed,
                float(parameters.get("energy", 0.99)),
                int(parameters.get("max_rank", 8)),
                float(parameters.get("floor_fraction", 1e-6)),
            )
        elif method == "pca":
            transformed = pca_block(block_splits)
        elif method == "hybrid_spectral":
            transformed = hybrid_spectral_block(
                block_splits,
                float(parameters.get("tau", 0.05)),
                int(parameters.get("anchors", 16)),
                seed,
                float(parameters.get("floor_fraction", 1e-6)),
            )
        elif method == "mahalanobis_gram":
            transformed = mahalanobis_gram_block(
                block_splits,
                int(parameters.get("anchors", 16)),
                seed,
                float(parameters.get("ridge", 1e-6)),
            )
        else:
            raise ValueError(f"unknown interface method {method}")
        new_start = len(columns)
        width = transformed.train.shape[1]
        target_mapping = feature_blocks if kind == "continuous" else categorical_blocks
        target_mapping[name] = list(range(new_start, new_start + width))
        columns.extend(f"{name}::{method}::{index}" for index in range(width))
        for split_index, values in enumerate((transformed.train, transformed.validation, transformed.test)):
            output[split_index].append(values)
        audits[name] = {"input_dimension": end - start, "kind": kind, **transformed.metadata}
        cursor = end
    if cursor < rep.X_train.shape[1]:
        for split_index, values in enumerate(splits_all):
            output[split_index].append(values[:, cursor:])
        columns.extend(rep.columns[cursor:])
    matrices = tuple(np.concatenate(parts, axis=1).astype(np.float64) for parts in output)
    ensure_finite(matrices, f"{dataset_key}/{method}/{rep.representation_id}")
    implementation_revision = (
        "nystrom_degeneracy_v3"
        if method == "nystrom_gram"
        else "pivot_tie_v2"
        if str(parameters.get("selection", "")) == "gram_pivot"
        or method == "hybrid_spectral"
        else "v1"
    )
    return bd.Representation(
        f"{method}__{implementation_revision}__{rep.representation_id}",
        "tournament_interface",
        rep.variant,
        rep.scope,
        rep.member,
        matrices[0],
        matrices[1],
        matrices[2],
        columns,
        feature_blocks,
        categorical_blocks,
        rep.transforms,
        {**rep.metadata, "interface": method, "parameters": parameters, "block_audits": audits},
        rep.is_reference,
    )


def audit_orbit_coordinates(reference: Any, orbit: list[Any]) -> list[dict[str, Any]]:
    records = []
    for rep in orbit:
        if rep.X_train.shape != reference.X_train.shape:
            records.append(
                {
                    "representation_id": rep.representation_id,
                    "shape_match": False,
                    "train_relative_error": float("inf"),
                    "validation_relative_error": float("inf"),
                    "test_relative_error": float("inf"),
                }
            )
            continue
        records.append(
            {
                "representation_id": rep.representation_id,
                "shape_match": True,
                "train_relative_error": relative_error(reference.X_train, rep.X_train),
                "validation_relative_error": relative_error(reference.X_validation, rep.X_validation),
                "test_relative_error": relative_error(reference.X_test, rep.X_test),
            }
        )
    return records
