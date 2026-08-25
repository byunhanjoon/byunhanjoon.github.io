"""Frozen broad-benchmark data loaders and exact representation builders."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.linalg import block_diag, qr

from .core import (
    DATA_ROOT,
    PARTS,
    Dataset,
    apply_transform,
    category_codes,
    clean_numeric,
    combine,
    condition_transform,
    contrast_block,
    diagonal_standardize,
    helmert,
    one_hot_codes,
    ple_blocks,
    quantile_numeric,
    standardize,
    whiten,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "experiments/day3/configs/broad_preregistered.json"
FINANCE_ROOT = Path("/data/tokenization_icaif_grouped_v2")


@dataclass(frozen=True)
class Representation:
    name: str
    parts: dict[str, np.ndarray]
    metadata: dict[str, object]
    reference: dict[str, np.ndarray] | None = None
    basis_transform: np.ndarray | None = None


def config() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text())


def _indices(length: int, limit: int | None, seed: int) -> np.ndarray:
    values = np.arange(length, dtype=np.int64)
    if limit is None or length <= limit:
        return values
    return np.sort(np.random.default_rng(seed).choice(values, limit, replace=False))


def _load_finance(name: str) -> Dataset:
    cfg = config()["data"]
    directory = FINANCE_ROOT / name
    info = json.loads((directory / "info.json").read_text())
    task = str(info["task_type"]).lower()
    if task == "binary":
        task = "binclass"
    if task not in ("binclass", "multiclass", "regression"):
        raise ValueError(f"Unsupported task {task!r} for {name}")
    indices = {}
    for offset, part in enumerate(PARTS):
        y_path = directory / f"y_{part}.npy"
        length = len(np.load(y_path, mmap_mode="r", allow_pickle=True))
        limit = cfg["max_train_rows"] if part == "train" else cfg["max_eval_rows"]
        indices[part] = _indices(length, int(limit), int(cfg["sample_seed"]) + offset)

    def arrays(stem: str, allow_pickle: bool = False) -> dict[str, np.ndarray] | None:
        if not (directory / f"{stem}_train.npy").exists():
            return None
        return {
            part: np.asarray(
                np.load(directory / f"{stem}_{part}.npy", allow_pickle=allow_pickle)[indices[part]]
            )
            for part in PARTS
        }

    x_num = arrays("N")
    x_cat = arrays("C", allow_pickle=True)
    y_raw = arrays("y", allow_pickle=True)
    assert y_raw is not None
    if task == "regression":
        y = {part: values.astype(np.float32) for part, values in y_raw.items()}
        n_classes = 1
    else:
        classes = sorted(set(y_raw["train"].tolist()), key=str)
        lookup = {value: index for index, value in enumerate(classes)}
        y = {
            part: np.asarray([lookup[value] for value in values.tolist()], dtype=np.int64)
            for part, values in y_raw.items()
        }
        n_classes = len(classes)
    digest = hashlib.sha256()
    digest.update(str(directory.resolve()).encode())
    for part in PARTS:
        digest.update(indices[part].tobytes())
    return Dataset(
        name=name,
        task=task,
        x_num=x_num,
        x_bin=None,
        x_cat=x_cat,
        y=y,
        n_classes=n_classes,
        split_fingerprint=digest.hexdigest()[:16],
    )


def load_broad_dataset(name: str) -> Dataset:
    cfg = config()
    if name not in cfg["datasets"]:
        raise KeyError(f"Dataset {name!r} is absent from the frozen benchmark")
    if (DATA_ROOT / name).is_dir():
        from .core import load_dataset

        data = cfg["data"]
        return load_dataset(
            name,
            max_train_rows=int(data["max_train_rows"]),
            max_eval_rows=int(data["max_eval_rows"]),
            sample_seed=int(data["sample_seed"]),
        )
    return _load_finance(name)


def _categorical_blocks(
    dataset: Dataset,
    family: str,
) -> tuple[list[dict[str, np.ndarray]], list[dict[str, object]]]:
    blocks: list[dict[str, np.ndarray]] = []
    metadata: list[dict[str, object]] = []
    if dataset.x_cat is None:
        return blocks, metadata
    maximum = int(config()["data"]["max_category_cardinality"])
    for column in range(dataset.x_cat["train"].shape[1]):
        codes, levels = category_codes(dataset.x_cat, column)
        k = len(levels)
        if k < 2:
            metadata.append({"column": column, "levels": k, "status": "constant_dropped"})
            continue
        if k > maximum:
            metadata.append({"column": column, "levels": k, "status": "high_cardinality_dropped"})
            continue
        if family == "helmert":
            basis = helmert(k)
        elif family == "adjacent":
            basis = np.zeros((k, k - 1), dtype=np.float64)
            for j in range(k - 1):
                basis[j, j] = -1.0
                basis[j + 1, j] = 1.0
        else:
            raise ValueError(family)
        blocks.append(contrast_block(codes, k, basis))
        metadata.append({"column": column, "levels": k, "status": family})
    return blocks, metadata


def _binary_blocks(dataset: Dataset) -> list[dict[str, np.ndarray]]:
    if dataset.x_bin is None:
        return []
    return [standardize(clean_numeric(dataset.x_bin))]


def _numeric_bins(dataset: Dataset) -> int:
    if dataset.x_num is None or dataset.x_num["train"].shape[1] == 0:
        return 0
    data = config()["data"]
    n_features = dataset.x_num["train"].shape[1]
    return max(
        int(data["min_bins"]),
        min(int(data["max_bins"]), int(data["max_numeric_representation_width"]) // n_features),
    )


def _numeric_ple_blocks(
    dataset: Dataset,
    family: str,
    *,
    whiten_each: bool,
) -> tuple[list[dict[str, np.ndarray]], int]:
    if dataset.x_num is None:
        return [], 0
    bins = _numeric_bins(dataset)
    raw_blocks, _ = ple_blocks(dataset.x_num, bins=bins)
    output = []
    for raw in raw_blocks:
        block = raw
        if family == "local":
            width = raw["train"].shape[1]
            transform = np.eye(width, dtype=np.float64)
            for j in range(1, width):
                transform[j - 1, j] = -1.0
            block = apply_transform(raw, transform)
        elif family != "cumulative":
            raise ValueError(family)
        if whiten_each:
            block, _ = whiten(block)
        output.append(block)
    return output, bins


def _combine_nonempty(blocks: list[dict[str, np.ndarray]], rows: dict[str, int]) -> dict[str, np.ndarray]:
    if not blocks:
        return {part: np.empty((rows[part], 0), dtype=np.float64) for part in PARTS}
    return {
        part: np.ascontiguousarray(np.column_stack([block[part] for block in blocks]), dtype=np.float64)
        for part in PARTS
    }


def _natural_blocks(
    dataset: Dataset, family: str
) -> tuple[list[dict[str, np.ndarray]], int, list[dict[str, object]]]:
    if family not in ("cumulative_helmert", "local_adjacent"):
        raise ValueError(family)
    numeric_family = "cumulative" if family == "cumulative_helmert" else "local"
    categorical_family = "helmert" if family == "cumulative_helmert" else "adjacent"
    # Whiten each semantic block separately. This preserves the natural block
    # partition while making the numerical equivalence check stable even for
    # wide datasets with nearly repeated spline columns. The two families can
    # still differ by a within-block orthogonal basis.
    numeric, bins = _numeric_ple_blocks(dataset, numeric_family, whiten_each=True)
    categorical, categorical_meta = _categorical_blocks(dataset, categorical_family)
    categorical = [whiten(block)[0] for block in categorical]
    return numeric + _binary_blocks(dataset) + categorical, bins, categorical_meta


def natural_representation(dataset: Dataset, family: str) -> Representation:
    blocks, bins, categorical_meta = _natural_blocks(dataset, family)
    rows = {part: len(dataset.y[part]) for part in PARTS}
    values = _combine_nonempty(blocks, rows)
    return Representation(
        name=family,
        parts=values,
        metadata={
            "equivalence_class": "natural_exact_spline_contrast",
            "numeric_bins": bins,
            "categorical": categorical_meta,
            "input_features": values["train"].shape[1],
        },
    )


def paired_natural_representations(
    dataset: Dataset,
) -> tuple[Representation, Representation, np.ndarray]:
    """Return the natural exact pair and its known blockwise linear map.

    A single global least-squares map can mix unrelated semantic blocks when
    they are collinear on the training rows.  This construction instead fits
    the square map within every retained semantic block, verifies it on every
    split, and then joins the maps block-diagonally.
    """

    reference_blocks, reference_bins, reference_categorical = _natural_blocks(
        dataset, "cumulative_helmert"
    )
    changed_blocks, changed_bins, changed_categorical = _natural_blocks(
        dataset, "local_adjacent"
    )
    if len(reference_blocks) != len(changed_blocks):
        raise AssertionError("Natural representations have different semantic block counts")
    transforms: list[np.ndarray] = []
    block_errors: list[dict[str, float]] = []
    for reference, changed in zip(reference_blocks, changed_blocks):
        if reference["train"].shape[1] != changed["train"].shape[1]:
            raise AssertionError("Natural pair retained different block ranks")
        transform = np.linalg.lstsq(reference["train"], changed["train"], rcond=1e-10)[0]
        if np.linalg.matrix_rank(transform, tol=1e-10) != transform.shape[0]:
            raise np.linalg.LinAlgError("Natural block map is not invertible")
        errors = {
            part: float(
                np.linalg.norm(reference[part] @ transform - changed[part])
                / max(np.linalg.norm(changed[part]), 1e-30)
            )
            for part in PARTS
        }
        transforms.append(transform)
        block_errors.append(errors)
    rows = {part: len(dataset.y[part]) for part in PARTS}
    reference_values = _combine_nonempty(reference_blocks, rows)
    changed_values = _combine_nonempty(changed_blocks, rows)
    transform = block_diag(*transforms) if transforms else np.empty((0, 0), dtype=np.float64)
    global_errors = {
        part: float(
            np.linalg.norm(reference_values[part] @ transform - changed_values[part])
            / max(np.linalg.norm(changed_values[part]), 1e-30)
        )
        for part in PARTS
    }
    metadata = {
        "equivalence_class": "natural_exact_spline_contrast",
        "reference_family": "cumulative_helmert",
        "changed_family": "local_adjacent",
        "reference_bins": reference_bins,
        "changed_bins": changed_bins,
        "reference_categorical": reference_categorical,
        "changed_categorical": changed_categorical,
        "block_relation_errors": block_errors,
        "basis_relation_errors": global_errors,
        "basis_condition": float(np.linalg.cond(transform)) if transform.size else 1.0,
    }
    return (
        Representation(
            name="natural_cumulative_helmert",
            parts=reference_values,
            metadata=metadata,
        ),
        Representation(
            name="natural_local_adjacent",
            parts=changed_values,
            metadata=metadata,
            reference=reference_values,
            basis_transform=transform,
        ),
        transform,
    )


def natural_blockwise_equivalence_errors(dataset: Dataset) -> dict[str, dict[str, float]]:
    """Verify the natural pair with the known semantic block correspondence.

    A global least-squares map is not a valid verifier when distinct semantic
    blocks happen to be collinear on training rows: its non-unique solution can
    mix blocks and then fail out of sample. Blockwise maps remain unique on the
    retained rank and directly verify the construction actually used.
    """

    source, _, _ = _natural_blocks(dataset, "cumulative_helmert")
    target, _, _ = _natural_blocks(dataset, "local_adjacent")
    if len(source) != len(target):
        raise AssertionError("Natural representations have different semantic block counts")

    def direction(left, right):
        predictions = {part: [] for part in PARTS}
        actual = {part: [] for part in PARTS}
        for a, b in zip(left, right):
            design = np.column_stack((np.ones(len(a["train"])), a["train"]))
            coefficients = np.linalg.lstsq(design, b["train"], rcond=1e-10)[0]
            for part in PARTS:
                predictions[part].append(
                    np.column_stack((np.ones(len(a[part])), a[part])) @ coefficients
                )
                actual[part].append(b[part])
        errors = {}
        for part in PARTS:
            predicted = np.column_stack(predictions[part])
            expected = np.column_stack(actual[part])
            errors[part] = float(
                np.linalg.norm(predicted - expected) / max(np.linalg.norm(expected), 1e-30)
            )
        return errors

    return {"cumulative_to_local": direction(source, target), "local_to_cumulative": direction(target, source)}


def standard_representation(dataset: Dataset, family: str) -> Representation:
    blocks: list[dict[str, np.ndarray]] = []
    if dataset.x_num is not None:
        if family == "raw_standard":
            blocks.append(standardize(clean_numeric(dataset.x_num)))
        elif family == "quantile_standard":
            blocks.append(quantile_numeric(dataset.x_num, seed=0))
        else:
            raise ValueError(family)
    blocks.extend(_binary_blocks(dataset))
    categorical, categorical_meta = _categorical_blocks(dataset, "helmert")
    blocks.extend(categorical)
    rows = {part: len(dataset.y[part]) for part in PARTS}
    values = _combine_nonempty(blocks, rows)
    return Representation(
        name=family,
        parts=values,
        metadata={
            "equivalence_class": "non_equivalent_preprocessing_control",
            "categorical": categorical_meta,
            "input_features": values["train"].shape[1],
        },
    )


def controlled_representation(dataset: Dataset, kappa: float, seed: int = 91000) -> Representation:
    numeric, bins = _numeric_ple_blocks(dataset, "cumulative", whiten_each=True)
    categorical, categorical_meta = _categorical_blocks(dataset, "helmert")
    blocks = numeric + _binary_blocks(dataset) + categorical
    rows = {part: len(dataset.y[part]) for part in PARTS}
    raw = _combine_nonempty(blocks, rows)
    reference, white_meta = whiten(raw)
    dimension = reference["train"].shape[1]
    transform = condition_transform(dimension, kappa, seed)
    covariance = reference["train"].T @ reference["train"] / len(reference["train"])
    original_energy = float(np.trace(covariance))
    transformed_energy = float(np.trace(transform.T @ covariance @ transform))
    transform *= math.sqrt(original_energy / max(transformed_energy, 1e-30))
    transformed = apply_transform(reference, transform)
    relation_errors = {
        part: float(
            np.linalg.norm(reference[part] @ transform - transformed[part])
            / max(np.linalg.norm(transformed[part]), 1e-30)
        )
        for part in PARTS
    }
    return Representation(
        name=f"controlled_kappa_{kappa:g}",
        parts=transformed,
        reference=reference,
        basis_transform=transform,
        metadata={
            "equivalence_class": "exact_invertible_control",
            "target_kappa": float(kappa),
            "numeric_bins": bins,
            "retained_rank": white_meta["retained_rank"],
            "categorical": categorical_meta,
            "input_features": dimension,
            "reference_trace": original_energy,
            "transformed_trace": float(
                np.trace(transformed["train"].T @ transformed["train"] / len(transformed["train"]))
            ),
            "basis_condition": float(np.linalg.cond(transform)),
            "basis_relation_errors": relation_errors,
        },
    )


def affine_reconstruction_errors(
    source: dict[str, np.ndarray], target: dict[str, np.ndarray]
) -> dict[str, float]:
    design = np.column_stack((np.ones(len(source["train"])), source["train"]))
    coefficients = np.linalg.lstsq(design, target["train"], rcond=1e-10)[0]
    errors = {}
    for part in PARTS:
        prediction = np.column_stack((np.ones(len(source[part])), source[part])) @ coefficients
        errors[part] = float(
            np.linalg.norm(prediction - target[part]) / max(np.linalg.norm(target[part]), 1e-30)
        )
    return errors


def sketched_anchor_canonicalize(
    parts: dict[str, np.ndarray],
    *,
    initial_rows: int = 512,
    maximum_rows: int = 8192,
    rtol: float = 1e-10,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Invariant anchor coordinates using a deterministic training-row sketch.

    The sketch is progressively enlarged until its anchors reconstruct the full
    training matrix. Selection depends on the sketch's column space, which is
    unchanged by an invertible right basis transform. This avoids an SVD over
    all training rows in the common case while preserving exactness.
    """

    train = np.asarray(parts["train"], dtype=np.float64)
    mean = train.mean(axis=0)
    centered = train - mean
    n_rows = len(train)
    size = min(initial_rows, n_rows)
    attempts = []
    while True:
        indices = np.unique(np.linspace(0, n_rows - 1, size, dtype=np.int64))
        sample = centered[indices]
        u, singular, _ = np.linalg.svd(sample, full_matrices=False)
        keep = singular > max(float(singular[0]), 1e-30) * rtol
        q = u[:, keep]
        _, _, pivots = qr(q.T, pivoting=True, mode="economic")
        anchors = indices[np.asarray(pivots[: q.shape[1]], dtype=np.int64)]
        anchor_rows = centered[anchors]
        coefficients = np.linalg.lstsq(anchor_rows.T, centered.T, rcond=rtol)[0].T
        error = float(
            np.linalg.norm(coefficients @ anchor_rows - centered)
            / max(np.linalg.norm(centered), 1e-30)
        )
        attempts.append({"sketch_rows": int(len(indices)), "rank": int(q.shape[1]), "error": error})
        if error <= 1e-8 or size >= min(maximum_rows, n_rows):
            break
        size = min(size * 2, maximum_rows, n_rows)
    if error > 1e-8:
        raise np.linalg.LinAlgError(
            f"sketched anchors failed to span training data: relative error={error:.3e}"
        )
    output = {}
    errors = {}
    for part, values in parts.items():
        centered_part = np.asarray(values, dtype=np.float64) - mean
        coefficients = np.linalg.lstsq(anchor_rows.T, centered_part.T, rcond=rtol)[0].T
        output[part] = coefficients
        errors[part] = float(
            np.linalg.norm(coefficients @ anchor_rows - centered_part)
            / max(np.linalg.norm(centered_part), 1e-30)
        )
    output = diagonal_standardize(output)
    return output, {
        "canonical_rank": int(len(anchors)),
        "anchor_rows": anchors.tolist(),
        "sketch_attempts": attempts,
        "reconstruction_errors": errors,
    }
