"""Pilot support-geometry numerical embeddings.

For each scalar numerical field, choose a small ordered support containing both
quantile knots and distributional count spikes.  A weighted path Dirichlet
form on that support yields generalized Laplacian eigenfunctions.  The pilot
separates three questions:

* do support-aware knots beat ordinary quantile PLE knots?
* does the spectral basis help beyond the knot choice?
* does heat attenuation (a smooth function prior) help beyond an orthonormal
  spectral basis?

All transforms use training covariates only.  This is a falsification pilot,
not a final implementation of the proposed learnable field layer.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from scipy.ndimage import median_filter


HERE = Path(__file__).resolve().parent
DAY3 = HERE.parent / "road-to-iclr-day-03"
sys.path.insert(0, str(DAY3))

from experiments.day3.core import (  # noqa: E402
    Dataset,
    PARTS,
    base_schema,
    clean_numeric,
    combine,
    diagonal_standardize,
    load_dataset,
    make_prepared,
    train_model,
    whiten,
)


TABRED_LEGACY_ROOT = Path(os.environ.get(
    "DAY4_TABRED_LEGACY_ROOT",
    "/home/byunhanjoon/2027ICLR/projects/multifeature_ple_tabular/data/tabred_legacy",
))


def load_tabred(
    name: str, max_train_rows: int, max_eval_rows: int, seed: int
) -> Dataset:
    """Load an official temporal TabReD split from the local legacy arrays."""
    directory = TABRED_LEGACY_ROOT / name
    info = json.loads((directory / "info.json").read_text())
    limits = {"train": max_train_rows, "val": max_eval_rows, "test": max_eval_rows}
    indices = {}
    for offset, part in enumerate(PARTS):
        rows = len(np.load(directory / f"Y_{part}.npy", mmap_mode="r"))
        if rows > limits[part]:
            indices[part] = np.sort(
                np.random.default_rng(seed + offset).choice(
                    rows, limits[part], replace=False
                )
            )
        else:
            indices[part] = np.arange(rows)

    def optional(stem: str) -> dict[str, np.ndarray] | None:
        probe = directory / f"{stem}_train.npy"
        if not probe.exists():
            return None
        return {
            part: np.asarray(
                np.load(directory / f"{stem}_{part}.npy", mmap_mode="r")[indices[part]]
            )
            for part in PARTS
        }

    y = optional("Y")
    assert y is not None
    return Dataset(
        name=f"tabred-{name}",
        task=info["task_type"],
        x_num=optional("X_num"),
        x_bin=optional("X_bin"),
        x_cat=optional("X_cat"),
        y={part: np.asarray(y[part], dtype=np.float32) for part in PARTS},
        n_classes=1,
        split_fingerprint=f"official-temporal-subsample-{seed}",
    )


def parameter_count(
    model_name: str,
    input_size: int,
    output_size: int,
    width: int,
    depth: int,
) -> int:
    """Return the exact trainable parameter count for a pilot backbone."""
    if model_name == "mlp":
        # Input layer, depth - 1 square hidden layers, and output layer.
        return (
            input_size * width
            + width
            + (depth - 1) * (width * width + width)
            + width * output_size
            + output_size
        )
    # Input layer; each residual block has LayerNorm and w->2w->w
    # projections; output has LayerNorm and a final projection.
    return (
        input_size * width
        + width
        + depth * (4 * width * width + 5 * width)
        + 2 * width
        + width * output_size
        + output_size
    )


def parameter_matched_width(
    model_name: str,
    input_size: int,
    output_size: int,
    depth: int,
    target: int,
) -> tuple[int, int]:
    """Choose the integer hidden width closest to a target parameter budget."""
    candidates = range(8, 1025)
    counts = [
        parameter_count(model_name, input_size, output_size, width, depth)
        for width in candidates
    ]
    index = min(range(len(counts)), key=lambda i: abs(counts[i] - target))
    return candidates[index], counts[index]


def quantile_nodes(values: np.ndarray, bins: int) -> np.ndarray:
    return np.unique(np.quantile(values, np.linspace(0.0, 1.0, bins + 1)))


def count_spike_statistics(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate threshold-clearing excess mass on the ordered support.

    Summing every one-count fluctuation creates false atoms for rounded
    continuous fields. Requiring an individual spike to clear the fixed,
    train-size-dependent threshold used by the node allocator prevents that
    accumulation artifact. This heuristic threshold is not a significance test.
    """
    levels, counts = np.unique(values, return_counts=True)
    local = median_filter(counts.astype(np.float64), size=5, mode="nearest")
    excess = np.maximum(counts - np.maximum(local, 1.0), 0.0)
    meaningful = excess >= max(2.0, 5e-4 * len(values))
    return levels, counts, excess, meaningful


def support_nodes(values: np.ndarray, bins: int, spike_fraction: float) -> np.ndarray:
    levels, counts, excess, meaningful = count_spike_statistics(values)
    if len(levels) <= bins + 1:
        return levels
    budget = bins + 1
    spike_budget = min(max(int(round(budget * spike_fraction)), 1), budget - 2)
    order = np.lexsort((levels, -excess))
    spike_indices = [index for index in order if meaningful[index]][:spike_budget]
    spikes = levels[spike_indices]
    remaining = max(budget - len(np.unique(spikes)), 2)
    quantiles = np.unique(
        np.quantile(values, np.linspace(0.0, 1.0, remaining))
    )
    nodes = np.unique(np.concatenate((spikes, quantiles, levels[[0, -1]])))
    if len(nodes) > budget:
        # Preserve endpoints and the strongest spikes; remove the most
        # redundant remaining points by nearest-neighbor gap.
        protected = set(spikes.tolist()) | {float(levels[0]), float(levels[-1])}
        while len(nodes) > budget:
            gaps = np.minimum(
                np.r_[np.inf, np.diff(nodes)], np.r_[np.diff(nodes), np.inf]
            )
            removable = np.array([float(x) not in protected for x in nodes])
            candidates = np.flatnonzero(removable)
            if not len(candidates):
                break
            nodes = np.delete(nodes, candidates[np.argmin(gaps[candidates])])
    return nodes


def linear_basis(query: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """Cumulative PLE coordinates induced by fixed ordered nodes."""
    if len(nodes) < 2:
        return np.empty((len(query), 0), dtype=np.float64)
    left, right = nodes[:-1], nodes[1:]
    width = np.maximum(right - left, 1e-12)
    return np.clip((query[:, None] - left) / width, 0.0, 1.0)


def hat_basis(query: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """Nodal piecewise-linear coordinates, including the constant mode."""
    output = np.zeros((len(query), len(nodes)), dtype=np.float64)
    right = np.searchsorted(nodes, query, side="right")
    below = right == 0
    above = right == len(nodes)
    output[below, 0] = 1.0
    output[above, -1] = 1.0
    interior = ~(below | above)
    rows = np.flatnonzero(interior)
    right_i = right[interior]
    left_i = right_i - 1
    gap = np.maximum(nodes[right_i] - nodes[left_i], 1e-12)
    weight_right = (query[interior] - nodes[left_i]) / gap
    output[rows, left_i] = 1.0 - weight_right
    output[rows, right_i] = weight_right
    return output


def riesz_basis(
    clean: dict[str, np.ndarray],
    column: int,
    nodes: np.ndarray,
    strength: float,
    *,
    permuted: bool = False,
    topology: str = "path",
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Coordinates whose Euclidean weight norm is the M + tau S norm.

    M is the empirical mass form of the nodal finite-element space and S is
    its gap-weighted path or ring stiffness.  The constant mode is removed by
    centering.  A node permutation supplies a same-space wrong-geometry
    control without changing input dimension or parameter count.
    """
    if len(nodes) < 2:
        return (
            {part: np.empty((len(clean[part]), 0), dtype=np.float64) for part in PARTS},
            {
                "riesz_rank": 0,
                "riesz_eigenvalue_max": 0.0,
                "riesz_attenuation_min": 1.0,
            },
        )
    blocks = {
        part: hat_basis(clean[part][:, column], nodes) for part in PARTS
    }
    mean = blocks["train"].mean(axis=0)
    centered = blocks["train"] - mean
    mass = centered.T @ centered / max(len(centered), 1)
    mass_eigenvalues, mass_vectors = np.linalg.eigh(mass)
    threshold = max(float(mass_eigenvalues[-1]), 0.0) * 1e-10
    keep = mass_eigenvalues > threshold
    whitener = mass_vectors[:, keep] / np.sqrt(mass_eigenvalues[keep])[None, :]

    gaps = np.maximum(np.diff(nodes), 1e-12)
    stiffness = np.zeros((len(nodes), len(nodes)), dtype=np.float64)
    for edge, conductance in enumerate(1.0 / gaps):
        stiffness[edge, edge] += conductance
        stiffness[edge + 1, edge + 1] += conductance
        stiffness[edge, edge + 1] -= conductance
        stiffness[edge + 1, edge] -= conductance
    if topology == "ring":
        # Cyclic states have no numerical wrap-around gap in their scalar
        # labels.  Use the median interior conductance for the declared closing
        # edge, making the construction invariant to a uniform change of unit.
        conductance = float(np.median(1.0 / gaps))
        stiffness[0, 0] += conductance
        stiffness[-1, -1] += conductance
        stiffness[0, -1] -= conductance
        stiffness[-1, 0] -= conductance
    elif topology != "path":
        raise ValueError(f"unknown field topology: {topology}")
    if permuted:
        permutation = np.random.default_rng(781_031 + column).permutation(len(nodes))
        stiffness = stiffness[np.ix_(permutation, permutation)]

    canonical_stiffness = whitener.T @ stiffness @ whitener
    eigenvalues, rotation = eigh(canonical_stiffness)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    positive = eigenvalues[eigenvalues > 1e-12]
    scale = float(np.median(positive)) if len(positive) else 1.0
    normalized = eigenvalues / max(scale, 1e-12)
    attenuation = 1.0 / np.sqrt(1.0 + strength * normalized)
    transform = (whitener @ rotation) * attenuation[None, :]
    output = {
        part: (blocks[part] - mean) @ transform for part in PARTS
    }
    return output, {
        "riesz_rank": int(keep.sum()),
        "riesz_eigenvalue_max": float(normalized.max(initial=0.0)),
        "riesz_attenuation_min": float(attenuation.min(initial=1.0)),
    }


def interpolate_node_values(
    query: np.ndarray, nodes: np.ndarray, node_values: np.ndarray
) -> np.ndarray:
    output = np.empty((len(query), node_values.shape[1]), dtype=np.float64)
    for column in range(node_values.shape[1]):
        output[:, column] = np.interp(
            query, nodes, node_values[:, column], left=node_values[0, column], right=node_values[-1, column]
        )
    return output


def mass_power_basis(
    blocks: dict[str, np.ndarray], power: float
) -> tuple[dict[str, np.ndarray], int]:
    """Center a field block and apply the fractional mass map M^(-power/2)."""
    if not blocks["train"].shape[1]:
        return blocks, 0
    mean = blocks["train"].mean(axis=0)
    centered = blocks["train"] - mean
    covariance = centered.T @ centered / max(len(centered), 1)
    eigenvalues, vectors = np.linalg.eigh(covariance)
    threshold = max(float(eigenvalues[-1]), 0.0) * 1e-10
    keep = eigenvalues > threshold
    transform = vectors[:, keep] * eigenvalues[keep][None, :] ** (-0.5 * power)
    return {
        part: (blocks[part] - mean) @ transform for part in PARTS
    }, int(keep.sum())


def spectral_node_values(
    train: np.ndarray, nodes: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Generalized eigenfunctions of a gap-weighted path Dirichlet form."""
    if len(nodes) < 2:
        return np.empty((len(nodes), 0)), np.empty(0), {}
    gaps = np.maximum(np.diff(nodes), 1e-12)
    positive = gaps[gaps > 0]
    gap_scale = float(np.median(positive)) if len(positive) else 1.0
    relative_gap = gaps / max(gap_scale, 1e-12)
    conductance = 1.0 / np.maximum(relative_gap, 1e-3)
    laplacian = np.zeros((len(nodes), len(nodes)), dtype=np.float64)
    for edge, weight in enumerate(conductance):
        laplacian[edge, edge] += weight
        laplacian[edge + 1, edge + 1] += weight
        laplacian[edge, edge + 1] -= weight
        laplacian[edge + 1, edge] -= weight

    nearest = np.searchsorted(nodes, train)
    nearest = np.clip(nearest, 0, len(nodes) - 1)
    left = np.maximum(nearest - 1, 0)
    choose_left = np.abs(train - nodes[left]) < np.abs(train - nodes[nearest])
    nearest[choose_left] = left[choose_left]
    counts = np.bincount(nearest, minlength=len(nodes)).astype(np.float64)
    mass = (counts + 0.5) / (counts.sum() + 0.5 * len(nodes))
    eigenvalues, eigenvectors = eigh(laplacian, np.diag(mass))
    # Drop the constant mode.  Signs are fixed only for deterministic files;
    # downstream predictions are sign-invariant up to first-layer transport.
    eigenvalues = np.maximum(eigenvalues[1:], 0.0)
    eigenvectors = eigenvectors[:, 1:]
    for column in range(eigenvectors.shape[1]):
        pivot = np.argmax(np.abs(eigenvectors[:, column]))
        if eigenvectors[pivot, column] < 0:
            eigenvectors[:, column] *= -1
    return eigenvectors, eigenvalues, {
        "median_gap": gap_scale,
        "maximum_relative_gap": float(relative_gap.max(initial=0.0)),
        "minimum_node_mass": float(mass.min(initial=0.0)),
        "maximum_node_mass": float(mass.max(initial=0.0)),
    }


def numeric_representations(
    parts: dict[str, np.ndarray],
    *,
    bins: int,
    spike_fraction: float,
    heat_strength: float,
    minimum_excess_mass: float = 0.02,
    enabled_columns: set[int] | None = None,
) -> tuple[dict[str, dict[str, np.ndarray]], list[dict[str, object]]]:
    clean = clean_numeric(parts)
    variants = {
        name: {part: [] for part in PARTS}
        for name in (
            "quantile_ple",
            "quantile_standardized",
            "quantile_whitened",
            "support_ple",
            "support_standardized",
            "support_whitened",
            "support_spectral",
            "support_heat",
            "adaptive_support_ple",
            "adaptive_support_standardized",
            "adaptive_support_whitened",
            "adaptive_support_mass25",
            "adaptive_support_mass50",
            "adaptive_support_mass75",
            "adaptive_support_riesz",
            "adaptive_support_wrong_riesz",
            "adaptive_support_spectral",
            "adaptive_support_heat",
        )
    }
    metadata: list[dict[str, object]] = []
    for column in range(clean["train"].shape[1]):
        train = clean["train"][:, column]
        _, _, excess, meaningful = count_spike_statistics(train)
        local_excess_mass = float(excess[meaningful].sum() / len(train))
        adaptive = local_excess_mass >= minimum_excess_mass and (
            enabled_columns is None or column in enabled_columns
        )
        q_nodes = quantile_nodes(train, bins)
        s_nodes = support_nodes(train, bins, spike_fraction)
        node_vectors, eigenvalues, spectral_meta = spectral_node_values(train, s_nodes)
        rank = len(s_nodes) - 1
        scale = float(np.median(eigenvalues[eigenvalues > 1e-12])) if np.any(eigenvalues > 1e-12) else 1.0
        attenuation = np.exp(-0.5 * heat_strength * eigenvalues / max(scale, 1e-12))
        train_spectral = interpolate_node_values(train, s_nodes, node_vectors)
        mean = train_spectral.mean(axis=0)
        std = train_spectral.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        quantile_blocks = {
            part: linear_basis(clean[part][:, column], q_nodes) for part in PARTS
        }
        support_blocks = {
            part: linear_basis(clean[part][:, column], s_nodes) for part in PARTS
        }
        quantile_standardized = diagonal_standardize(quantile_blocks)
        support_standardized = diagonal_standardize(support_blocks)
        if quantile_blocks["train"].shape[1]:
            quantile_whitened, quantile_white_meta = whiten(quantile_blocks)
        else:
            quantile_whitened = quantile_blocks
            quantile_white_meta = {"retained_rank": 0}
        if support_blocks["train"].shape[1]:
            support_whitened, support_white_meta = whiten(support_blocks)
        else:
            support_whitened = support_blocks
            support_white_meta = {"retained_rank": 0}
        support_mass25, _ = mass_power_basis(support_blocks, 0.25)
        support_mass50, _ = mass_power_basis(support_blocks, 0.50)
        support_mass75, _ = mass_power_basis(support_blocks, 0.75)
        support_riesz, riesz_meta = riesz_basis(
            clean, column, s_nodes, heat_strength
        )
        support_wrong_riesz, _ = riesz_basis(
            clean, column, s_nodes, heat_strength, permuted=True
        )
        for part in PARTS:
            x = clean[part][:, column]
            quantile = quantile_blocks[part]
            support = support_blocks[part]
            spectral = (interpolate_node_values(x, s_nodes, node_vectors) - mean) / std
            variants["quantile_ple"][part].append(quantile)
            variants["quantile_standardized"][part].append(
                quantile_standardized[part]
            )
            variants["quantile_whitened"][part].append(quantile_whitened[part])
            variants["support_ple"][part].append(support)
            variants["support_standardized"][part].append(
                support_standardized[part]
            )
            variants["support_whitened"][part].append(support_whitened[part])
            variants["support_spectral"][part].append(spectral)
            variants["support_heat"][part].append(spectral * attenuation)
            variants["adaptive_support_ple"][part].append(
                support if adaptive else quantile
            )
            variants["adaptive_support_standardized"][part].append(
                support_standardized[part] if adaptive else quantile
            )
            variants["adaptive_support_whitened"][part].append(
                support_whitened[part] if adaptive else quantile
            )
            variants["adaptive_support_mass25"][part].append(
                support_mass25[part] if adaptive else quantile
            )
            variants["adaptive_support_mass50"][part].append(
                support_mass50[part] if adaptive else quantile
            )
            variants["adaptive_support_mass75"][part].append(
                support_mass75[part] if adaptive else quantile
            )
            variants["adaptive_support_riesz"][part].append(
                support_riesz[part] if adaptive else quantile
            )
            variants["adaptive_support_wrong_riesz"][part].append(
                support_wrong_riesz[part] if adaptive else quantile
            )
            variants["adaptive_support_spectral"][part].append(
                spectral if adaptive else quantile
            )
            variants["adaptive_support_heat"][part].append(
                spectral * attenuation if adaptive else quantile
            )
        metadata.append(
            {
                "column": column,
                "cardinality": int(len(np.unique(train))),
                "quantile_nodes": int(len(q_nodes)),
                "support_nodes": int(len(s_nodes)),
                "rank": rank,
                "quantile_whitened_rank": quantile_white_meta["retained_rank"],
                "support_whitened_rank": support_white_meta["retained_rank"],
                "local_excess_mass": local_excess_mass,
                "adaptive": adaptive,
                "heat_attenuation_min": float(attenuation.min(initial=1.0)),
                **spectral_meta,
                **riesz_meta,
            }
        )
    output: dict[str, dict[str, np.ndarray]] = {}
    for name, by_part in variants.items():
        output[name] = {
            part: np.ascontiguousarray(np.column_stack(by_part[part]), dtype=np.float32)
            for part in PARTS
        }
    return output, metadata


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        fields.extend(field for field in row if field not in fields)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["adult", "diamond", "california", "churn"])
    parser.add_argument("--models", nargs="+", default=["mlp", "resnet"])
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=[
            "quantile_ple",
            "quantile_standardized",
            "quantile_whitened",
            "support_ple",
            "support_standardized",
            "support_whitened",
            "support_spectral",
            "support_heat",
            "adaptive_support_ple",
            "adaptive_support_standardized",
            "adaptive_support_whitened",
            "adaptive_support_mass25",
            "adaptive_support_mass50",
            "adaptive_support_mass75",
            "adaptive_support_riesz",
            "adaptive_support_wrong_riesz",
            "adaptive_support_spectral",
            "adaptive_support_heat",
        ],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260826])
    parser.add_argument("--bins", type=int, default=32)
    parser.add_argument("--spike-fraction", type=float, default=0.35)
    parser.add_argument("--heat-strength", type=float, default=1.0)
    parser.add_argument("--minimum-excess-mass", type=float, default=0.02)
    parser.add_argument(
        "--enabled-columns",
        nargs="+",
        type=int,
        help="Optional field-ablation mask applied after the support diagnostic.",
    )
    parser.add_argument("--max-train-rows", type=int, default=50000)
    parser.add_argument("--max-eval-rows", type=int, default=15000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument(
        "--match-parameters",
        action="store_true",
        help="Match every representation to the quantile-PLE model's parameter count.",
    )
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--output", type=Path, default=HERE / "results/support_heat_pilot.csv")
    args = parser.parse_args()

    rows: list[dict[str, object]] = list(read_rows(args.output))
    completed = {
        (str(row["dataset"]), str(row["model"]), int(row["seed"]), str(row["method"]))
        for row in rows
    }
    all_metadata: dict[str, object] = {}
    for name in args.datasets:
        if name == "tabred-weather":
            from innovation_pilot import load_weather

            dataset = load_weather(
                args.max_train_rows, args.max_eval_rows, 20260826
            )
        elif name.startswith("tabred-"):
            dataset = load_tabred(
                name.removeprefix("tabred-"),
                args.max_train_rows,
                args.max_eval_rows,
                20260826,
            )
        else:
            dataset = load_dataset(
                name,
                max_train_rows=args.max_train_rows,
                max_eval_rows=args.max_eval_rows,
                sample_seed=20260826,
            )
        if dataset.x_num is None:
            continue
        numeric, metadata = numeric_representations(
            dataset.x_num,
            bins=args.bins,
            spike_fraction=args.spike_fraction,
            heat_strength=args.heat_strength,
            minimum_excess_mass=args.minimum_excess_mass,
            enabled_columns=(
                set(args.enabled_columns) if args.enabled_columns is not None else None
            ),
        )
        nonnumeric = base_schema(dataset, seed=20260826, include_num=False)
        baseline_features = combine([numeric["quantile_ple"], nonnumeric])
        variants = {
            method: combine([features, nonnumeric])
            for method, features in numeric.items()
            if args.methods is None or method in args.methods
        }
        all_metadata[name] = metadata
        for model in args.models:
            output_size = dataset.n_classes if dataset.task == "multiclass" else 1
            baseline_parameters = parameter_count(
                model,
                baseline_features["train"].shape[1],
                output_size,
                args.width,
                args.depth,
            )
            for seed in args.seeds:
                for method, features in variants.items():
                    key = (name, model, seed, method)
                    if key in completed:
                        continue
                    train_width = args.width
                    train_parameters = parameter_count(
                        model,
                        features["train"].shape[1],
                        output_size,
                        train_width,
                        args.depth,
                    )
                    if args.match_parameters:
                        train_width, train_parameters = parameter_matched_width(
                            model,
                            features["train"].shape[1],
                            output_size,
                            args.depth,
                            baseline_parameters,
                        )
                    result, _ = train_model(
                        make_prepared(dataset, features, {"method": method}),
                        seed=seed,
                        device=args.device,
                        model_name=model,
                        width=train_width,
                        depth=args.depth,
                        dropout=0.1,
                        learning_rate=1e-3,
                        weight_decay=1e-4,
                        batch_size=512,
                        max_epochs=args.epochs,
                        patience=args.patience,
                    )
                    row = {
                        "dataset": name,
                        "task": dataset.task,
                        "model": model,
                        "seed": seed,
                        "method": method,
                        "bins": args.bins,
                        "spike_fraction": args.spike_fraction,
                        "heat_strength": args.heat_strength,
                        "input_size": features["train"].shape[1],
                        "width": train_width,
                        "parameter_count": train_parameters,
                        "baseline_parameter_count": baseline_parameters,
                        "parameter_error_fraction": (
                            train_parameters - baseline_parameters
                        )
                        / baseline_parameters,
                        **result,
                    }
                    rows.append(row)
                    completed.add(key)
                    write_rows(args.output, rows)
                    print(json.dumps(row, sort_keys=True), flush=True)
    (args.output.parent / "support_heat_metadata.json").write_text(
        json.dumps(all_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
