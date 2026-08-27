#!/usr/bin/env python3
"""Residual-Riesz surfaces for predeclared two-coordinate field groups."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

from king_county_spatial_pilot import load_king_county
from ft_support_pilot import (
    FieldTokenTransformer,
    parameter_count as ft_parameter_count,
    train_ft,
)
from residual_riesz_pilot import (
    METHODS,
    field_forms,
    fit_shared_raple,
    representer_values,
    standardized_scalar,
)
from support_heat_pilot import (
    HERE,
    PARTS,
    Dataset,
    base_schema,
    clean_numeric,
    combine,
    hat_basis,
    linear_basis,
    load_dataset,
    make_prepared,
    parameter_count,
    parameter_matched_width,
    quantile_nodes,
    read_rows,
    support_nodes,
    train_model,
    write_rows,
)
from tabm_support_pilot import FlatTabM, count_parameters, train_tabm


PRODUCT_METHODS = (
    "raple_raw",
    "anchor_only",
    "anchor_product_mass",
    "anchor_product_riesz",
    "anchor_product_wrong",
    "anchor_product_isospectral",
    "anchor_product_rho_mixture",
    "anchor_product_rho_mixture_wrong",
    "anchor_product_rho_mixture_isospectral",
)


def generalized_median_scale(
    mass: np.ndarray, stiffness: np.ndarray
) -> float:
    """Median positive generalized frequency, restricted to mass support."""
    values, vectors = np.linalg.eigh(mass)
    keep = values > max(float(values[-1]), 0.0) * 1e-10
    if not np.any(keep):
        return 1.0
    whitener = vectors[:, keep] / np.sqrt(values[keep])[None, :]
    spectrum = np.maximum(
        np.linalg.eigvalsh(whitener.T @ stiffness @ whitener), 0.0
    )
    positive = spectrum[spectrum > max(float(spectrum[-1]), 1.0) * 1e-12]
    return float(np.median(positive)) if len(positive) else 1.0


def support_graph_stiffness(
    phi_train: np.ndarray,
    coordinates: np.ndarray,
    neighbors: int,
) -> tuple[np.ndarray, dict[str, float]]:
    """Galerkin stiffness of a train-only geodesic support graph.

    Coordinates are latitude/longitude in degrees.  The graph uses haversine
    distance, and its Dirichlet form is projected into the product basis.
    """
    radians = np.deg2rad(np.asarray(coordinates, dtype=np.float64))
    count = min(max(int(neighbors), 2) + 1, len(radians))
    finder = NearestNeighbors(n_neighbors=count, metric="haversine")
    finder.fit(radians)
    distances, indices = finder.kneighbors(radians)
    distances = distances[:, 1:]
    indices = indices[:, 1:]
    positive = distances[distances > 0]
    bandwidth = float(np.median(positive)) if len(positive) else 1.0
    weights = np.exp(-0.5 * (distances / max(bandwidth, 1e-12)) ** 2)

    stiffness = np.zeros((phi_train.shape[1], phi_train.shape[1]))
    for offset in range(indices.shape[1]):
        difference = phi_train - phi_train[indices[:, offset]]
        stiffness += difference.T @ (weights[:, offset, None] * difference)
    stiffness /= max(float(len(phi_train) * indices.shape[1]), 1.0)
    stiffness = 0.5 * (stiffness + stiffness.T)
    return stiffness, {
        "knn_neighbors": int(indices.shape[1]),
        "knn_bandwidth_radians": bandwidth,
    }


def exact_isospectral_operator(
    mass: np.ndarray,
    stiffness: np.ndarray,
    strength: float,
    seed: int,
) -> np.ndarray:
    values, vectors = np.linalg.eigh(mass)
    keep = values > max(float(values[-1]), 0.0) * 1e-10
    if not np.any(keep):
        return mass.copy()
    whitener = vectors[:, keep] / np.sqrt(values[keep])[None, :]
    spectrum = np.maximum(
        np.linalg.eigvalsh(whitener.T @ stiffness @ whitener), 0.0
    )
    rng = np.random.default_rng(seed)
    rotation, triangular = np.linalg.qr(
        rng.normal(size=(len(spectrum), len(spectrum)))
    )
    rotation *= np.where(np.diag(triangular) < 0, -1.0, 1.0)[None, :]
    whitened = (rotation * spectrum[None, :]) @ rotation.T
    mass_sqrt = vectors[:, keep] * np.sqrt(values[keep])[None, :]
    control = mass_sqrt @ whitened @ mass_sqrt.T
    control = 0.5 * (control + control.T)
    return mass + strength * control


def product_block(
    clean: dict[str, np.ndarray],
    columns: tuple[int, int],
    bins: int,
    interaction_projection: str = "empirical-anova",
) -> tuple[
    dict[str, np.ndarray], tuple[np.ndarray, np.ndarray], np.ndarray
]:
    nodes = tuple(
        support_nodes(clean["train"][:, column], bins, 0.35)
        for column in columns
    )
    marginal: list[dict[str, np.ndarray]] = []
    marginal_means: list[np.ndarray] = []
    for column, field_nodes in zip(columns, nodes):
        hats = {
            part: hat_basis(clean[part][:, column], field_nodes) for part in PARTS
        }
        mean = hats["train"].mean(axis=0)
        marginal_means.append(mean)
        marginal.append({part: values - mean for part, values in hats.items()})
    product = {
        part: np.einsum(
            "ni,nj->nij", marginal[0][part], marginal[1][part], optimize=True
        ).reshape(len(marginal[0][part]), -1)
        for part in PARTS
    }
    grid_marginal = [
        np.eye(len(field_nodes)) - mean[None, :]
        for field_nodes, mean in zip(nodes, marginal_means)
    ]
    reference_product = np.einsum(
        "ni,mj->nmij", grid_marginal[0], grid_marginal[1], optimize=True
    ).reshape(len(nodes[0]) * len(nodes[1]), -1)
    if interaction_projection == "empirical-anova":
        additive = {
            part: np.column_stack(
                [
                    np.ones(len(product[part])),
                    marginal[0][part],
                    marginal[1][part],
                ]
            )
            for part in PARTS
        }
        projection = np.linalg.pinv(additive["train"], rcond=1e-10) @ product[
            "train"
        ]
        product = {
            part: values - additive[part] @ projection
            for part, values in product.items()
        }
        reference_additive = np.column_stack(
            [
                np.ones(len(reference_product)),
                np.repeat(grid_marginal[0], len(nodes[1]), axis=0),
                np.tile(grid_marginal[1], (len(nodes[0]), 1)),
            ]
        )
        reference_product = reference_product - reference_additive @ projection
    elif interaction_projection == "center-only":
        mean = product["train"].mean(axis=0)
        product = {part: values - mean for part, values in product.items()}
        reference_product = reference_product - mean
    else:
        raise ValueError(
            f"unknown interaction projection: {interaction_projection}"
        )
    def trapezoid_weights(field_nodes: np.ndarray) -> np.ndarray:
        gaps = np.diff(field_nodes)
        weights = np.empty(len(field_nodes), dtype=np.float64)
        weights[0] = gaps[0] / 2
        weights[-1] = gaps[-1] / 2
        weights[1:-1] = (gaps[:-1] + gaps[1:]) / 2
        total = float(weights.sum())
        if total <= 0:
            return np.full(len(weights), 1 / len(weights))
        return weights / total

    reference_weights = np.kron(
        trapezoid_weights(nodes[0]), trapezoid_weights(nodes[1])
    )
    reference_mass = reference_product.T @ (
        reference_weights[:, None] * reference_product
    )
    reference_mass = 0.5 * (reference_mass + reference_mass.T)
    return product, nodes, reference_mass


def build_product_features(
    dataset: Dataset,
    columns: tuple[int, int],
    *,
    base_bins: int,
    product_bins: int,
    strength: float,
    seed: int,
    control_seed: int,
    stiffness_family: str = "product",
    normalize_stiffness: bool = True,
    knn_neighbors: int = 16,
    interaction_projection: str = "empirical-anova",
    reference_mass_weight: float = 0.0,
    reference_mass_mixture: tuple[float, ...] = (),
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, object]]:
    assert dataset.x_num is not None
    clean = clean_numeric(dataset.x_num)
    qblocks = []
    for column in range(clean["train"].shape[1]):
        nodes = quantile_nodes(clean["train"][:, column], base_bins)
        qblocks.append(
            {part: linear_basis(clean[part][:, column], nodes) for part in PARTS}
        )
    ple = combine([*qblocks, base_schema(dataset, seed=seed, include_num=False)])
    raple, anchor, residual, folds, raple_metadata = fit_shared_raple(dataset, seed)
    anchor_feature = standardized_scalar(
        {part: values[:, None] for part, values in anchor.items()}
    )
    product, nodes, reference_mass = product_block(
        clean, columns, product_bins, interaction_projection
    )
    phi_train = product["train"]
    empirical_mass = phi_train.T @ phi_train / len(phi_train)
    if not 0.0 <= reference_mass_weight <= 1.0:
        raise ValueError("reference_mass_weight must lie in [0, 1]")
    mass = (
        (1.0 - reference_mass_weight) * empirical_mass
        + reference_mass_weight * reference_mass
    )

    marginal_forms = []
    for column, field_nodes in zip(columns, nodes):
        hats = hat_basis(clean["train"][:, column], field_nodes)
        phi = hats - hats.mean(axis=0)
        m, correct, wrong, _ = field_forms(
            phi, field_nodes, column, 1.0, control_seed
        )
        marginal_forms.append((m, correct - m, wrong - m))
    m1, s1, w1 = marginal_forms[0]
    m2, s2, w2 = marginal_forms[1]
    product_stiffness = np.kron(s1, m2) + np.kron(m1, s2)
    product_wrong = np.kron(w1, m2) + np.kron(m1, w2)
    geometry_metadata: dict[str, object] = {}
    if stiffness_family == "product":
        raw_stiffness = product_stiffness
        raw_wrong_stiffness = product_wrong
    elif stiffness_family == "knn":
        raw_stiffness, graph_metadata = support_graph_stiffness(
            phi_train, clean["train"][:, list(columns)], knn_neighbors
        )
        geometry_metadata.update(graph_metadata)
        permutation = np.random.default_rng(control_seed).permutation(
            raw_stiffness.shape[0]
        )
        raw_wrong_stiffness = raw_stiffness[np.ix_(permutation, permutation)]
    else:
        raise ValueError(f"unknown stiffness family: {stiffness_family}")
    stiffness_scale = generalized_median_scale(mass, raw_stiffness)
    stiffness = raw_stiffness
    wrong_stiffness = raw_wrong_stiffness
    if normalize_stiffness:
        stiffness = stiffness / max(stiffness_scale, 1e-12)
        wrong_stiffness = wrong_stiffness / max(stiffness_scale, 1e-12)
    operators = {
        "mass": mass,
        "riesz": mass + strength * stiffness,
        "wrong": mass + strength * wrong_stiffness,
        "isospectral": exact_isospectral_operator(
            mass, stiffness, strength, control_seed + 2_000_003
        ),
    }
    blocks = {
        kind: standardized_scalar(
            representer_values(product, residual, folds, operator)
        )
        for kind, operator in operators.items()
    }
    variants = {
        "raple_raw": combine([ple, standardized_scalar(raple)]),
        "anchor_only": combine([ple, anchor_feature]),
        "anchor_product_mass": combine([ple, anchor_feature, blocks["mass"]]),
        "anchor_product_riesz": combine([ple, anchor_feature, blocks["riesz"]]),
        "anchor_product_wrong": combine([ple, anchor_feature, blocks["wrong"]]),
        "anchor_product_isospectral": combine(
            [ple, anchor_feature, blocks["isospectral"]]
        ),
    }
    if reference_mass_mixture:
        mixture_kinds = ("riesz", "wrong", "isospectral")
        mixture_values = {kind: [] for kind in mixture_kinds}
        for rho in reference_mass_mixture:
            if not 0.0 < rho <= 1.0:
                raise ValueError("reference_mass_mixture values must lie in (0, 1]")
            rho_mass = (1.0 - rho) * empirical_mass + rho * reference_mass
            rho_scale = generalized_median_scale(rho_mass, raw_stiffness)
            rho_stiffness = raw_stiffness
            rho_wrong = raw_wrong_stiffness
            if normalize_stiffness:
                rho_stiffness = rho_stiffness / max(rho_scale, 1e-12)
                rho_wrong = rho_wrong / max(rho_scale, 1e-12)
            rho_operators = {
                "riesz": rho_mass + strength * rho_stiffness,
                "wrong": rho_mass + strength * rho_wrong,
                "isospectral": exact_isospectral_operator(
                    rho_mass, rho_stiffness, strength,
                    control_seed + 2_000_003,
                ),
            }
            for kind, operator in rho_operators.items():
                mixture_values[kind].append(
                    standardized_scalar(
                        representer_values(product, residual, folds, operator)
                    )
                )
        mixture_blocks = {
            kind: standardized_scalar(
                {
                    part: np.mean(
                        [values[part] for values in kind_values], axis=0
                    )
                    for part in PARTS
                }
            )
            for kind, kind_values in mixture_values.items()
        }
        variants.update(
            {
                "anchor_product_rho_mixture": combine(
                    [ple, anchor_feature, mixture_blocks["riesz"]]
                ),
                "anchor_product_rho_mixture_wrong": combine(
                    [ple, anchor_feature, mixture_blocks["wrong"]]
                ),
                "anchor_product_rho_mixture_isospectral": combine(
                    [ple, anchor_feature, mixture_blocks["isospectral"]]
                ),
            }
        )
    values, vectors = np.linalg.eigh(mass)
    keep = values > max(float(values[-1]), 0.0) * 1e-10
    whitener = vectors[:, keep] / np.sqrt(values[keep])[None, :]
    semantic_spectrum = np.linalg.eigvalsh(whitener.T @ stiffness @ whitener)
    control_spectrum = np.linalg.eigvalsh(
        whitener.T @ (operators["isospectral"] - mass) @ whitener / strength
    )
    def numerical_rank(matrix: np.ndarray) -> int:
        spectrum = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
        threshold = max(float(np.max(np.abs(spectrum), initial=0.0)), 1.0) * 1e-10
        return int(np.count_nonzero(np.abs(spectrum) > threshold))

    metadata = {
        "columns": list(columns),
        "nodes": [len(value) for value in nodes],
        "product_dimension": int(phi_train.shape[1]),
        "product_empirical_mass_rank": numerical_rank(empirical_mass),
        "product_reference_mass_rank": numerical_rank(reference_mass),
        "product_mass_rank": int(np.count_nonzero(keep)),
        "product_mass_nullity": int(phi_train.shape[1] - np.count_nonzero(keep)),
        "reference_mass_weight": reference_mass_weight,
        "reference_mass_mixture": list(reference_mass_mixture),
        "reference_measure": "tensor-trapezoid-on-declared-support-nodes",
        "operator_ranks": {
            kind: numerical_rank(operator) for kind, operator in operators.items()
        },
        "isospectral_scope": (
            "reference-completed-function-space-spectrum"
            if reference_mass_weight > 0
            else "finite-empirical-mass-supported-generalized-spectrum"
        ),
        "isospectral_max_abs_error": float(
            np.max(np.abs(semantic_spectrum - control_spectrum), initial=0.0)
        ),
        "control_seed": control_seed,
        "base_numeric_block_dimensions": [
            int(qblocks[column]["train"].shape[1])
            for column in range(len(qblocks))
        ],
        "base_schema_dimension": int(
            base_schema(dataset, seed=seed, include_num=False)["train"].shape[1]
        ),
        "stiffness_family": stiffness_family,
        "stiffness_normalization": (
            "joint-generalized-median" if normalize_stiffness else "legacy-none"
        ),
        "stiffness_scale": stiffness_scale,
        "interaction_projection": interaction_projection,
        **geometry_metadata,
        **raple_metadata,
    }
    return variants, metadata


def token_dimensions(method: str, metadata: dict[str, object]) -> list[int]:
    dimensions = [int(value) for value in metadata["base_numeric_block_dimensions"]]
    schema_dimension = int(metadata["base_schema_dimension"])
    if schema_dimension:
        dimensions.append(schema_dimension)
    if method == "raple_raw":
        dimensions.append(int(metadata["raple_feature_count"]))
    elif method == "anchor_only":
        dimensions.append(1)
    elif method in PRODUCT_METHODS:
        dimensions.extend([1, 1])
    else:
        raise ValueError(method)
    return [dimension for dimension in dimensions if dimension > 0]


def dataset_and_columns(name: str) -> tuple[Dataset, tuple[int, int], list[str]]:
    if name == "king-county-sales":
        dataset, metadata = load_king_county()
        names = list(metadata["feature_names"])
        return dataset, (names.index("lat"), names.index("long")), names
    if name == "california":
        dataset = load_dataset(name, sample_seed=20260826)
        names = [
            "MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population",
            "AveOccup", "Latitude", "Longitude",
        ]
        return dataset, (6, 7), names
    raise ValueError(f"unsupported spatial dataset: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets", nargs="+", default=["california", "king-county-sales"]
    )
    parser.add_argument("--models", nargs="+", default=["mlp", "resnet", "tabm"])
    parser.add_argument("--methods", nargs="+", choices=PRODUCT_METHODS)
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[20260850, 20260851, 20260852]
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--base-bins", type=int, default=32)
    parser.add_argument("--product-bins", type=int, default=12)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--control-seed", type=int, default=991337)
    parser.add_argument(
        "--stiffness-family", choices=["product", "knn"], default="product"
    )
    parser.add_argument(
        "--legacy-unnormalized-stiffness", action="store_true",
        help="Reproduce the first product pilot without joint spectral scaling.",
    )
    parser.add_argument("--knn-neighbors", type=int, default=16)
    parser.add_argument(
        "--reference-mass-weight", type=float, default=0.0,
        help=(
            "Mix empirical mass with tensor trapezoid mass on the declared "
            "support nodes; positive values complete empirically unseen modes."
        ),
    )
    parser.add_argument(
        "--reference-mass-mixture", nargs="+", type=float, default=[],
        help="Average completed representers over these fixed rho values.",
    )
    parser.add_argument(
        "--interaction-projection",
        choices=["empirical-anova", "center-only"],
        default="empirical-anova",
    )
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--members", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--output", type=Path,
        default=HERE / "results/spatial_product_riesz.csv",
    )
    args = parser.parse_args()
    rows = list(read_rows(args.output))
    complete = {
        (str(row["dataset"]), str(row["model"]), int(row["seed"]), str(row["method"]))
        for row in rows
    }
    all_metadata: dict[str, object] = {}
    for name in args.datasets:
        dataset, columns, names = dataset_and_columns(name)
        variants, metadata = build_product_features(
            dataset,
            columns,
            base_bins=args.base_bins,
            product_bins=args.product_bins,
            strength=args.strength,
            seed=20260826,
            control_seed=args.control_seed,
            stiffness_family=args.stiffness_family,
            normalize_stiffness=not args.legacy_unnormalized_stiffness,
            knn_neighbors=args.knn_neighbors,
            interaction_projection=args.interaction_projection,
            reference_mass_weight=args.reference_mass_weight,
            reference_mass_mixture=tuple(args.reference_mass_mixture),
        )
        metadata["field_names"] = [names[index] for index in columns]
        all_metadata[name] = metadata
        methods = args.methods or list(variants)
        for model in args.models:
            if model == "tabm":
                target_parameters = count_parameters(
                    FlatTabM(
                        variants["anchor_only"]["train"].shape[1], 1,
                        args.width, args.depth, args.members,
                    )
                )
            elif model == "ft":
                target_parameters = ft_parameter_count(
                    FieldTokenTransformer(
                        token_dimensions("anchor_only", metadata), 1
                    )
                )
            else:
                target_parameters = parameter_count(
                    model, variants["anchor_only"]["train"].shape[1], 1,
                    args.width, args.depth,
                )
            for seed in args.seeds:
                for method in methods:
                    key = (name, model, seed, method)
                    if key in complete:
                        continue
                    features = variants[method]
                    prepared = make_prepared(dataset, features, {"method": method})
                    if model == "tabm":
                        result = train_tabm(
                            prepared, seed=seed, device=args.device,
                            width=args.width, depth=args.depth,
                            ensemble_size=args.members, epochs=args.epochs,
                            patience=args.patience,
                            target_parameters=target_parameters,
                        )
                    elif model == "ft":
                        result = train_ft(
                            prepared,
                            token_dimensions(method, metadata),
                            seed=seed,
                            device=args.device,
                            epochs=args.epochs,
                            patience=args.patience,
                            target_parameters=target_parameters,
                        )
                    else:
                        width, parameters = parameter_matched_width(
                            model, features["train"].shape[1], 1,
                            args.depth, target_parameters,
                        )
                        result, _ = train_model(
                            prepared, seed=seed, device=args.device,
                            model_name=model, width=width, depth=args.depth,
                            dropout=0.1, learning_rate=1e-3,
                            weight_decay=1e-4, batch_size=512,
                            max_epochs=args.epochs, patience=args.patience,
                        )
                        result.update(
                            {
                                "width": width,
                                "parameters": parameters,
                                "parameter_error_fraction": (
                                    parameters - target_parameters
                                ) / target_parameters,
                            }
                        )
                    row = {
                        "dataset": name, "task": dataset.task, "model": model,
                        "seed": seed, "method": method,
                        "strength": args.strength,
                        "target_parameters": target_parameters, **result,
                    }
                    rows.append(row)
                    complete.add(key)
                    write_rows(args.output, rows)
                    print(json.dumps(row, sort_keys=True), flush=True)
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(all_metadata, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
