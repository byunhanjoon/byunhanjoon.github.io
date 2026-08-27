#!/usr/bin/env python3
"""Leakage-safe pilot of cross-fitted residual Riesz representers.

The same out-of-fold LightGBM anchor and residuals are shared by every method.
The only difference among mass/correct/control variants is the per-field Riesz
operator used to turn the residual covector into one scalar field function.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
from scipy.linalg import eigh
from scipy.stats import norm
from sklearn.model_selection import KFold

from support_heat_pilot import (
    HERE,
    PARTS,
    base_schema,
    clean_numeric,
    combine,
    hat_basis,
    linear_basis,
    load_dataset,
    load_tabred,
    make_prepared,
    parameter_count,
    parameter_matched_width,
    quantile_nodes,
    read_rows,
    support_nodes,
    train_model,
    write_rows,
)


METHODS = (
    "quantile_ple",
    "raple_raw",
    "anchor_only",
    "anchor_mass_representer",
    "anchor_riesz_representer",
    "anchor_wrong_representer",
    "anchor_isospectral_representer",
)
RAPLE_ROOT = Path(os.environ.get(
    "DAY4_RAPLE_ROOT",
    "/home/byunhanjoon/2027ICLR/projects/multifeature_ple_tabular",
))


def standardized_scalar(parts: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    mean = np.mean(parts["train"], axis=0)
    scale = np.std(parts["train"], axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    return {
        part: np.asarray((values - mean) / scale, dtype=np.float32)
        for part, values in parts.items()
    }


def fit_anchor(dataset, features: dict[str, np.ndarray], seed: int):
    if dataset.task != "regression":
        raise ValueError("This first residual-representer pilot is regression-only")
    y_train = np.asarray(dataset.y["train"], dtype=np.float64)
    folds = list(KFold(3, shuffle=True, random_state=seed).split(y_train))
    oof = np.empty(len(y_train), dtype=np.float64)

    def model(offset: int):
        return lgb.LGBMRegressor(
            objective="regression",
            n_estimators=250,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=64,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_lambda=1.0,
            random_state=seed + offset,
            n_jobs=16,
            verbosity=-1,
        )

    for fold, (fit, hold) in enumerate(folds):
        anchor = model(fold + 1)
        anchor.fit(features["train"][fit], y_train[fit])
        oof[hold] = anchor.predict(features["train"][hold])
    full = model(0)
    full.fit(features["train"], y_train)
    predictions = {
        "train": oof,
        "val": full.predict(features["val"]),
        "test": full.predict(features["test"]),
    }
    return predictions, y_train - oof, folds


def fit_shared_raple(dataset, seed: int):
    """Return RAPLE features and the exact anchor residual they were built on."""
    if not RAPLE_ROOT.exists():
        raise FileNotFoundError(RAPLE_ROOT)
    sys.path.insert(0, str(RAPLE_ROOT))
    from raple import RAPLEEncoder

    assert dataset.x_num is not None
    counts = [len(dataset.y[part]) for part in PARTS]
    x_num = np.vstack([dataset.x_num[part] for part in PARTS])
    y = np.concatenate([dataset.y[part] for part in PARTS]).astype(np.float32)
    auxiliary_parts = base_schema(dataset, seed=seed, include_num=False)
    x_aux = np.vstack([auxiliary_parts[part] for part in PARTS])
    train_indices = np.arange(counts[0])
    encoder = RAPLEEncoder(seed=seed, n_estimators=250, n_jobs=16)
    result = encoder.fit_transform(
        x_num, y, train_indices, x_aux=x_aux if x_aux.shape[1] else None
    )

    starts = np.cumsum([0, *counts])
    raple_features = {
        part: result.features[starts[i] : starts[i + 1]]
        for i, part in enumerate(PARTS)
    }
    anchor = {
        part: result.anchor[starts[i] : starts[i + 1]]
        for i, part in enumerate(PARTS)
    }
    folds = list(KFold(3, shuffle=True, random_state=seed).split(train_indices))
    residual = np.asarray(dataset.y["train"], dtype=np.float64) - anchor["train"]
    metadata = {
        "raple_feature_count": int(result.features.shape[1]),
        "raple_selected_features": result.selected_features.tolist(),
        "raple_relation_types": result.relation_types,
        "raple_selected_pairs": result.selected_pairs,
    }
    return raple_features, anchor, residual, folds, metadata


def field_forms(
    phi_train: np.ndarray,
    nodes: np.ndarray,
    column: int,
    strength: float,
    wrong_permutation_seed: int = 991_337,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mass = phi_train.T @ phi_train / len(phi_train)
    gaps = np.maximum(np.diff(nodes), 1e-12)
    stiffness = np.zeros_like(mass)
    for edge, conductance in enumerate(1.0 / gaps):
        stiffness[edge, edge] += conductance
        stiffness[edge + 1, edge + 1] += conductance
        stiffness[edge, edge + 1] -= conductance
        stiffness[edge + 1, edge] -= conductance

    eigenvalues, vectors = np.linalg.eigh(mass)
    threshold = max(float(eigenvalues[-1]), 0.0) * 1e-10
    keep = eigenvalues > threshold
    if not np.any(keep):
        return mass, mass.copy(), mass.copy(), mass.copy()
    whitener = vectors[:, keep] / np.sqrt(eigenvalues[keep])[None, :]
    generalized = np.maximum(eigh(whitener.T @ stiffness @ whitener)[0], 0.0)
    positive = generalized[generalized > 1e-12]
    scale = float(np.median(positive)) if len(positive) else 1.0
    scaled_stiffness = stiffness / max(scale, 1e-12)

    permutation = np.random.default_rng(
        wrong_permutation_seed + column
    ).permutation(len(nodes))
    wrong = scaled_stiffness[np.ix_(permutation, permutation)]

    # Harder negative control: preserve the *generalized* spectrum relative to
    # empirical mass while randomizing its modes. A node permutation preserves
    # the Euclidean spectrum of S but generally not the spectrum of (S, M).
    # In M-whitened coordinates A has the generalized eigenvalues. A Haar-like
    # orthogonal rotation changes their orientation but keeps them exactly.
    whitened_stiffness = whitener.T @ scaled_stiffness @ whitener
    generalized_values = np.maximum(
        np.linalg.eigvalsh(whitened_stiffness), 0.0
    )
    rng = np.random.default_rng(wrong_permutation_seed + 1_000_003 + column)
    rotation, triangular = np.linalg.qr(
        rng.normal(size=(len(generalized_values), len(generalized_values)))
    )
    signs = np.where(np.diag(triangular) < 0, -1.0, 1.0)
    rotation = rotation * signs[None, :]
    whitened_control = (
        rotation * generalized_values[None, :]
    ) @ rotation.T
    mass_sqrt = vectors[:, keep] * np.sqrt(eigenvalues[keep])[None, :]
    isospectral_stiffness = mass_sqrt @ whitened_control @ mass_sqrt.T
    isospectral_stiffness = 0.5 * (
        isospectral_stiffness + isospectral_stiffness.T
    )
    return (
        mass,
        mass + strength * scaled_stiffness,
        mass + strength * wrong,
        mass + strength * isospectral_stiffness,
    )


def representer_values(
    phi: dict[str, np.ndarray],
    residual: np.ndarray,
    folds,
    operator: np.ndarray,
) -> dict[str, np.ndarray]:
    inverse = np.linalg.pinv(operator, rcond=1e-10)
    train = np.empty(len(residual), dtype=np.float64)
    for fit, hold in folds:
        covector = phi["train"][fit].T @ residual[fit] / len(fit)
        train[hold] = phi["train"][hold] @ (inverse @ covector)
    covector = phi["train"].T @ residual / len(residual)
    coefficients = inverse @ covector
    return {
        "train": train[:, None],
        "val": (phi["val"] @ coefficients)[:, None],
        "test": (phi["test"] @ coefficients)[:, None],
    }


def oof_representer_values(
    phi_train: np.ndarray,
    residual: np.ndarray,
    folds,
    operator: np.ndarray,
) -> np.ndarray:
    """Compute only OOF training values for cheap field screening."""
    inverse = np.linalg.pinv(operator, rcond=1e-10)
    train = np.empty(len(residual), dtype=np.float64)
    for fit, hold in folds:
        covector = phi_train[fit].T @ residual[fit] / len(fit)
        train[hold] = phi_train[hold] @ (inverse @ covector)
    return train


def build_features(
    dataset,
    bins: int,
    strength: float,
    seed: int,
    *,
    shared_raple_anchor: bool = False,
    wrong_permutation_seed: int = 991_337,
    max_representer_fields: int = 0,
    representer_fields: list[int] | None = None,
    field_selector: str = "mass_energy",
    positive_selector_only: bool = False,
    selector_min_score: float = 0.0,
    selector_fdr: float = 0.0,
):
    assert dataset.x_num is not None
    if not (0.0 <= selector_fdr < 1.0):
        raise ValueError("selector_fdr must lie in [0, 1)")
    clean = clean_numeric(dataset.x_num)
    qblocks = []
    for column in range(clean["train"].shape[1]):
        nodes = quantile_nodes(clean["train"][:, column], bins)
        qblocks.append(
            {part: linear_basis(clean[part][:, column], nodes) for part in PARTS}
        )
    nonnumeric = base_schema(dataset, seed=seed, include_num=False)
    ple = combine([*qblocks, nonnumeric])

    # The anchor receives compact train-fitted scalar/categorical features, not
    # the wide PLE vector. Its OOF predictions are identical for every method.
    raple_feature = None
    raple_metadata = {}
    if shared_raple_anchor:
        raple_feature, anchor_prediction, residual, folds, raple_metadata = (
            fit_shared_raple(dataset, seed)
        )
    else:
        anchor_input = base_schema(dataset, seed=seed)
        anchor_prediction, residual, folds = fit_anchor(dataset, anchor_input, seed)
    anchor_feature = standardized_scalar(
        {part: values[:, None] for part, values in anchor_prediction.items()}
    )

    metadata = []
    for column in range(clean["train"].shape[1]):
        nodes = support_nodes(clean["train"][:, column], bins, 0.35)
        hats_train = hat_basis(clean["train"][:, column], nodes)
        phi_train = hats_train - hats_train.mean(axis=0)
        operators = field_forms(
            phi_train, nodes, column, strength, wrong_permutation_seed
        )
        field_metadata = {"column": column, "nodes": len(nodes)}
        covector = phi_train.T @ residual / len(residual)
        for kind, operator in zip(
            ("mass", "riesz", "wrong", "isospectral"), operators
        ):
            coefficients = np.linalg.pinv(operator, rcond=1e-10) @ covector
            field_metadata[f"{kind}_energy"] = float(covector @ coefficients)
            train_values = oof_representer_values(
                phi_train, residual, folds, operator
            )
            alignment = float(np.mean(residual * train_values))
            representer_power = float(np.mean(train_values * train_values))
            alignment_se = float(
                np.std(residual * train_values, ddof=1) / np.sqrt(len(residual))
            )
            field_metadata[f"{kind}_oof_gain"] = float(
                np.mean(2 * residual * train_values - train_values * train_values)
            )
            field_metadata[f"{kind}_oof_alignment"] = alignment
            field_metadata[f"{kind}_oof_alignment_se"] = alignment_se
            field_metadata[f"{kind}_oof_alignment_z"] = float(
                alignment / max(alignment_se, 1e-20)
            )
            field_metadata[f"{kind}_oof_optimal_gain"] = float(
                np.sign(alignment) * alignment * alignment
                / max(representer_power, 1e-20)
            )
        field_metadata["energy_gap"] = float(
            field_metadata["riesz_energy"] - field_metadata["wrong_energy"]
        )
        field_metadata["isospectral_gap"] = float(
            field_metadata["riesz_energy"]
            - field_metadata["isospectral_energy"]
        )
        field_metadata["isospectral_retention_gap"] = float(
            field_metadata["isospectral_gap"]
            / max(field_metadata["mass_energy"], 1e-30)
        )
        metadata.append(field_metadata)

    if representer_fields is not None:
        selected_fields = sorted(set(representer_fields))
        invalid = [
            index for index in selected_fields
            if index < 0 or index >= len(metadata)
        ]
        if invalid:
            raise ValueError(f"representer field indices out of range: {invalid}")
    elif max_representer_fields > 0 and (
        max_representer_fields < len(metadata)
        or positive_selector_only
        or selector_min_score > 0
        or selector_fdr > 0
    ):
        selector_key = {
            "mass_energy": "mass_energy",
            "mass_oof_gain": "mass_oof_gain",
            "mass_oof_optimal_gain": "mass_oof_optimal_gain",
            "mass_oof_alignment_z": "mass_oof_alignment_z",
            "energy": "riesz_energy",
            "energy_gap": "energy_gap",
            "isospectral_gap": "isospectral_gap",
            "isospectral_retention_gap": "isospectral_retention_gap",
            "oof_gain": "riesz_oof_gain",
            "oof_optimal_gain": "riesz_oof_optimal_gain",
            "oof_alignment_z": "riesz_oof_alignment_z",
        }[field_selector]
        candidates = list(range(len(metadata)))
        if selector_fdr > 0:
            if not selector_key.endswith("_z"):
                raise ValueError("FDR screening requires an alignment-z selector")
            ranked = sorted(
                candidates,
                key=lambda index: float(norm.sf(metadata[index][selector_key])),
            )
            accepted = 0
            for rank, index in enumerate(ranked, start=1):
                p_value = float(norm.sf(metadata[index][selector_key]))
                if p_value <= selector_fdr * rank / len(ranked):
                    accepted = rank
            candidates = ranked[:accepted]
        if positive_selector_only or selector_min_score > 0:
            candidates = [
                index for index in candidates
                if float(metadata[index][selector_key])
                > max(0.0 if positive_selector_only else -np.inf, selector_min_score)
            ]
        selected_fields = sorted(
            candidates,
            key=lambda index: float(metadata[index][selector_key]),
            reverse=True,
        )[:max_representer_fields]
        selected_fields.sort()
    else:
        selected_fields = list(range(len(metadata)))

    by_kind = {
        kind: {part: [] for part in PARTS}
        for kind in ("mass", "riesz", "wrong", "isospectral")
    }
    for column in selected_fields:
        nodes = support_nodes(clean["train"][:, column], bins, 0.35)
        hats = {
            part: hat_basis(clean[part][:, column], nodes) for part in PARTS
        }
        mean = hats["train"].mean(axis=0)
        phi = {part: values - mean for part, values in hats.items()}
        operators = field_forms(
            phi["train"], nodes, column, strength, wrong_permutation_seed
        )
        for kind, operator in zip(
            ("mass", "riesz", "wrong", "isospectral"), operators
        ):
            values = standardized_scalar(
                representer_values(phi, residual, folds, operator)
            )
            for part in PARTS:
                by_kind[kind][part].append(values[part])

    blocks = {
        kind: {
            part: (
                np.column_stack(
                    values[part]
                ).astype(np.float32)
                if selected_fields
                else np.empty((len(dataset.y[part]), 0), dtype=np.float32)
            )
            for part in PARTS
        }
        for kind, values in by_kind.items()
    }
    variants = {
        "quantile_ple": ple,
        "anchor_only": combine([ple, anchor_feature]),
        "anchor_mass_representer": combine([ple, anchor_feature, blocks["mass"]]),
        "anchor_riesz_representer": combine([ple, anchor_feature, blocks["riesz"]]),
        "anchor_wrong_representer": combine([ple, anchor_feature, blocks["wrong"]]),
        "anchor_isospectral_representer": combine(
            [ple, anchor_feature, blocks["isospectral"]]
        ),
    }
    if raple_feature is not None:
        variants["raple_raw"] = combine(
            [ple, standardized_scalar(raple_feature)]
        )
    return variants, {
        "fields": metadata,
        "selected_fields": selected_fields,
        "representer_fields": representer_fields,
        "field_selector": field_selector,
        "max_representer_fields": max_representer_fields,
        "positive_selector_only": positive_selector_only,
        "selector_min_score": selector_min_score,
        "selector_fdr": selector_fdr,
        "wrong_permutation_seed": wrong_permutation_seed,
        **raple_metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["california"])
    parser.add_argument("--models", nargs="+", default=["mlp", "resnet"])
    parser.add_argument("--methods", nargs="+", choices=METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260847])
    parser.add_argument("--bins", type=int, default=32)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--wrong-permutation-seed", type=int, default=991_337)
    parser.add_argument("--max-representer-fields", type=int, default=0)
    parser.add_argument("--representer-fields", nargs="+", type=int)
    parser.add_argument(
        "--field-selector",
        choices=(
            "mass_energy", "mass_oof_gain", "mass_oof_optimal_gain",
            "mass_oof_alignment_z", "energy", "energy_gap",
            "isospectral_gap", "isospectral_retention_gap", "oof_gain",
            "oof_optimal_gain", "oof_alignment_z"
        ),
        default="mass_energy",
    )
    parser.add_argument("--positive-selector-only", action="store_true")
    parser.add_argument("--selector-min-score", type=float, default=0.0)
    parser.add_argument("--selector-fdr", type=float, default=0.0)
    parser.add_argument(
        "--shared-raple-anchor",
        action="store_true",
        help="Add raw RAPLE and build every representer from RAPLE's exact OOF anchor.",
    )
    parser.add_argument("--max-train-rows", type=int, default=50000)
    parser.add_argument("--max-eval-rows", type=int, default=15000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--output", type=Path,
        default=HERE / "results/residual_riesz_pilot.csv",
    )
    args = parser.parse_args()

    rows: list[dict[str, object]] = list(read_rows(args.output))
    completed = {
        (str(r["dataset"]), str(r["model"]), int(r["seed"]), str(r["method"]))
        for r in rows
    }
    all_metadata = {}
    for name in args.datasets:
        if name.startswith("tabred-"):
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
        variants, metadata = build_features(
            dataset,
            args.bins,
            args.strength,
            20260826,
            shared_raple_anchor=args.shared_raple_anchor,
            wrong_permutation_seed=args.wrong_permutation_seed,
            max_representer_fields=args.max_representer_fields,
            representer_fields=args.representer_fields,
            field_selector=args.field_selector,
            positive_selector_only=args.positive_selector_only,
            selector_min_score=args.selector_min_score,
            selector_fdr=args.selector_fdr,
        )
        all_metadata[name] = metadata
        methods = args.methods or [method for method in METHODS if method in variants]
        for model in args.models:
            baseline_parameters = parameter_count(
                model, variants["quantile_ple"]["train"].shape[1], 1,
                args.width, args.depth,
            )
            for seed in args.seeds:
                for method in methods:
                    key = (name, model, seed, method)
                    if key in completed:
                        continue
                    features = variants[method]
                    width, parameters = parameter_matched_width(
                        model, features["train"].shape[1], 1, args.depth,
                        baseline_parameters,
                    )
                    result, _ = train_model(
                        make_prepared(dataset, features, {"method": method}),
                        seed=seed,
                        device=args.device,
                        model_name=model,
                        width=width,
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
                        "model": model,
                        "seed": seed,
                        "method": method,
                        "strength": args.strength,
                        "width": width,
                        "baseline_parameter_count": baseline_parameters,
                        "parameter_count": parameters,
                        "parameter_error_fraction": (
                            parameters - baseline_parameters
                        ) / baseline_parameters,
                        **result,
                    }
                    rows.append(row)
                    write_rows(args.output, rows)
                    print(json.dumps(row, sort_keys=True), flush=True)
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(all_metadata, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
