"""Run the pre-registered Day 3 experiment branches.

Each branch writes an independent CSV so jobs may safely run on separate GPUs.
The command is resumable at the dataset/model/seed/representation level.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from .core import (
    DATA_ROOT,
    PARTS,
    Dataset,
    Prepared,
    apply_transform,
    base_schema,
    block_pair_geometry,
    category_codes,
    combine,
    condition_transform,
    contrast_block,
    cumulative_ordinal,
    diagonal_standardize,
    equivalence_diagnostics,
    exact_state_ple_and_identity,
    geometry,
    helmert,
    load_dataset,
    make_prepared,
    path_spectral,
    ple_blocks,
    procrustes_align,
    quantile_numeric,
    real_fourier_basis,
    residualize,
    standardize,
    target,
    train_model,
    whiten,
)


HERE = Path(__file__).resolve().parents[2]
RESULTS = HERE / "results" / "day3"
KAPPAS = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0)
SEEDS = (0, 1, 2, 3, 4)

ORDINAL = {
    "adult": {
        1: ["Preschool", "1st-4th", "5th-6th", "7th-8th", "9th", "10th", "11th", "12th", "HS-grad", "Some-college", "Assoc-voc", "Assoc-acdm", "Bachelors", "Masters", "Prof-school", "Doctorate"],
    },
    "black-friday": {
        1: ["0-17", "18-25", "26-35", "36-45", "46-50", "51-55", "55+"],
        3: ["0", "1", "2", "3", "4+"],
    },
    "diamond": {
        0: ["Fair", "Good", "Very Good", "Premium", "Ideal"],
        1: ["J", "I", "H", "G", "F", "E", "D"],
        2: ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"],
    },
}


def _read(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _append(path: Path, rows: list[dict[str, object]], row: dict[str, object]) -> None:
    rows.append(row)
    _write(path, rows)


def _row_base(dataset: Dataset, experiment: str, representation: str, seed: int, model: str, intervention: str) -> dict[str, object]:
    return {
        "experiment": experiment,
        "intervention_class": intervention,
        "dataset": dataset.name,
        "task": dataset.task,
        "model": model,
        "optimizer": "AdamW",
        "weight_decay": 1e-4,
        "seed": seed,
        "representation": representation,
        "split_fingerprint": dataset.split_fingerprint,
    }


def _run_one(
    dataset: Dataset,
    experiment: str,
    representation: str,
    seed: int,
    model: str,
    x: dict[str, np.ndarray],
    output: Path,
    rows: list[dict[str, object]],
    curves_path: Path,
    curves_rows: list[dict[str, object]],
    device: str,
    intervention: str = "A",
    regularizer: str = "standard",
    extra: dict[str, object] | None = None,
) -> None:
    key = (dataset.name, model, seed, representation, regularizer)
    completed = {(r.get("dataset"), r.get("model"), int(r.get("seed", -1)), r.get("representation"), r.get("regularizer", "standard")) for r in rows}
    if key in completed:
        return
    prepared = make_prepared(dataset, x, extra or {})
    fit, curves = train_model(prepared, seed=seed, device=device, model_name=model, regularizer=regularizer)
    row = _row_base(dataset, experiment, representation, seed, model, intervention)
    row["regularizer"] = regularizer
    row.update(geometry(x["train"]))
    if extra:
        row.update(extra)
    row.update(fit)
    _append(output, rows, row)
    for curve in curves:
        curve_row = {"experiment": experiment, "dataset": dataset.name, "model": model, "seed": seed, "representation": representation, "regularizer": regularizer, **curve}
        curves_rows.append(curve_row)
    _write(curves_path, curves_rows)
    print(f"{experiment:22s} {dataset.name:14s} {model:6s} s{seed} {representation:28s} metric={fit['test_metric']:.6f}", flush=True)


def _canonical_numeric(dataset: Dataset, kappa: float, transform_seed: int = 31000) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    assert dataset.x_num is not None
    schema = base_schema(dataset, include_num=True, include_cat=True)
    blocks, _ = ple_blocks(dataset.x_num, bins=32)
    transformed = []
    realized = []
    ranks = []
    for j, block in enumerate(blocks):
        white, meta = whiten(block)
        matrix = condition_transform(white["train"].shape[1], kappa, transform_seed + j)
        changed = apply_transform(white, matrix)
        transformed.append(changed)
        realized.append(float(np.linalg.cond(matrix)))
        ranks.append(int(meta["retained_rank"]))
    x = combine([schema, *transformed])
    return x, {
        "target_kappa": kappa,
        "realized_block_kappa_mean": float(np.mean(realized)),
        "realized_block_kappa_max": float(np.max(realized)),
        "block_ranks": ";".join(map(str, ranks)),
        "transform_seed": transform_seed,
        "global_scale_control": "geometric_mean_singular_value_1",
    }


def run_numeric(args: argparse.Namespace) -> None:
    output = RESULTS / "numeric_kappa.csv"
    curves_path = RESULTS / "curves_numeric_kappa.csv"
    rows, curves = _read(output), _read(curves_path)
    for name in args.datasets or ["adult", "california", "diamond"]:
        dataset = load_dataset(name)
        for kappa in args.kappas:
            x, extra = _canonical_numeric(dataset, kappa)
            representation = f"numeric_kappa_{kappa:g}"
            for model in args.models:
                for seed in args.seeds:
                    _run_one(dataset, "numeric_kappa", representation, seed, model, x, output, rows, curves_path, curves, args.device, extra=extra)


def _canonical_categorical(dataset: Dataset, kappa: float, transform_seed: int = 41000) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    assert dataset.x_cat is not None
    schema = base_schema(dataset, include_num=True, include_cat=False)
    blocks, realized, entropies, cardinalities = [], [], [], []
    for j in range(dataset.x_cat["train"].shape[1]):
        codes, levels = category_codes(dataset.x_cat, j)
        block = contrast_block(codes, len(levels))
        white, _ = whiten(block)
        matrix = condition_transform(white["train"].shape[1], kappa, transform_seed + j)
        blocks.append(apply_transform(white, matrix))
        realized.append(float(np.linalg.cond(matrix)))
        counts = np.bincount(codes["train"], minlength=len(levels)).astype(float)
        probability = counts / counts.sum()
        entropies.append(float(-np.sum(probability * np.log(np.maximum(probability, 1e-300)))))
        cardinalities.append(len(levels))
    return combine([schema, *blocks]), {
        "target_kappa": kappa,
        "realized_block_kappa_mean": float(np.mean(realized)),
        "realized_block_kappa_max": float(np.max(realized)),
        "categorical_cardinalities": ";".join(map(str, cardinalities)),
        "categorical_entropy_mean": float(np.mean(entropies)),
        "transform_seed": transform_seed,
    }


def run_categorical(args: argparse.Namespace) -> None:
    output = RESULTS / "categorical_kappa.csv"
    curves_path = RESULTS / "curves_categorical_kappa.csv"
    rows, curves = _read(output), _read(curves_path)
    for name in args.datasets or ["adult", "diamond"]:
        dataset = load_dataset(name)
        for kappa in args.kappas:
            x, extra = _canonical_categorical(dataset, kappa)
            representation = f"categorical_kappa_{kappa:g}"
            for model in args.models:
                for seed in args.seeds:
                    _run_one(dataset, "categorical_kappa", representation, seed, model, x, output, rows, curves_path, curves, args.device, extra=extra)


def _ordinal_blocks(dataset: Dataset, variant: str) -> tuple[list[dict[str, np.ndarray]], dict[str, object]]:
    assert dataset.x_cat is not None
    blocks = []
    spectra = []
    for column, levels in ORDINAL[dataset.name].items():
        codes, _ = category_codes(dataset.x_cat, column, levels)
        k = len(levels)
        if variant == "local":
            block = contrast_block(codes, k)
        elif variant == "cumulative":
            block = cumulative_ordinal(codes, k)
        elif variant == "cumulative_standardized":
            block = diagonal_standardize(cumulative_ordinal(codes, k))
        elif variant == "path_spectral":
            block = contrast_block(codes, k, path_spectral(k))
        elif variant == "whitened":
            block, _ = whiten(cumulative_ordinal(codes, k))
        else:
            raise ValueError(variant)
        blocks.append(block)
        spectra.append(geometry(block["train"])["condition_number"])
    return blocks, {"ordinal_columns": ";".join(map(str, ORDINAL[dataset.name])), "ordinal_block_condition_mean": float(np.mean(spectra))}


def _nonordinal_schema(dataset: Dataset) -> dict[str, np.ndarray]:
    blocks = []
    if dataset.x_num is not None:
        blocks.append(quantile_numeric(dataset.x_num))
    if dataset.x_bin is not None:
        blocks.append(standardize({p: v.astype(float) for p, v in dataset.x_bin.items()}))
    assert dataset.x_cat is not None
    ordinal_columns = set(ORDINAL[dataset.name])
    for j in range(dataset.x_cat["train"].shape[1]):
        if j in ordinal_columns:
            continue
        codes, levels = category_codes(dataset.x_cat, j)
        blocks.append(contrast_block(codes, len(levels)))
    return combine(blocks)


def run_ordinal(args: argparse.Namespace) -> None:
    output = RESULTS / "ordinal_basis.csv"
    curves_path = RESULTS / "curves_ordinal_basis.csv"
    rows, curves = _read(output), _read(curves_path)
    variants = ["local", "cumulative", "cumulative_standardized", "path_spectral", "whitened"]
    for name in args.datasets or ["adult", "black-friday", "diamond"]:
        dataset = load_dataset(name)
        schema = _nonordinal_schema(dataset)
        for variant in variants:
            blocks, extra = _ordinal_blocks(dataset, variant)
            x = combine([schema, *blocks])
            for model in args.models:
                for seed in args.seeds:
                    _run_one(dataset, "ordinal_basis", variant, seed, model, x, output, rows, curves_path, curves, args.device, extra=extra)

    # Exact controlled sweep starts from each block's sample-whitened coordinates.
    output = RESULTS / "ordinal_kappa.csv"
    curves_path = RESULTS / "curves_ordinal_kappa.csv"
    rows, curves = _read(output), _read(curves_path)
    for name in args.datasets or ["adult", "black-friday", "diamond"]:
        dataset = load_dataset(name)
        schema = _nonordinal_schema(dataset)
        white_blocks, _ = _ordinal_blocks(dataset, "whitened")
        for kappa in args.kappas:
            changed = []
            realized = []
            for j, block in enumerate(white_blocks):
                matrix = condition_transform(block["train"].shape[1], kappa, 51000 + j)
                changed.append(apply_transform(block, matrix))
                realized.append(float(np.linalg.cond(matrix)))
            x = combine([schema, *changed])
            extra = {"target_kappa": kappa, "realized_block_kappa_mean": float(np.mean(realized)), "transform_seed": 51000}
            representation = f"ordinal_kappa_{kappa:g}"
            for model in args.models:
                for seed in args.seeds:
                    _run_one(dataset, "ordinal_kappa", representation, seed, model, x, output, rows, curves_path, curves, args.device, extra=extra)


def run_whitening(args: argparse.Namespace) -> None:
    dataset = load_dataset("adult")
    assert dataset.x_num is not None
    schema = base_schema(dataset, include_num=True, include_cat=True)
    ple_blocks_selected, identity_blocks = [], []
    diagnostics = {}
    for column in (3, 4):
        ple, identity, levels = exact_state_ple_and_identity(dataset.x_num, column)
        ple_blocks_selected.append(ple)
        identity_blocks.append(identity)
        diagnostics[str(column)] = {"levels": len(levels), **equivalence_diagnostics(ple, identity)}
    diag_path = RESULTS / "ple_identity_equivalence_exact.json"
    diag_path.parent.mkdir(parents=True, exist_ok=True)
    diag_path.write_text(json.dumps(diagnostics, indent=2))
    ple_full, identity_full = combine(ple_blocks_selected), combine(identity_blocks)
    variants: dict[str, tuple[dict[str, np.ndarray], dict[str, np.ndarray]]] = {
        "raw": (ple_full, identity_full),
        "centered": (
            {p: ple_full[p] - ple_full["train"].mean(axis=0) for p in PARTS},
            {p: identity_full[p] - identity_full["train"].mean(axis=0) for p in PARTS},
        ),
        "standardized": (diagonal_standardize(ple_full), diagonal_standardize(identity_full)),
    }
    ple_white, _ = whiten(ple_full)
    identity_white, _ = whiten(identity_full)
    variants["whitened"] = (ple_white, identity_white)
    identity_aligned, _ = procrustes_align(ple_white, identity_white)
    variants["aligned"] = (ple_white, identity_aligned)
    output = RESULTS / "ple_identity_whitening_exact.csv"
    curves_path = RESULTS / "curves_ple_identity_whitening_exact.csv"
    rows, curves = _read(output), _read(curves_path)
    for variant, (ple, identity) in variants.items():
        for family, block in (("ple", ple), ("identity", identity)):
            x = combine([schema, block])
            representation = f"{family}_{variant}"
            extra = {"family": family, "canonicalization": variant, "selected_columns": "3;4"}
            for model in args.models:
                for seed in args.seeds:
                    _run_one(dataset, "ple_identity_whitening", representation, seed, model, x, output, rows, curves_path, curves, args.device, extra=extra)


def run_regularizer(args: argparse.Namespace) -> None:
    output = RESULTS / "invariant_regularizer.csv"
    curves_path = RESULTS / "curves_invariant_regularizer.csv"
    rows, curves = _read(output), _read(curves_path)
    kappas = args.kappas if args.kappas != list(KAPPAS) else [1.0, 30.0, 300.0, 3000.0]
    for name in args.datasets or ["adult", "diamond"]:
        dataset = load_dataset(name)
        for kappa in kappas:
            x, extra = _canonical_numeric(dataset, kappa)
            for regularizer in ("standard", "no_first_wd", "invariant"):
                representation = f"numeric_kappa_{kappa:g}_{regularizer}"
                for model in args.models:
                    for seed in args.seeds:
                        _run_one(dataset, "invariant_regularizer", representation, seed, model, x, output, rows, curves_path, curves, args.device, intervention="B", regularizer=regularizer, extra=extra)


def run_block(args: argparse.Namespace) -> None:
    output = RESULTS / "block_residualization.csv"
    curves_path = RESULTS / "curves_block_residualization.csv"
    rows, curves = _read(output), _read(curves_path)
    for name in args.datasets or ["diamond", "adult"]:
        dataset = load_dataset(name)
        assert dataset.x_num is not None and dataset.x_cat is not None
        numeric_blocks, _ = ple_blocks(dataset.x_num, bins=32)
        numeric = combine([whiten(block)[0] for block in numeric_blocks])
        categorical = combine([
            contrast_block(*((lambda pair: (pair[0], len(pair[1])))(category_codes(dataset.x_cat, j))))
            for j in range(dataset.x_cat["train"].shape[1])
        ])
        categorical_white, _ = whiten(categorical)
        categorical_standardized = diagonal_standardize(categorical)
        categorical_perp, beta = residualize(numeric, categorical)
        numeric_white, _ = whiten(numeric)
        block_perp_white, _ = whiten(categorical_perp)
        joint = combine([numeric, categorical])
        joint_white, _ = whiten(joint)
        variants = {
            "raw_joint": joint,
            "standardized_categorical": combine([numeric, categorical_standardized]),
            "block_residualized": combine([numeric, categorical_perp]),
            "block_residualized_whitened": combine([numeric_white, block_perp_white]),
            "blockwise_whitened": combine([numeric_white, categorical_white]),
            "joint_whitened": joint_white,
        }
        before = block_pair_geometry(numeric["train"], categorical["train"])
        after = block_pair_geometry(numeric["train"], categorical_perp["train"])
        diagnostics = {"dataset": name, "before": before, "after": after, "beta_shape": list(beta.shape), "joint_reconstruction_error": float(np.linalg.norm(categorical["train"] - (categorical_perp["train"] + numeric["train"] @ beta)) / np.linalg.norm(categorical["train"]))}
        path = RESULTS / f"block_diagnostics_{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(diagnostics, indent=2))
        for representation, x in variants.items():
            extra = {"cross_gram_before": before["normalized_cross_gram"], "cross_gram_after": after["normalized_cross_gram"], "joint_reconstruction_error": diagnostics["joint_reconstruction_error"]}
            for model in args.models:
                for seed in args.seeds:
                    _run_one(dataset, "block_residualization", representation, seed, model, x, output, rows, curves_path, curves, args.device, extra=extra)


def _cycle_dataset(seed: int = 2026, k: int = 24) -> tuple[Dataset, dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    sizes = {"train": 12000, "val": 3000, "test": 6000}
    codes, y = {}, {}
    for i, p in enumerate(PARTS):
        c = rng.integers(0, k, size=sizes[p])
        signal = 1.2 * np.sin(2 * np.pi * c / k) + 0.65 * np.cos(6 * np.pi * c / k) + 0.35 * ((c == 7).astype(float) - (c == 18).astype(float))
        codes[p] = c
        y[p] = (signal + rng.normal(0, 0.7, size=len(c))).astype(np.float32)
    dataset = Dataset("synthetic-hour-of-day", "regression", None, None, None, y, 1, "synthetic-2026")
    return dataset, codes


def run_cyclic(args: argparse.Namespace) -> None:
    dataset, codes = _cycle_dataset()
    k = 24
    onehot = {p: np.eye(k)[codes[p]] @ helmert(k) for p in PARTS}
    variants = {"centered_onehot": onehot}
    for phase in (0, 1, 5, 11):
        variants[f"full_fourier_phase_{phase}"] = {p: real_fourier_basis(k, phase)[codes[p]] for p in PARTS}
    first_harmonic = real_fourier_basis(k)[:, :2]
    variants["truncated_first_harmonic"] = {p: first_harmonic[codes[p]] for p in PARTS}
    white, _ = whiten(onehot)
    for kappa in args.kappas:
        matrix = condition_transform(k - 1, kappa, 61000)
        variants[f"cyclic_kappa_{kappa:g}"] = apply_transform(white, matrix)
    diag = {
        "onehot_to_fourier": equivalence_diagnostics(onehot, variants["full_fourier_phase_0"]),
        "phases": {str(phase): geometry(variants[f"full_fourier_phase_{phase}"]["train"]) for phase in (0, 1, 5, 11)},
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "cyclic_equivalence.json").write_text(json.dumps(diag, indent=2))
    output = RESULTS / "cyclic_geometry.csv"
    curves_path = RESULTS / "curves_cyclic_geometry.csv"
    rows, curves = _read(output), _read(curves_path)
    for representation, x in variants.items():
        intervention = "C" if representation == "truncated_first_harmonic" else "A"
        extra = {"target_kappa": float(representation.rsplit("_", 1)[-1]) if representation.startswith("cyclic_kappa") else math.nan, "phase": int(representation.rsplit("_", 1)[-1]) if representation.startswith("full_fourier") else math.nan, "synthetic_control": True}
        for model in args.models:
            for seed in args.seeds:
                _run_one(dataset, "cyclic_geometry", representation, seed, model, x, output, rows, curves_path, curves, args.device, intervention=intervention, extra=extra)


def audit() -> None:
    rows = []
    for name in ("adult", "black-friday", "california", "churn", "diamond", "higgs-small", "house", "microsoft", "otto"):
        dataset = load_dataset(name)
        for kind, source in (("numerical", dataset.x_num), ("binary", dataset.x_bin), ("categorical", dataset.x_cat)):
            if source is None:
                continue
            for j in range(source["train"].shape[1]):
                values = source["train"][:, j]
                missing = float(np.mean(np.asarray([str(v) == "nan" for v in values]))) if values.dtype.kind in "USO" else float(np.mean(~np.isfinite(values)))
                semantic = "ordinal" if kind == "categorical" and name in ORDINAL and j in ORDINAL[name] else ("nominal" if kind == "categorical" else kind)
                unique, counts = np.unique(values, return_counts=True)
                probability = counts.astype(float) / counts.sum()
                entropy = float(-np.sum(probability * np.log(np.maximum(probability, 1e-300))))
                gini = float(1.0 - np.sum(probability**2))
                covariance_condition = math.nan
                if kind == "categorical" and len(unique) > 1:
                    covariance = np.diag(probability) - np.outer(probability, probability)
                    eigen = np.linalg.eigvalsh(covariance)
                    kept = eigen[eigen > eigen[-1] * 1e-10]
                    covariance_condition = float(math.sqrt(kept[-1] / kept[0]))
                rows.append({
                    "dataset": name,
                    "column": j,
                    "raw_dtype": str(values.dtype),
                    "semantic_type": semantic,
                    "ordinal_levels_order": ";".join(ORDINAL.get(name, {}).get(j, [])),
                    "cardinality": int(len(unique)),
                    "missing_rate": missing,
                    "entropy": entropy,
                    "gini": gini,
                    "min_frequency": int(counts.min()),
                    "median_frequency": float(np.median(counts)),
                    "max_frequency": int(counts.max()),
                    "categorical_covariance_condition": covariance_condition,
                    "split_fingerprint": dataset.split_fingerprint,
                })
    _write(RESULTS / "structured_feature_audit.csv", rows)
    git_hash = subprocess.run(["git", "-C", str(HERE.parent.parent), "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    manifest = {"git_commit": git_hash, "python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "data_root": str(DATA_ROOT)}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "environment.json").write_text(json.dumps(manifest, indent=2))


RUNNERS = {
    "numeric": run_numeric,
    "categorical": run_categorical,
    "ordinal": run_ordinal,
    "whitening": run_whitening,
    "regularizer": run_regularizer,
    "block": run_block,
    "cyclic": run_cyclic,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=[*RUNNERS, "audit", "all"])
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--models", nargs="+", choices=["mlp", "resnet"], default=["mlp"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--kappas", nargs="+", type=float, default=list(KAPPAS))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    audit()
    if args.experiment == "audit":
        return
    names = list(RUNNERS) if args.experiment == "all" else [args.experiment]
    for name in names:
        RUNNERS[name](args)


if __name__ == "__main__":
    main()
