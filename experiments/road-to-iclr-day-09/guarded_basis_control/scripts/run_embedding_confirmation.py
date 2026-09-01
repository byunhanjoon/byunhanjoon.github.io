#!/usr/bin/env python3
"""Run the locked PLE/RBF basis grid and Gram-after-embedding control."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from guarded_basis.blockguard import gram_interface  # noqa: E402
from guarded_basis.common import (  # noqa: E402
    SAFE_ROOT,
    bd,
    cached_representation_predictions,
    development_specs,
    disagreement,
    load_blocks,
    load_prediction_bundle,
    load_protocol,
    mix_predictions,
    normalized_excess_risk,
    task_error,
    write_json,
)
from guarded_basis.gating import guarded_evidence, select_g2, strip_samples  # noqa: E402
from safe_basis.embeddings import embedding_orbit  # noqa: E402


BACKBONES = ("controlled_mlp", "tabm_d", "resnet_tabular")


def prior_prediction(
    *, model: str, dataset: str, seed: int, embedding: str, dimension: int, filename: str
) -> tuple[dict[str, np.ndarray], dict[str, Any]] | None:
    path = (
        SAFE_ROOT
        / "results"
        / "raw"
        / "embeddings"
        / model
        / dataset
        / f"seed_{seed}"
        / embedding
        / f"k{dimension}"
        / filename
    )
    if not path.exists() or not path.with_suffix(".json").exists():
        return None
    prediction, metadata = load_prediction_bundle(path)
    return prediction, {**metadata, "source": "frozen_safe_basis_embedding", "source_path": str(path)}


def raw_prediction(
    *, blocks: Any, rep: Any, model: str, seed: int, embedding: str, dimension: int, device: str
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    prior = prior_prediction(
        model=model,
        dataset=blocks.dataset.key,
        seed=seed,
        embedding=embedding,
        dimension=dimension,
        filename=f"Raw/{rep.representation_id}.npz",
    )
    if prior is not None:
        return prior
    path = (
        ROOT
        / "results"
        / "raw"
        / "embeddings"
        / "predictions"
        / model
        / blocks.dataset.key
        / f"seed_{seed}"
        / embedding
        / f"k{dimension}"
        / "Raw"
        / f"{rep.representation_id}.npz"
    )
    return cached_representation_predictions(
        path,
        model=model,
        blocks=blocks,
        rep=rep,
        seed=seed,
        device=device,
        definition={
            "condition": "Raw embedding" if rep.is_reference else "Rotated embedding",
            "embedding": embedding,
            "dimension": int(dimension),
            "rotation_member": int(rep.member),
            "interface_location": "between_numerical_embedding_and_backbone",
        },
    )


def gram_prediction(
    *, blocks: Any, gram_rep: Any, model: str, seed: int, embedding: str, dimension: int, device: str
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    prior = prior_prediction(
        model=model,
        dataset=blocks.dataset.key,
        seed=seed,
        embedding=embedding,
        dimension=dimension,
        filename="GramAfterEmbedding.npz",
    )
    if prior is not None:
        return prior
    path = (
        ROOT
        / "results"
        / "raw"
        / "embeddings"
        / "predictions"
        / model
        / blocks.dataset.key
        / f"seed_{seed}"
        / embedding
        / f"k{dimension}"
        / "GramAfterEmbedding.npz"
    )
    return cached_representation_predictions(
        path,
        model=model,
        blocks=blocks,
        rep=gram_rep,
        seed=seed,
        device=device,
        definition={
            "condition": "Gram-after-embedding",
            "embedding": embedding,
            "dimension": int(dimension),
            "anchors": 16,
            "anchor_selection": "gram_pivot",
            "normalize": True,
            "interface_location": "between_numerical_embedding_and_backbone",
        },
    )


def run_unit(
    spec: dict[str, Any],
    model: str,
    seed: int,
    embedding: str,
    dimension: int,
    device: str,
    protocol: dict[str, Any],
    stage: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    blocks = load_blocks(spec, protocol)
    dataset = blocks.dataset
    orbit = embedding_orbit(dataset, embedding, dimension, int(protocol["orbit_members"]))
    raw: list[dict[str, np.ndarray]] = []
    raw_metadata: list[dict[str, Any]] = []
    for rep in orbit:
        prediction, metadata = raw_prediction(
            blocks=blocks,
            rep=rep,
            model=model,
            seed=seed,
            embedding=embedding,
            dimension=dimension,
            device=device,
        )
        raw.append(prediction)
        raw_metadata.append(metadata)
    gram_orbit = [gram_interface(rep, dataset.key) for rep in orbit]
    coordinate_errors = []
    for rep in gram_orbit[1:]:
        for split in ("X_train", "X_validation", "X_test"):
            reference_values = np.asarray(getattr(gram_orbit[0], split))
            values = np.asarray(getattr(rep, split))
            coordinate_errors.append(
                float(np.linalg.norm(values - reference_values) / max(np.linalg.norm(reference_values), 1e-12))
            )
    maximum_coordinate_error = max(coordinate_errors)
    if maximum_coordinate_error >= 1e-6:
        raise RuntimeError(
            f"embedding Gram coordinate audit failed: {dataset.key}/{embedding}/k{dimension}"
        )
    gram, gram_metadata = gram_prediction(
        blocks=blocks,
        gram_rep=gram_orbit[0],
        model=model,
        seed=seed,
        embedding=embedding,
        dimension=dimension,
        device=device,
    )
    evidence = guarded_evidence(
        dataset.problem_type,
        dataset.y_validation,
        dataset.y_train,
        raw[0]["validation"],
        gram["validation"],
        alphas=[0.75, 0.5, 0.25, 0.0],
        resamples=1000,
        seed=bd.stable_seed("embedding-GuardedG2", dataset.key, model, seed, embedding, dimension),
    )
    guarded_alpha = select_g2(evidence, tau=0.01, gamma=0.0)
    rotations: list[dict[str, Any]] = []
    methods: list[dict[str, Any]] = []
    for split, target in (("validation", dataset.y_validation), ("test", dataset.y_test)):
        reference_prediction = raw[0][split]
        reference_loss = task_error(dataset.problem_type, target, reference_prediction)
        raw_ds = []
        for index, rep in enumerate(orbit):
            prediction = raw[index][split]
            value = disagreement(dataset.problem_type, target, reference_prediction, prediction)
            if index:
                raw_ds.append(value)
            rotations.append(
                {
                    "stage": stage,
                    "dataset": dataset.key,
                    "problem_type": dataset.problem_type,
                    "model": model,
                    "seed": int(seed),
                    "embedding": embedding,
                    "k": int(dimension),
                    "split": split,
                    "condition": "default" if rep.is_reference else "random_orthogonal",
                    "rotation_member": int(rep.member),
                    "default_task_error": reference_loss,
                    "task_error": task_error(dataset.problem_type, target, prediction),
                    "task_effect": task_error(dataset.problem_type, target, prediction) - reference_loss,
                    "disagreement": value,
                    "fit_seconds": float(raw_metadata[index].get("telemetry", {}).get("fit_seconds", 0.0)),
                    "source": raw_metadata[index].get("source", "guarded_embedding_fit"),
                    "interface_location": "between_numerical_embedding_and_backbone",
                }
            )
        raw_d = float(np.mean(raw_ds))
        for method, alpha in (
            ("Raw embedding", 0.0),
            ("Gram-after-embedding", 1.0),
            ("GuardedGram-G2-after-embedding", guarded_alpha),
        ):
            prediction = mix_predictions(reference_prediction, gram[split], alpha)
            method_disagreements = [
                disagreement(
                    dataset.problem_type,
                    target,
                    prediction,
                    mix_predictions(raw[index][split], gram[split], alpha),
                )
                for index in range(1, len(orbit))
            ]
            method_d = float(np.mean(method_disagreements))
            methods.append(
                {
                    "stage": stage,
                    "dataset": dataset.key,
                    "problem_type": dataset.problem_type,
                    "model": model,
                    "seed": int(seed),
                    "embedding": embedding,
                    "k": int(dimension),
                    "split": split,
                    "method": method,
                    "alpha": float(alpha),
                    "raw_disagreement": raw_d,
                    "method_disagreement": method_d,
                    "disagreement_reduction": 0.0 if raw_d <= 1e-12 else 1.0 - method_d / raw_d,
                    "coordinate_error": maximum_coordinate_error,
                    "fit_seconds": 0.0 if method == "Raw embedding" else float(gram_metadata.get("telemetry", {}).get("fit_seconds", 0.0)),
                    **normalized_excess_risk(
                        dataset.problem_type,
                        target,
                        reference_prediction,
                        prediction,
                        dataset.y_train,
                        epsilon=1e-8,
                    ),
                }
            )
    audits = [
        {
            "stage": stage,
            "dataset": dataset.key,
            "model": model,
            "seed": int(seed),
            "embedding": embedding,
            "k": int(dimension),
            "maximum_coordinate_error": maximum_coordinate_error,
            "passes_1e_minus_6": maximum_coordinate_error < 1e-6,
        }
    ]
    path = (
        ROOT
        / "results"
        / "raw"
        / "embeddings"
        / stage
        / model
        / dataset.key
        / f"seed_{seed}"
        / embedding
        / f"k{dimension}.json"
    )
    write_json(
        path,
        {
            "status": "COMPLETE",
            "stage": stage,
            "dataset": dataset.key,
            "model": model,
            "seed": int(seed),
            "embedding": embedding,
            "k": int(dimension),
            "selection_split": "validation_only",
            "test_outcomes_used_for_selection": False,
            "prospective_outcomes_accessed": False,
            "interface_location": "between_numerical_embedding_and_backbone",
            "guarded_g2_evidence": strip_samples(evidence),
            "guarded_g2_alpha": guarded_alpha,
            "rotations": rotations,
            "methods": methods,
            "audits": audits,
        },
    )
    print(
        f"[embedding] {dataset.key} {model} seed={seed} {embedding} k={dimension} "
        f"D={methods[0]['raw_disagreement']:.6g} coord={maximum_coordinate_error:.3g}",
        flush=True,
    )
    return rotations, methods, audits


def collect(
    stage: str,
    specs: list[dict[str, Any]],
    models: list[str],
    seeds: list[int],
    embeddings: list[str],
    dimensions: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rotations: list[dict[str, Any]] = []
    methods: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for spec in specs:
        for model in models:
            for seed in seeds:
                for embedding in embeddings:
                    for dimension in dimensions:
                        path = (
                            ROOT / "results" / "raw" / "embeddings" / stage / model / spec["key"]
                            / f"seed_{seed}" / embedding / f"k{dimension}.json"
                        )
                        if not path.exists():
                            raise FileNotFoundError(f"missing embedding unit: {path}")
                        payload = json.loads(path.read_text())
                        rotations.extend(payload["rotations"])
                        methods.extend(payload["methods"])
                        audits.extend(payload["audits"])
    return rotations, methods, audits


def headroom_table(rotations: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, frame in rotations.groupby(["dataset", "problem_type", "model", "seed", "embedding", "k"]):
        validation = frame[frame.split == "validation"].set_index("rotation_member")
        test = frame[frame.split == "test"].set_index("rotation_member")
        random_test = test[test.index >= 0]
        selected_member = int(validation.task_error.idxmin())
        default_error = float(test.loc[-1].task_error)
        rows.append(
            {
                "dataset": keys[0],
                "problem_type": keys[1],
                "model": keys[2],
                "seed": keys[3],
                "embedding": keys[4],
                "k": keys[5],
                "default_task_error": default_error,
                "mean_random_basis_error": float(random_test.task_error.mean()),
                "best_random_basis_error": float(random_test.task_error.min()),
                "worst_random_basis_error": float(random_test.task_error.max()),
                "validation_selected_member": selected_member,
                "validation_selected_basis_error": float(test.loc[selected_member].task_error),
                "oracle_best_test_member": int(test.task_error.idxmin()),
                "oracle_best_test_error": float(test.task_error.min()),
                "default_is_best_test_basis": bool(int(test.task_error.idxmin()) == -1),
                "default_minus_best_random": default_error - float(random_test.task_error.min()),
                "validation_selected_minus_default": float(test.loc[selected_member].task_error) - default_error,
                "raw_disagreement": float(random_test.disagreement.mean()),
            }
        )
    return pd.DataFrame(rows)


def scaling_table(headroom: pd.DataFrame) -> pd.DataFrame:
    rows = []
    averaged = (
        headroom.groupby(["dataset", "problem_type", "model", "embedding", "k"], as_index=False)
        .raw_disagreement.median()
    )
    for keys, frame in averaged.groupby(["dataset", "problem_type", "model", "embedding"]):
        ordered = frame.sort_values("k")
        slope, intercept = np.polyfit(np.log2(ordered.k.to_numpy(float)), ordered.raw_disagreement.to_numpy(float), 1)
        fitted = intercept + slope * np.log2(ordered.k.to_numpy(float))
        residual = ordered.raw_disagreement.to_numpy(float) - fitted
        total = ordered.raw_disagreement.to_numpy(float) - ordered.raw_disagreement.mean()
        r2 = 1.0 - float(np.sum(residual**2) / max(np.sum(total**2), 1e-12))
        rows.append(
            {
                "dataset": keys[0],
                "problem_type": keys[1],
                "model": keys[2],
                "embedding": keys[3],
                "intercept_a": float(intercept),
                "log2_dimension_slope_b": float(slope),
                "r2": r2,
                "dimensions": json.dumps(ordered.k.astype(int).tolist()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["dimension", "full"], required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dataset")
    parser.add_argument("--model", choices=BACKBONES)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--embedding", choices=["PLE", "RBF"])
    parser.add_argument("--dimension", type=int)
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    protocol = load_protocol()
    specs = development_specs(protocol)
    if args.stage == "dimension":
        names = set(protocol["embedding_confirmation"]["dimension_datasets"])
        specs = [spec for spec in specs if spec["key"] in names]
        dimensions = [int(value) for value in protocol["embedding_confirmation"]["dimension_grid"]]
    else:
        dimensions = [int(value) for value in protocol["embedding_confirmation"]["full_dimensions"]]
    specs = [spec for spec in specs if args.dataset is None or spec["key"] == args.dataset]
    dimensions = [value for value in dimensions if args.dimension is None or value == args.dimension]
    models = [args.model] if args.model else list(BACKBONES)
    seeds = [args.seed] if args.seed is not None else [int(value) for value in protocol["development_seeds"]]
    embeddings = [args.embedding] if args.embedding else list(protocol["embedding_confirmation"]["types"])
    if args.aggregate_only:
        rotations, methods, audits = collect(
            args.stage, specs, models, seeds, embeddings, dimensions
        )
    else:
        rotations, methods, audits = [], [], []
        for spec in specs:
            for model in models:
                for seed in seeds:
                    for embedding in embeddings:
                        for dimension in dimensions:
                            unit = run_unit(
                                spec, model, seed, embedding, dimension, args.device, protocol, args.stage
                            )
                            rotations.extend(unit[0])
                            methods.extend(unit[1])
                            audits.extend(unit[2])
    processed = ROOT / "results" / "processed"
    filtered = any(
        value is not None
        for value in (args.dataset, args.model, args.seed, args.embedding, args.dimension)
    )
    suffix = (
        f"__{args.dataset or 'all'}__{args.model or 'all'}__"
        f"{args.seed if args.seed is not None else 'all'}__{args.embedding or 'all'}__"
        f"{args.dimension if args.dimension is not None else 'all'}"
        if filtered else ""
    )
    prefix = f"embedding_{args.stage}"
    rotation_frame = pd.DataFrame(rotations)
    method_frame = pd.DataFrame(methods)
    audit_frame = pd.DataFrame(audits)
    rotation_frame.to_csv(processed / f"{prefix}_rotation_cells{suffix}.csv", index=False)
    method_frame.to_csv(processed / f"{prefix}_method_cells{suffix}.csv", index=False)
    audit_frame.to_csv(processed / f"{prefix}_coordinate_audits{suffix}.csv", index=False)
    if not filtered:
        headroom = headroom_table(rotation_frame)
        headroom.to_csv(processed / f"{prefix}_headroom.csv", index=False)
        if args.stage == "dimension":
            scaling_table(headroom).to_csv(processed / "embedding_dimension_scaling.csv", index=False)
        method_units = (
            method_frame[method_frame.split == "test"]
            .groupby(["dataset", "problem_type", "model", "embedding", "k", "method"], as_index=False)
            .median(numeric_only=True)
        )
        method_units.to_csv(processed / f"{prefix}_method_units.csv", index=False)
        write_json(
            processed / f"{prefix}_manifest.json",
            {
                "status": "COMPLETE",
                "stage": args.stage,
                "rotation_cells": len(rotation_frame),
                "method_cells": len(method_frame),
                "coordinate_audits": len(audit_frame),
                "headroom_units": len(headroom),
                "datasets": len(specs),
                "models": len(models),
                "seeds": len(seeds),
                "embedding_types": embeddings,
                "dimensions": dimensions,
                "interface_location": "between_numerical_embedding_and_backbone",
                "prospective_outcomes_accessed": False,
            },
        )


if __name__ == "__main__":
    main()
