#!/usr/bin/env python3
"""Validate invariant interfaces on local/spectral-hat and one-hot/Helmert pairs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tournament.common import (  # noqa: E402
    bd,
    development_specs,
    disagreement,
    load_blocks,
    load_prior_protocol,
    load_protocol,
    read_prediction_bundle,
    save_prediction_bundle,
    task_error,
)
from tournament.representations import audit_orbit_coordinates, build_interface  # noqa: E402


METHODS = {
    "GramAnchor": (
        "gram_anchor",
        {"anchors": 16, "selection": "gram_pivot", "normalize": True},
    ),
    "GramAnchor-m8": (
        "gram_anchor",
        {"anchors": 8, "selection": "gram_pivot", "normalize": True},
    ),
    "NystromGram": (
        "nystrom_gram",
        {"anchors": 16, "selection": "gram_pivot", "energy": 0.99, "max_rank": 8},
    ),
    "PCA": ("pca", {}),
}


def fit_cached(path: Path, model: str, problem_type: str, rep: Any, blocks: Any, seed: int, device: str, prior):
    if path.exists():
        return read_prediction_bundle(path)
    validation, test, telemetry = bd.fit_predict(
        model,
        problem_type,
        rep,
        blocks.dataset.y_train,
        blocks.dataset.y_validation,
        seed,
        device,
        prior,
    )
    metadata = {"telemetry": telemetry, "representation_id": rep.representation_id}
    save_prediction_bundle(path, validation, test, metadata)
    return validation, test, metadata


def run_unit(spec, model: str, seed: int, method: str, device: str) -> None:
    protocol = load_protocol()
    prior = load_prior_protocol()
    blocks = load_blocks(spec, protocol)
    pairs = [("local_vs_spectral_hat", bd.local_spectral_pair(blocks))]
    helmert = bd.helmert_pair(blocks)
    if helmert is not None:
        pairs.append(("onehot_vs_helmert", helmert))
    interface, parameters = METHODS[method]
    rows = []
    for pair_name, (base, transformed) in pairs:
        reconstruction = float(transformed.metadata.get("reconstruction_error", 0.0))
        if reconstruction >= 1e-6:
            raise RuntimeError(f"natural basis reconstruction failed: {pair_name} {reconstruction}")
        raw_predictions = []
        for rep in (base, transformed):
            path = (
                ROOT
                / "results"
                / "raw"
                / "natural_bases"
                / model
                / blocks.dataset.key
                / f"seed_{seed}"
                / pair_name
                / "Raw"
                / f"{rep.representation_id}.npz"
            )
            raw_predictions.append(
                fit_cached(path, model, blocks.dataset.problem_type, rep, blocks, seed, device, prior)
            )
        mapped = [
            build_interface(rep, interface, blocks.dataset.key, **parameters)
            for rep in (base, transformed)
        ]
        coordinate_audit = audit_orbit_coordinates(mapped[0], [mapped[1]])[0]
        coordinate_error = max(
            coordinate_audit["train_relative_error"],
            coordinate_audit["validation_relative_error"],
            coordinate_audit["test_relative_error"],
        )
        if coordinate_error >= 1e-8:
            raise RuntimeError(f"natural interface audit failed: {pair_name}/{method} {coordinate_error}")
        interface_path = (
            ROOT
            / "results"
            / "raw"
            / "natural_bases"
            / model
            / blocks.dataset.key
            / f"seed_{seed}"
            / pair_name
            / method
            / f"{mapped[0].representation_id}.npz"
        )
        invariant = fit_cached(
            interface_path,
            model,
            blocks.dataset.problem_type,
            mapped[0],
            blocks,
            seed,
            device,
            prior,
        )
        for split, target, index in (
            ("validation", blocks.dataset.y_validation, 0),
            ("test", blocks.dataset.y_test, 1),
        ):
            raw_disagreement = disagreement(
                blocks.dataset.problem_type,
                target,
                raw_predictions[0][index],
                raw_predictions[1][index],
            )
            raw_error = task_error(blocks.dataset.problem_type, target, raw_predictions[0][index])
            method_error = task_error(blocks.dataset.problem_type, target, invariant[index])
            rows.extend(
                [
                    {
                        "dataset": blocks.dataset.key,
                        "problem_type": blocks.dataset.problem_type,
                        "model": model,
                        "seed": seed,
                        "pair": pair_name,
                        "method": "Raw",
                        "split": split,
                        "reconstruction_error": reconstruction,
                        "coordinate_error": 0.0,
                        "disagreement": raw_disagreement,
                        "disagreement_reduction": 0.0,
                        "task_error": raw_error,
                        "relative_task_change": 0.0,
                    },
                    {
                        "dataset": blocks.dataset.key,
                        "problem_type": blocks.dataset.problem_type,
                        "model": model,
                        "seed": seed,
                        "pair": pair_name,
                        "method": method,
                        "split": split,
                        "reconstruction_error": reconstruction,
                        "coordinate_error": coordinate_error,
                        "disagreement": 0.0,
                        "disagreement_reduction": 1.0 if raw_disagreement > 1e-12 else 0.0,
                        "task_error": method_error,
                        "relative_task_change": (method_error - raw_error) / max(abs(raw_error), 1e-12),
                    },
                ]
            )
        print(f"[natural] {model} {blocks.dataset.key} seed={seed} {pair_name} {method}", flush=True)
    destination = ROOT / "results" / "processed" / "natural_bases"
    destination.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        destination / f"{model}__{blocks.dataset.key}__seed_{seed}__{method}.csv", index=False
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["controlled_mlp", "tabm_d", "tabicl_v2", "tabpfn_2_6", "catboost"],
        required=True,
    )
    parser.add_argument("--method", choices=sorted(METHODS), default="GramAnchor")
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--seed", default="all")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    protocol = load_protocol()
    specs = development_specs(protocol)
    if args.dataset != "all":
        specs = [spec for spec in specs if spec["key"] == args.dataset]
    seeds = protocol["model_seeds"] if args.seed == "all" else [int(args.seed)]
    for spec in specs:
        for seed in seeds:
            run_unit(spec, args.model, int(seed), args.method, args.device)


if __name__ == "__main__":
    main()
