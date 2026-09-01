#!/usr/bin/env python3
"""Evaluate frozen finalists on the locked prospective panel without tuning."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tournament.common import (  # noqa: E402
    bd,
    disagreement,
    environment_metadata,
    load_blocks,
    load_json,
    load_prior_protocol,
    load_protocol,
    orthogonal_all_orbit,
    prospective_specs,
    protocol_hashes,
    read_prediction_bundle,
    save_prediction_bundle,
    sha256_file,
    task_error,
    write_json,
)
from tournament.models import fit_model  # noqa: E402
from tournament.representations import audit_orbit_coordinates, build_interface  # noqa: E402


FINALIST_PATH = ROOT / "configs" / "FINALIST_CONFIGS.json"
FINALIST_SHA_PATH = ROOT / "configs" / "FINALIST_CONFIGS.sha256"


def load_finalists() -> tuple[dict[str, Any], str]:
    if not FINALIST_PATH.exists() or not FINALIST_SHA_PATH.exists():
        raise RuntimeError("prospective access refused: frozen finalist config and SHA are required")
    expected = FINALIST_SHA_PATH.read_text().strip().split()[0]
    actual = sha256_file(FINALIST_PATH)
    if expected != actual:
        raise RuntimeError("prospective access refused: FINALIST_CONFIGS SHA mismatch")
    config = load_json(FINALIST_PATH)
    finalists = config.get("finalists", [])
    if config.get("status") != "FROZEN_BEFORE_PROSPECTIVE_DATA_ACCESS":
        raise RuntimeError("prospective access refused: finalist status is not frozen")
    if not 1 <= len(finalists) <= 3:
        raise RuntimeError(f"prospective access refused: expected 1--3 finalists, got {len(finalists)}")
    if config.get("prospective_panel_sha256") != protocol_hashes()["new_prospective_panel_sha256"]:
        raise RuntimeError("prospective access refused: prospective panel hash drift")
    return config, actual


def fit_cached(
    path: Path,
    *,
    model: str,
    problem_type: str,
    rep: Any,
    blocks: Any,
    seed: int,
    device: str,
    protocol: dict[str, Any],
    prior_protocol: dict[str, Any],
    optimizer: str = "adamw",
    initialization: str = "default",
    optimizer_overrides: dict[str, Any] | None = None,
    metadata: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if path.exists() and path.with_suffix(".json").exists():
        validation, test, stored = read_prediction_bundle(path)
        if stored.get("frozen_hashes") != metadata["frozen_hashes"]:
            raise RuntimeError(f"frozen-config drift in cached prospective artifact {path}")
        return validation, test, stored
    started = time.perf_counter()
    if model in {"controlled_mlp", "tabm_d"}:
        result = fit_model(
            model,
            problem_type,
            rep,
            blocks.dataset.y_train,
            blocks.dataset.y_validation,
            seed,
            device,
            protocol,
            optimizer_method=optimizer,
            optimizer_overrides=optimizer_overrides,
            initialization=initialization,
        )
        validation, test, telemetry = result.validation, result.test, result.telemetry
    else:
        validation, test, telemetry = bd.fit_predict(
            model,
            problem_type,
            rep,
            blocks.dataset.y_train,
            blocks.dataset.y_validation,
            seed,
            device,
            prior_protocol,
        )
    stored = {**metadata, "telemetry": telemetry, "wall_seconds": time.perf_counter() - started}
    save_prediction_bundle(path, validation, test, stored)
    return np.asarray(validation), np.asarray(test), stored


def add_method_rows(
    rows: list[dict[str, Any]],
    *,
    blocks: Any,
    model: str,
    seed: int,
    method: str,
    method_type: str,
    orbit: list[Any],
    predictions: dict[str, dict[str, np.ndarray]],
    fit_seconds: dict[str, float],
) -> None:
    reference_id = orbit[0].representation_id
    for rep in orbit:
        for split, target in (("validation", blocks.dataset.y_validation), ("test", blocks.dataset.y_test)):
            prediction = predictions[rep.representation_id][split]
            reference = predictions[reference_id][split]
            rows.append(
                {
                    "panel": "NEW_prospective",
                    "dataset": blocks.dataset.key,
                    "problem_type": blocks.dataset.problem_type,
                    "model": model,
                    "seed": seed,
                    "method": method,
                    "track": method_type,
                    "representation_id": rep.representation_id,
                    "member": rep.member,
                    "is_reference": rep.is_reference,
                    "split": split,
                    "task_error": task_error(blocks.dataset.problem_type, target, prediction),
                    "disagreement": disagreement(blocks.dataset.problem_type, target, reference, prediction),
                    "fit_seconds": float(fit_seconds.get(rep.representation_id, 0.0)),
                }
            )


def run_unit(
    spec: dict[str, Any],
    model: str,
    seed: int,
    device: str,
    config: dict[str, Any],
    finalist_hash: str,
) -> None:
    protocol = load_protocol()
    prior_protocol = load_prior_protocol()
    blocks = load_blocks(spec, protocol)
    orbit = orthogonal_all_orbit(blocks, protocol)
    frozen_hashes = {**protocol_hashes(), "finalist_configs_sha256": finalist_hash}
    unit_root = ROOT / "results" / "raw" / "prospective" / model / blocks.dataset.key / f"seed_{seed}"

    baseline_definition = config["baselines"][model]
    baseline_predictions: dict[str, dict[str, np.ndarray]] = {}
    baseline_fit: dict[str, float] = {}
    for rep in orbit:
        path = unit_root / "Raw" / f"{rep.representation_id}.npz"
        validation, test, metadata = fit_cached(
            path,
            model=model,
            problem_type=blocks.dataset.problem_type,
            rep=rep,
            blocks=blocks,
            seed=seed,
            device=device,
            protocol=protocol,
            prior_protocol=prior_protocol,
            optimizer=baseline_definition.get("optimizer", "adamw"),
            initialization=baseline_definition.get("initialization", "default"),
            optimizer_overrides=baseline_definition.get("optimizer_overrides", {}),
            metadata={
                "stage": "NEW_prospective",
                "method": "Raw",
                "definition": baseline_definition,
                "frozen_hashes": frozen_hashes,
            },
        )
        baseline_predictions[rep.representation_id] = {"validation": validation, "test": test}
        baseline_fit[rep.representation_id] = float(metadata["telemetry"]["fit_seconds"])
    rows: list[dict[str, Any]] = []
    add_method_rows(
        rows,
        blocks=blocks,
        model=model,
        seed=seed,
        method="Raw",
        method_type="baseline",
        orbit=orbit,
        predictions=baseline_predictions,
        fit_seconds=baseline_fit,
    )

    for finalist in config["finalists"]:
        if model not in finalist["applicable_models"]:
            continue
        method_id = finalist["method_id"]
        method_type = finalist["type"]
        if method_type == "optimizer":
            predictions: dict[str, dict[str, np.ndarray]] = {}
            fits: dict[str, float] = {}
            definition = finalist["per_model"][model]
            for rep in orbit:
                path = unit_root / method_id / f"{rep.representation_id}.npz"
                validation, test, metadata = fit_cached(
                    path,
                    model=model,
                    problem_type=blocks.dataset.problem_type,
                    rep=rep,
                    blocks=blocks,
                    seed=seed,
                    device=device,
                    protocol=protocol,
                    prior_protocol=prior_protocol,
                    optimizer=definition["optimizer"],
                    initialization=definition["initialization"],
                    optimizer_overrides=definition.get("optimizer_overrides", {}),
                    metadata={
                        "stage": "NEW_prospective",
                        "method": method_id,
                        "definition": definition,
                        "frozen_hashes": frozen_hashes,
                    },
                )
                predictions[rep.representation_id] = {"validation": validation, "test": test}
                fits[rep.representation_id] = float(metadata["telemetry"]["fit_seconds"])
            add_method_rows(
                rows,
                blocks=blocks,
                model=model,
                seed=seed,
                method=method_id,
                method_type=method_type,
                orbit=orbit,
                predictions=predictions,
                fit_seconds=fits,
            )
            continue

        interface = finalist["interface"]
        parameters = finalist["interface_parameters"]
        mapped = [build_interface(rep, interface, blocks.dataset.key, **parameters) for rep in orbit]
        audits = audit_orbit_coordinates(mapped[0], mapped[1:])
        maximum_coordinate_error = max(
            max(audit["train_relative_error"], audit["validation_relative_error"], audit["test_relative_error"])
            for audit in audits
        )
        if maximum_coordinate_error >= 1e-8:
            raise RuntimeError(f"prospective interface audit failed {method_id}: {maximum_coordinate_error}")
        path = unit_root / method_id / f"{mapped[0].representation_id}.npz"
        validation, test, metadata = fit_cached(
            path,
            model=model,
            problem_type=blocks.dataset.problem_type,
            rep=mapped[0],
            blocks=blocks,
            seed=seed,
            device=device,
            protocol=protocol,
            prior_protocol=prior_protocol,
            metadata={
                "stage": "NEW_prospective",
                "method": method_id,
                "definition": finalist,
                "maximum_coordinate_error": maximum_coordinate_error,
                "frozen_hashes": frozen_hashes,
            },
        )
        invariant = {"validation": validation, "test": test}
        if method_type == "interface":
            predictions = {
                rep.representation_id: invariant for rep in mapped
            }
            output_orbit = mapped
        elif method_type == "hybrid_prediction_mixture":
            alpha = float(finalist["alpha"])
            predictions = {
                rep.representation_id: {
                    split: (1.0 - alpha) * baseline_predictions[rep.representation_id][split]
                    + alpha * invariant[split]
                    for split in ("validation", "test")
                }
                for rep in orbit
            }
            output_orbit = orbit
        else:
            raise RuntimeError(f"unknown finalist type {method_type}")
        add_method_rows(
            rows,
            blocks=blocks,
            model=model,
            seed=seed,
            method=method_id,
            method_type=method_type,
            orbit=output_orbit,
            predictions=predictions,
            fit_seconds={output_orbit[0].representation_id: float(metadata["telemetry"]["fit_seconds"])},
        )
        print(f"[prospective] {model} {blocks.dataset.key} seed={seed} {method_id}", flush=True)

    destination = ROOT / "results" / "processed" / "prospective_cells"
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / f"{model}__{blocks.dataset.key}__seed_{seed}.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    write_json(
        output.with_suffix(".json"),
        {
            "panel": "NEW_prospective",
            "dataset": blocks.dataset.key,
            "model": model,
            "seed": seed,
            "frozen_hashes": frozen_hashes,
            "environment": environment_metadata(),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["controlled_mlp", "tabm_d", "tabicl_v2", "tabpfn_2_6", "catboost"],
        required=True,
    )
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--seed", default="all")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config, finalist_hash = load_finalists()
    # This is intentionally the first call that resolves/loads prospective
    # datasets, and it occurs only after the finalist hash gate above.
    specs = prospective_specs()
    if args.dataset != "all":
        specs = [spec for spec in specs if spec["key"] == args.dataset]
        if not specs:
            raise RuntimeError(f"unknown prospective dataset {args.dataset}")
    seeds = load_protocol()["model_seeds"] if args.seed == "all" else [int(args.seed)]
    for spec in specs:
        for seed in seeds:
            run_unit(spec, args.model, int(seed), args.device, config, finalist_hash)


if __name__ == "__main__":
    main()
