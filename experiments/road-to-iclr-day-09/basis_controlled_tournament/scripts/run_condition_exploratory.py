#!/usr/bin/env python3
"""Exploratory condition<=3 evaluation for the frozen finalists."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CONFIRMATION = ROOT.parent / "basis_dependence_confirmation"
sys.path.insert(0, str(ROOT))

from scripts.run_prospective import fit_cached, load_finalists  # noqa: E402
from scripts.run_stage2_representations import prediction_values  # noqa: E402
from tournament.common import (  # noqa: E402
    bd,
    development_specs,
    disagreement,
    load_blocks,
    load_prior_protocol,
    load_protocol,
    protocol_hashes,
    sha256_file,
    task_error,
)
from tournament.representations import audit_orbit_coordinates, build_interface  # noqa: E402


def condition_orbit(blocks: Any, protocol: dict[str, Any]) -> list[Any]:
    reps = bd.build_primary_representations(blocks, int(protocol["orbit_members"]))
    selected = [rep for rep in reps if rep.is_reference or rep.variant == "condition_le_3_all"]
    selected.sort(key=lambda rep: (-int(rep.is_reference), rep.member))
    if len(selected) != int(protocol["orbit_members"]) + 1:
        raise RuntimeError(f"condition<=3 orbit incomplete: {len(selected)}")
    maximum = max(
        float(record["condition_number"])
        for rep in selected[1:]
        for record in rep.metadata.get("equivalence", {}).values()
    )
    if maximum > 3.0 + 1e-8:
        raise RuntimeError(f"condition-number contract failed: {maximum}")
    return selected


def load_prior_raw(model: str, dataset: str, seed: int, orbit: list[Any]) -> dict[str, dict[str, np.ndarray]]:
    bundle = (
        CONFIRMATION / "results" / "raw" / "development" / "replication" / model / dataset / f"seed_{seed}"
    )
    frame = pd.read_csv(bundle / "predictions.csv.gz")
    wanted_ids = {rep.representation_id for rep in orbit}
    frame = frame[frame["representation_id"].isin(wanted_ids)]
    predictions: dict[str, dict[str, np.ndarray]] = {}
    for (representation_id, split), part in frame.groupby(["representation_id", "split"], sort=False):
        predictions.setdefault(str(representation_id), {})[str(split)] = prediction_values(
            part.sort_values("row_id")
        )
    if set(predictions) != wanted_ids:
        raise RuntimeError(f"prior condition<=3 baseline incomplete: {set(predictions) ^ wanted_ids}")
    return predictions


def rows_for_predictions(
    blocks: Any,
    model: str,
    seed: int,
    method: str,
    orbit: list[Any],
    predictions: dict[str, dict[str, np.ndarray]],
    coordinate_error: float,
) -> list[dict[str, Any]]:
    records = []
    reference_id = orbit[0].representation_id
    for rep in orbit:
        for split, target in (("validation", blocks.dataset.y_validation), ("test", blocks.dataset.y_test)):
            prediction = predictions[rep.representation_id][split]
            reference = predictions[reference_id][split]
            records.append(
                {
                    "dataset": blocks.dataset.key,
                    "problem_type": blocks.dataset.problem_type,
                    "model": model,
                    "seed": seed,
                    "method": method,
                    "transform_family": "condition_le_3_all",
                    "representation_id": rep.representation_id,
                    "member": rep.member,
                    "is_reference": rep.is_reference,
                    "split": split,
                    "coordinate_error": coordinate_error,
                    "task_error": task_error(blocks.dataset.problem_type, target, prediction),
                    "disagreement": disagreement(blocks.dataset.problem_type, target, reference, prediction),
                    "fit_seconds": 0.0,
                }
            )
    return records


def run_unit(spec: dict[str, Any], model: str, seed: int, device: str) -> None:
    finalist_config, finalist_hash = load_finalists()
    protocol = load_protocol()
    prior = load_prior_protocol()
    blocks = load_blocks(spec, protocol)
    orbit = condition_orbit(blocks, protocol)
    raw = load_prior_raw(model, blocks.dataset.key, seed, orbit)
    records = rows_for_predictions(blocks, model, seed, "Raw", orbit, raw, 0.0)
    frozen_hashes = {
        **protocol_hashes(),
        "prior_protocol_sha256": sha256_file(CONFIRMATION / "configs" / "development_protocol.yaml"),
        "finalist_configs_sha256": finalist_hash,
    }
    for finalist in finalist_config["finalists"]:
        if model not in finalist["applicable_models"]:
            continue
        method_id = finalist["method_id"]
        if finalist["type"] == "optimizer":
            definition = finalist["per_model"][model]
            predictions = {}
            for rep in orbit:
                path = (
                    ROOT / "results" / "raw" / "condition_exploratory" / model /
                    blocks.dataset.key / f"seed_{seed}" / method_id / f"{rep.representation_id}.npz"
                )
                validation, test, _ = fit_cached(
                    path,
                    model=model,
                    problem_type=blocks.dataset.problem_type,
                    rep=rep,
                    blocks=blocks,
                    seed=seed,
                    device=device,
                    protocol=protocol,
                    prior_protocol=prior,
                    optimizer=definition["optimizer"],
                    initialization=definition["initialization"],
                    optimizer_overrides=definition.get("optimizer_overrides", {}),
                    metadata={"stage": "condition_exploratory", "method": method_id, "frozen_hashes": frozen_hashes},
                )
                predictions[rep.representation_id] = {"validation": validation, "test": test}
            records.extend(rows_for_predictions(blocks, model, seed, method_id, orbit, predictions, 0.0))
            continue

        mapped = [
            build_interface(
                rep, finalist["interface"], blocks.dataset.key, **finalist["interface_parameters"]
            )
            for rep in orbit
        ]
        audits = audit_orbit_coordinates(mapped[0], mapped[1:])
        coordinate_error = max(
            max(audit["train_relative_error"], audit["validation_relative_error"], audit["test_relative_error"])
            for audit in audits
        )
        invariant_predictions = {}
        for rep, mapped_rep in zip(orbit, mapped):
            path = (
                ROOT / "results" / "raw" / "condition_exploratory" / model /
                blocks.dataset.key / f"seed_{seed}" / method_id / f"{mapped_rep.representation_id}.npz"
            )
            validation, test, _ = fit_cached(
                path,
                model=model,
                problem_type=blocks.dataset.problem_type,
                rep=mapped_rep,
                blocks=blocks,
                seed=seed,
                device=device,
                protocol=protocol,
                prior_protocol=prior,
                metadata={"stage": "condition_exploratory", "method": method_id, "frozen_hashes": frozen_hashes},
            )
            invariant_predictions[rep.representation_id] = {"validation": validation, "test": test}
        if finalist["type"] == "hybrid_prediction_mixture":
            alpha = float(finalist["alpha"])
            predictions = {
                rep.representation_id: {
                    split: (1 - alpha) * raw[rep.representation_id][split]
                    + alpha * invariant_predictions[rep.representation_id][split]
                    for split in ("validation", "test")
                }
                for rep in orbit
            }
        else:
            predictions = invariant_predictions
        records.extend(rows_for_predictions(blocks, model, seed, method_id, orbit, predictions, coordinate_error))

    # The predeclared covariance-normalized interface is evaluated alongside
    # interface finalists, with all three frozen ridges and no claim of exact
    # general-linear invariance.
    if any(
        finalist["type"] in {"interface", "hybrid_prediction_mixture"}
        and model in finalist["applicable_models"]
        for finalist in finalist_config["finalists"]
    ):
        for ridge in protocol["representations"]["mahalanobis_gram"]["ridge_values"]:
            method_id = f"MahalanobisGram-lambda{float(ridge):g}"
            mapped = [
                build_interface(
                    rep, "mahalanobis_gram", blocks.dataset.key, anchors=16, ridge=float(ridge)
                )
                for rep in orbit
            ]
            audits = audit_orbit_coordinates(mapped[0], mapped[1:])
            coordinate_error = max(
                max(audit["train_relative_error"], audit["validation_relative_error"], audit["test_relative_error"])
                for audit in audits
            )
            predictions = {}
            for rep, mapped_rep in zip(orbit, mapped):
                path = (
                    ROOT / "results" / "raw" / "condition_exploratory" / model /
                    blocks.dataset.key / f"seed_{seed}" / method_id / f"{mapped_rep.representation_id}.npz"
                )
                validation, test, _ = fit_cached(
                    path,
                    model=model,
                    problem_type=blocks.dataset.problem_type,
                    rep=mapped_rep,
                    blocks=blocks,
                    seed=seed,
                    device=device,
                    protocol=protocol,
                    prior_protocol=prior,
                    metadata={"stage": "condition_exploratory", "method": method_id, "frozen_hashes": frozen_hashes},
                )
                predictions[rep.representation_id] = {"validation": validation, "test": test}
            records.extend(rows_for_predictions(blocks, model, seed, method_id, orbit, predictions, coordinate_error))
    destination = ROOT / "results" / "processed" / "condition_exploratory"
    destination.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(
        destination / f"{model}__{blocks.dataset.key}__seed_{seed}.csv", index=False
    )
    print(f"[condition<=3] {model} {blocks.dataset.key} seed={seed}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["controlled_mlp", "tabm_d", "tabicl_v2", "tabpfn_2_6", "catboost"],
        required=True,
    )
    parser.add_argument("--dataset", default="stage1")
    parser.add_argument("--seed", default="0")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    protocol = load_protocol()
    specs = development_specs(protocol)
    if args.dataset == "stage1":
        specs = [spec for spec in specs if spec["key"] in protocol["stage1_datasets"]]
    elif args.dataset != "all":
        specs = [spec for spec in specs if spec["key"] == args.dataset]
    seeds = protocol["model_seeds"] if args.seed == "all" else [int(args.seed)]
    for spec in specs:
        for seed in seeds:
            run_unit(spec, args.model, int(seed), args.device)


if __name__ == "__main__":
    main()
