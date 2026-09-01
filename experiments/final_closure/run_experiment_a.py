"""Run one restartable Experiment-A neural prediction-pool cell."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

import closure_core as core


def open_seed_expandable_memmap(
    path: Path, shape: tuple[int, ...], dtype: np.dtype,
) -> np.memmap:
    """Expand the seed axis while preserving completed restartable entries."""
    if not path.exists():
        return core.open_memmap(path, shape, dtype)
    current = np.lib.format.open_memmap(path, mode="r")
    if current.shape == shape and current.dtype == np.dtype(dtype):
        del current
        return core.open_memmap(path, shape, dtype)
    compatible = (
        current.dtype == np.dtype(dtype)
        and len(current.shape) == len(shape)
        and current.shape[0] == shape[0]
        and current.shape[2:] == shape[2:]
        and current.shape[1] < shape[1]
    )
    if not compatible:
        raise AssertionError(f"non-expandable resume array at {path}: {current.shape}/{current.dtype}")
    old_seed_count = current.shape[1]
    temporary = path.with_name(f"{path.stem}.seed-expansion.npy")
    expanded = np.lib.format.open_memmap(temporary, mode="w+", dtype=dtype, shape=shape)
    expanded[...] = 0
    slices = [slice(None)] * len(shape); slices[1] = slice(0, old_seed_count)
    expanded[tuple(slices)] = current
    expanded.flush(); del expanded; del current
    os.replace(temporary, path)
    return core.open_memmap(path, shape, dtype)


def run_cell(dataset: str, split_seed: int, model_name: str, device_name: str) -> dict:
    config = core.completion_config()
    aconfig = core.CONFIG["experiment_a"]
    if dataset not in core.CONFIG["all_datasets"]:
        raise ValueError("dataset outside frozen Experiment A")
    if split_seed not in core.CONFIG["split_seeds"]:
        raise ValueError("split outside frozen Experiment A")
    if model_name not in core.CONFIG["primary_models"]:
        raise ValueError("model outside frozen Experiment A")

    torch.set_num_threads(1)
    device = torch.device(device_name)
    data = core.completion.prepare(dataset, split_seed, config)
    design = core.completion.views(data, config)
    actions = core.schema_actions(data, design)
    cards = core.action_cards(data, design)
    minimum_seeds = int(aconfig["joint_master_seeds_per_schema"])
    maximum_budget = max(int(value) for value in aconfig["budgets"])
    # Cached draws are explicitly conditional on a finite trained pool.  Keep
    # a uniform 256-fit joint cache so B=64 IID draws almost never exhaust one
    # action's distinct master seeds, even when nuisance factors collapse.
    seeds_per_schema = max(minimum_seeds, math.ceil(4 * maximum_budget / len(actions)))
    canonical_count = int(aconfig["canonical_master_seeds"])
    output_width = 2 if data.task == "classification" else 1
    stem = f"{dataset}__{model_name}__split{split_seed}"
    output = core.RAW / "experiment_a" / stem
    output.mkdir(parents=True, exist_ok=True)

    joint_validation = open_seed_expandable_memmap(
        output / "joint_validation.npy",
        (len(actions), seeds_per_schema, len(data.y["validation"]), output_width),
        np.float32,
    )
    joint_test = open_seed_expandable_memmap(
        output / "joint_test.npy",
        (len(actions), seeds_per_schema, len(data.y["test"]), output_width),
        np.float32,
    )
    joint_complete = open_seed_expandable_memmap(
        output / "joint_complete.npy", (len(actions), seeds_per_schema), np.bool_
    )
    canonical_validation = core.open_memmap(
        output / "canonical_validation.npy",
        (canonical_count, len(data.y["validation"]), output_width), np.float32,
    )
    canonical_test = core.open_memmap(
        output / "canonical_test.npy",
        (canonical_count, len(data.y["test"]), output_width), np.float32,
    )
    canonical_complete = core.open_memmap(
        output / "canonical_complete.npy", (canonical_count,), np.bool_
    )

    schema_array = np.asarray(actions, dtype=np.int16)
    np.save(output / "schema_actions.npy", schema_array)
    np.save(output / "validation_y.npy", data.y["validation"])
    np.save(output / "test_y.npy", data.y["test"])
    joint_master = np.empty((len(actions), seeds_per_schema), dtype=np.int64)
    for action_index in range(len(actions)):
        for repeat in range(seeds_per_schema):
            joint_master[action_index, repeat] = core.stable_seed(
                "A", dataset, split_seed, model_name, "joint", action_index, repeat
            )
    canonical_master = np.empty(canonical_count, dtype=np.int64)
    canonical_master[:minimum_seeds] = joint_master[0, :minimum_seeds]
    for repeat in range(minimum_seeds, canonical_count):
        canonical_master[repeat] = core.stable_seed(
            "A", dataset, split_seed, model_name, "canonical-extension", repeat
        )
    joint_master[0, :seeds_per_schema] = canonical_master[:seeds_per_schema]
    if len(np.unique(np.concatenate((joint_master.reshape(-1)[seeds_per_schema:], canonical_master)))) != (
        joint_master.size - seeds_per_schema + canonical_count
    ):
        raise AssertionError("independent master seed collision")
    np.save(output / "joint_master_seeds.npy", joint_master)
    np.save(output / "canonical_master_seeds.npy", canonical_master)

    model_hash = core.model_config_hash(model_name, config["training"])
    started = time.perf_counter()
    fitted = 0

    def train_one(action_index: int, repeat: int, master_seed: int) -> tuple[np.ndarray, np.ndarray]:
        nonlocal fitted
        validation, test, elapsed, peak, _ = core.train_predict(
            data=data, design=design, schema_action=actions[action_index],
            model_name=model_name, master_seed=int(master_seed), config=config,
            device=device,
        )
        key = core.fit_key(
            dataset=dataset, split=split_seed, model=model_name,
            model_hash=model_hash, schema_digest=core.schema_hash(dataset, actions[action_index]),
            master_seed=int(master_seed), training_size=len(data.y["train"]),
            training_budget="epochs=20",
        )
        core.register_fit(
            key=key, experiment="A", dataset=dataset, split_seed=split_seed,
            model=model_name, master_seed=int(master_seed), artifact=output,
            array_index=f"joint[{action_index},{repeat}]", validation=validation,
            test=test, wall_seconds=elapsed, peak_device_bytes=peak,
        )
        fitted += 1
        return validation, test

    # Canonical reference first.  The per-action seed prefix is the canonical-
    # schema slice of the joint pool and therefore represents one shared
    # physical fit rather than a duplicate training run.
    for repeat in range(canonical_count):
        if bool(canonical_complete[repeat]):
            try:
                core.validate_probabilities(canonical_test[repeat], data.task)
                if repeat < seeds_per_schema and not bool(joint_complete[0, repeat]):
                    joint_validation[0, repeat] = canonical_validation[repeat]
                    joint_test[0, repeat] = canonical_test[repeat]
                    joint_complete[0, repeat] = True
                    joint_validation.flush(); joint_test.flush(); joint_complete.flush()
                continue
            except AssertionError:
                canonical_complete[repeat] = False
        if repeat < seeds_per_schema and bool(joint_complete[0, repeat]):
            validation = np.asarray(joint_validation[0, repeat])
            test = np.asarray(joint_test[0, repeat])
            core.validate_probabilities(test, data.task)
        else:
            validation, test = train_one(0, repeat, int(canonical_master[repeat]))
            if repeat < seeds_per_schema:
                joint_validation[0, repeat] = validation
                joint_test[0, repeat] = test
                joint_complete[0, repeat] = True
                joint_validation.flush(); joint_test.flush(); joint_complete.flush()
        canonical_validation[repeat] = validation
        canonical_test[repeat] = test
        canonical_complete[repeat] = True
        canonical_validation.flush(); canonical_test.flush(); canonical_complete.flush()
        if fitted and fitted % 16 == 0:
            print(f"A {stem}: fitted {fitted}", flush=True)

    for action_index in range(len(actions)):
        for repeat in range(seeds_per_schema):
            if bool(joint_complete[action_index, repeat]):
                try:
                    core.validate_probabilities(joint_test[action_index, repeat], data.task)
                    continue
                except AssertionError:
                    joint_complete[action_index, repeat] = False
            validation, test = train_one(
                action_index, repeat, int(joint_master[action_index, repeat])
            )
            joint_validation[action_index, repeat] = validation
            joint_test[action_index, repeat] = test
            joint_complete[action_index, repeat] = True
            joint_validation.flush(); joint_test.flush(); joint_complete.flush()
            if fitted % 16 == 0:
                print(f"A {stem}: fitted {fitted}", flush=True)

    if not bool(np.asarray(joint_complete).all()) or not bool(np.asarray(canonical_complete).all()):
        raise AssertionError("Experiment-A cell did not close")
    core.validate_probabilities(np.asarray(joint_test), data.task)
    core.validate_probabilities(np.asarray(canonical_test), data.task)
    manifest = {
        "status": "complete",
        "experiment": "A",
        "dataset": dataset,
        "task": data.task,
        "split_seed": split_seed,
        "model": model_name,
        "schema_cards": cards,
        "schema_actions": len(actions),
        "joint_master_seeds_per_schema": seeds_per_schema,
        "joint_pool_fits": len(actions) * seeds_per_schema,
        "canonical_reference_fits": canonical_count,
        "shared_canonical_joint_fits": seeds_per_schema,
        "unique_represented_fits": len(actions) * seeds_per_schema + canonical_count - seeds_per_schema,
        "new_fits_this_invocation": fitted,
        "rows": {part: len(data.y[part]) for part in core.completion.PARTS},
        "model_config_sha256": model_hash,
        "protocol_sha256": core.sha256(core.HERE / "FINAL_CLOSURE_PROTOCOL.md"),
        "config_sha256": core.sha256(core.CONFIG_PATH),
        "wall_seconds_this_invocation": time.perf_counter() - started,
        "device": device_name,
    }
    core.save_json_atomic(output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split-seed", required=True, type=int)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run_cell(args.dataset, args.split_seed, args.model, args.device)


if __name__ == "__main__":
    main()
