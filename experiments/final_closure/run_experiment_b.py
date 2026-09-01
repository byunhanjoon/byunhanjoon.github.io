"""Restartable realistic-scale and convergence experiment (Experiment B)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

import closure_core as core
from closure_designs import trajectory_strength3


def budget_label(budget: int | str) -> str:
    return "convergence" if budget == "convergence" else f"epochs{int(budget)}"


def condition_rows(
    data: core.completion.Prepared, design: dict[str, list[Any]], full_product: bool
) -> np.ndarray:
    cards = core.action_cards(data, design)
    if full_product:
        return np.asarray(
            [(*schema, seed) for schema in core.schema_actions(data, design) for seed in range(8)],
            dtype=np.int16,
        )
    return trajectory_strength3((*cards, 8))


def train_diagnostic(
    *, data: core.completion.Prepared, design: dict[str, list[Any]], row: np.ndarray,
    model_name: str, master_seed: int, config: dict[str, Any], device: torch.device,
    budget: int | str,
) -> tuple[np.ndarray, np.ndarray, float, int, list[dict[str, float]], int, int]:
    fi, ci, li, _ = [int(value) for value in row]
    class_map = design["class"][li]
    rendered = {
        part: core.completion.render(
            data, part, design["feature"][fi], design["category"][ci]
        )[0]
        for part in core.completion.PARTS
    }
    output_width = 2 if data.task == "classification" else 1
    subseeds = core.derive_subseeds(master_seed)
    model = core.initialize_model(
        model_name, rendered["train"].shape[1], output_width,
        subseeds["initialization"], config, device,
    )
    transformed_train_y = (
        class_map[data.y["train"]] if data.task == "classification" else data.y["train"]
    )
    transformed_validation_y = (
        class_map[data.y["validation"]]
        if data.task == "classification"
        else data.y["validation"]
    )
    elapsed, peak, trajectory, best_epoch, stopped_epoch = core.fit_diagnostic(
        model, rendered["train"], transformed_train_y,
        rendered["validation"], transformed_validation_y,
        data.task, model_name, subseeds, config["training"], device, budget,
    )
    validation = core.completion.predict(
        model, rendered["validation"], data.task, model_name, class_map, device
    )
    test = core.completion.predict(
        model, rendered["test"], data.task, model_name, class_map, device
    )
    core.validate_probabilities(validation, data.task)
    core.validate_probabilities(test, data.task)
    return validation, test, elapsed, peak, trajectory, best_epoch, stopped_epoch


def a_reuse_arrays(
    dataset: str, split_seed: int, model_name: str, data: core.completion.Prepared,
    rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    source = core.RAW / "experiment_a" / f"{dataset}__{model_name}__split{split_seed}"
    manifest_path = source / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete" or manifest["rows"]["train"] != len(data.y["train"]):
        return None
    if not np.array_equal(np.load(source / "validation_y.npy"), data.y["validation"]):
        return None
    if not np.array_equal(np.load(source / "test_y.npy"), data.y["test"]):
        return None
    actions = np.load(source / "schema_actions.npy")
    lookup = {tuple(int(value) for value in action): index for index, action in enumerate(actions)}
    joint_validation = np.load(source / "joint_validation.npy", mmap_mode="r")
    joint_test = np.load(source / "joint_test.npy", mmap_mode="r")
    joint_master = np.load(source / "joint_master_seeds.npy")
    validation = np.empty((len(rows), *joint_validation.shape[2:]), dtype=np.float32)
    test = np.empty((len(rows), *joint_test.shape[2:]), dtype=np.float32)
    masters = np.empty(len(rows), dtype=np.int64)
    for index, row in enumerate(rows):
        action_index = lookup[tuple(int(value) for value in row[:3])]
        seed_level = int(row[3])
        validation[index] = joint_validation[action_index, seed_level]
        test[index] = joint_test[action_index, seed_level]
        masters[index] = joint_master[action_index, seed_level]
    return validation, test, masters


def run_condition(
    *, dataset: str, split_seed: int, model_name: str,
    size_label: str, data: core.completion.Prepared, smallest: str, largest: str,
    budget: int | str, device: torch.device, config: dict[str, Any],
) -> dict[str, Any]:
    label = budget_label(budget)
    full_product = size_label in {smallest, largest} and budget in {20, "convergence"}
    design = core.completion.views(data, config)
    rows = condition_rows(data, design, full_product)
    stem = f"{dataset}__{model_name}__n{size_label}__{label}"
    output = core.RAW / "experiment_b" / stem
    output.mkdir(parents=True, exist_ok=True)
    output_width = 2 if data.task == "classification" else 1
    validation_array = core.open_memmap(
        output / "validation_predictions.npy",
        (len(rows), len(data.y["validation"]), output_width), np.float32,
    )
    test_array = core.open_memmap(
        output / "test_predictions.npy",
        (len(rows), len(data.y["test"]), output_width), np.float32,
    )
    complete = core.open_memmap(output / "complete.npy", (len(rows),), np.bool_)
    canonical_count = 64
    canonical_validation = core.open_memmap(
        output / "canonical_validation_predictions.npy",
        (canonical_count, len(data.y["validation"]), output_width), np.float32,
    )
    canonical_test = core.open_memmap(
        output / "canonical_test_predictions.npy",
        (canonical_count, len(data.y["test"]), output_width), np.float32,
    )
    canonical_complete = core.open_memmap(
        output / "canonical_complete.npy", (canonical_count,), np.bool_
    )
    np.save(output / "design_rows.npy", rows)
    np.save(output / "validation_y.npy", data.y["validation"])
    np.save(output / "test_y.npy", data.y["test"])

    masters = np.asarray(
        [
            core.stable_seed(
                "B", dataset, split_seed, model_name, size_label, label,
                index, *[int(value) for value in row]
            )
            for index, row in enumerate(rows)
        ],
        dtype=np.int64,
    )
    if len(np.unique(masters)) != len(masters):
        raise AssertionError("Experiment-B master seed collision")
    canonical_masters = np.asarray(
        [
            core.stable_seed(
                "B", dataset, split_seed, model_name, size_label, label,
                "canonical", index,
            )
            for index in range(canonical_count)
        ],
        dtype=np.int64,
    )
    if len(np.unique(np.concatenate((masters, canonical_masters)))) != len(masters) + canonical_count:
        raise AssertionError("Experiment-B joint/canonical seed collision")
    reused = None
    if size_label == smallest and budget == 20 and full_product:
        reused = a_reuse_arrays(dataset, split_seed, model_name, data, rows)
        if reused is not None:
            reuse_validation, reuse_test, masters = reused
            validation_array[:] = reuse_validation
            test_array[:] = reuse_test
            complete[:] = True
            validation_array.flush(); test_array.flush(); complete.flush()
            a_source = core.RAW / "experiment_a" / f"{dataset}__{model_name}__split{split_seed}"
            a_canonical_validation = np.load(a_source / "canonical_validation.npy", mmap_mode="r")
            a_canonical_test = np.load(a_source / "canonical_test.npy", mmap_mode="r")
            a_canonical_masters = np.load(a_source / "canonical_master_seeds.npy")
            canonical_validation[:] = a_canonical_validation[:canonical_count]
            canonical_test[:] = a_canonical_test[:canonical_count]
            canonical_masters = np.asarray(a_canonical_masters[:canonical_count], dtype=np.int64)
            canonical_complete[:] = True
            canonical_validation.flush(); canonical_test.flush(); canonical_complete.flush()
    np.save(output / "master_seeds.npy", masters)
    np.save(output / "canonical_master_seeds.npy", canonical_masters)

    model_hash = core.model_config_hash(
        model_name, {**config["training"], "budget": budget, "training_size": len(data.y["train"])}
    )
    fitted = 0
    diagnostic_replays = 0
    started = time.perf_counter()
    best_epochs = []
    stopped_epochs = []
    wall_total = 0.0
    peak_max = 0
    if reused is not None:
        for index in range(min(8, len(rows))):
            validation, test, elapsed, peak, trajectory, best_epoch, stopped_epoch = train_diagnostic(
                data=data, design=design, row=rows[index], model_name=model_name,
                master_seed=int(masters[index]), config=config, device=device, budget=20,
            )
            validation_gap = float(np.max(np.abs(validation - validation_array[index])))
            test_gap = float(np.max(np.abs(test - test_array[index])))
            if max(validation_gap, test_gap) > 2e-6:
                raise AssertionError(
                    f"A/B diagnostic replay mismatch {validation_gap}/{test_gap}"
                )
            core.save_json_atomic(
                output / "curves" / f"{index:04d}.json",
                {
                    "fit_index": index, "master_seed": int(masters[index]),
                    "schema_action": [int(value) for value in rows[index, :3]],
                    "seed_level": int(rows[index, 3]), "diagnostic_replay": True,
                    "best_epoch": best_epoch, "stopped_epoch": stopped_epoch,
                    "trajectory": trajectory,
                    "validation_prediction_gap": validation_gap,
                    "test_prediction_gap": test_gap,
                },
            )
            diagnostic_replays += 1; wall_total += elapsed; peak_max = max(peak_max, peak)
    canonical_row = np.asarray([0, 0, 0, 0], dtype=np.int16)
    for index in range(canonical_count):
        if bool(canonical_complete[index]):
            try:
                core.validate_probabilities(canonical_test[index], data.task)
                continue
            except AssertionError:
                canonical_complete[index] = False
        validation, test, elapsed, peak, trajectory, best_epoch, stopped_epoch = train_diagnostic(
            data=data, design=design, row=canonical_row, model_name=model_name,
            master_seed=int(canonical_masters[index]), config=config, device=device, budget=budget,
        )
        canonical_validation[index] = validation; canonical_test[index] = test
        canonical_validation.flush(); canonical_test.flush()
        core.save_json_atomic(
            output / "canonical_curves" / f"{index:04d}.json",
            {
                "fit_index": index, "master_seed": int(canonical_masters[index]),
                "best_epoch": best_epoch, "stopped_epoch": stopped_epoch,
                "trajectory": trajectory,
            },
        )
        canonical_complete[index] = True; canonical_complete.flush()
        key = core.fit_key(
            dataset=dataset, split=split_seed, model=model_name,
            model_hash=model_hash, schema_digest=core.schema_hash(dataset, canonical_row[:3]),
            master_seed=int(canonical_masters[index]), training_size=len(data.y["train"]),
            training_budget=label,
        )
        core.register_fit(
            key=key, experiment="B", dataset=dataset, split_seed=split_seed,
            model=model_name, master_seed=int(canonical_masters[index]), artifact=output,
            array_index=f"canonical[{index}]", validation=validation, test=test,
            wall_seconds=elapsed, peak_device_bytes=peak,
        )
        fitted += 1; wall_total += elapsed; peak_max = max(peak_max, peak)
        if fitted % 8 == 0:
            print(f"B {stem}: fitted {fitted} (canonical/design)", flush=True)
    for index, row in enumerate(rows):
        if bool(complete[index]):
            try:
                core.validate_probabilities(test_array[index], data.task)
                curve_path = output / "curves" / f"{index:04d}.json"
                if curve_path.exists():
                    curve = json.loads(curve_path.read_text())
                    best_epochs.append(int(curve["best_epoch"]))
                    stopped_epochs.append(int(curve["stopped_epoch"]))
                continue
            except AssertionError:
                complete[index] = False
        validation, test, elapsed, peak, trajectory, best_epoch, stopped_epoch = train_diagnostic(
            data=data, design=design, row=row, model_name=model_name,
            master_seed=int(masters[index]), config=config, device=device, budget=budget,
        )
        validation_array[index] = validation
        test_array[index] = test
        validation_array.flush(); test_array.flush()
        curve_payload = {
            "fit_index": index,
            "master_seed": int(masters[index]),
            "schema_action": [int(value) for value in row[:3]],
            "seed_level": int(row[3]),
            "best_epoch": best_epoch,
            "stopped_epoch": stopped_epoch,
            "trajectory": trajectory,
        }
        core.save_json_atomic(output / "curves" / f"{index:04d}.json", curve_payload)
        complete[index] = True; complete.flush()
        key = core.fit_key(
            dataset=dataset, split=split_seed, model=model_name,
            model_hash=model_hash, schema_digest=core.schema_hash(dataset, row[:3]),
            master_seed=int(masters[index]), training_size=len(data.y["train"]),
            training_budget=label,
        )
        core.register_fit(
            key=key, experiment="B", dataset=dataset, split_seed=split_seed,
            model=model_name, master_seed=int(masters[index]), artifact=output,
            array_index=f"prediction[{index}]", validation=validation, test=test,
            wall_seconds=elapsed, peak_device_bytes=peak,
        )
        fitted += 1
        wall_total += elapsed
        peak_max = max(peak_max, peak)
        best_epochs.append(best_epoch); stopped_epochs.append(stopped_epoch)
        if fitted % 8 == 0:
            print(f"B {stem}: fitted {fitted}/{len(rows)}", flush=True)
    if not bool(np.asarray(complete).all()) or not bool(np.asarray(canonical_complete).all()):
        raise AssertionError(f"incomplete B condition {stem}")
    core.validate_probabilities(np.asarray(test_array), data.task)
    manifest = {
        "status": "complete", "experiment": "B", "dataset": dataset,
        "task": data.task, "split_seed": split_seed, "model": model_name,
        "training_size_label": size_label, "training_rows": len(data.y["train"]),
        "budget": budget, "full_product": full_product,
        "design_rows": len(rows), "schema_cards": core.action_cards(data, design),
        "unique_master_seeds": len(np.unique(masters)),
        "canonical_reference_fits": canonical_count,
        "represented_fits": len(rows) + canonical_count,
        "independent_seed_records": len(np.unique(np.concatenate((masters, canonical_masters)))),
        "reused_experiment_a": reused is not None,
        "new_fits_this_invocation": fitted,
        "diagnostic_replays_this_invocation": diagnostic_replays,
        "best_epoch_mean": float(np.mean(best_epochs)) if best_epochs else None,
        "stopped_epoch_mean": float(np.mean(stopped_epochs)) if stopped_epochs else None,
        "fit_wall_seconds_this_invocation": wall_total,
        "maximum_peak_device_bytes_this_invocation": peak_max,
        "invocation_wall_seconds": time.perf_counter() - started,
        "model_config_sha256": model_hash,
        "protocol_sha256": core.sha256(core.HERE / "FINAL_CLOSURE_PROTOCOL.md"),
        "config_sha256": core.sha256(core.CONFIG_PATH),
    }
    core.save_json_atomic(output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


def run_dataset_model(dataset: str, model_name: str, device_name: str) -> None:
    if dataset not in core.CONFIG["experiment_b"]["datasets"]:
        raise ValueError("dataset outside frozen Experiment B")
    if model_name not in core.CONFIG["primary_models"]:
        raise ValueError("model outside frozen Experiment B")
    torch.set_num_threads(1)
    split_seed = int(core.CONFIG["experiment_b"]["split_seed"])
    config = core.completion_config()
    prepared, raw_indices = core.b_prepared_datasets(dataset, split_seed, config)
    labels = list(prepared)
    smallest, largest = labels[0], labels[-1]
    nested = [set(raw_indices[label].tolist()) for label in labels]
    if any(not nested[index].issubset(nested[index + 1]) for index in range(len(nested) - 1)):
        raise AssertionError("training subsets are not nested")
    device = torch.device(device_name)
    budgets: list[int | str] = [
        *[int(value) for value in core.CONFIG["experiment_b"]["epoch_budgets"]],
        "convergence",
    ]
    for size_label, data in prepared.items():
        for budget in budgets:
            run_condition(
                dataset=dataset, split_seed=split_seed, model_name=model_name,
                size_label=size_label, data=data, smallest=smallest, largest=largest,
                budget=budget, device=device, config=config,
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run_dataset_model(args.dataset, args.model, args.device)


if __name__ == "__main__":
    main()
