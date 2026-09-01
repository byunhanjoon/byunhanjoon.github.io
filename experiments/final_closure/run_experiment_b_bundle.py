"""Checkpointed, restartable Experiment-B dataset×model trajectory runner."""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import time
from typing import Any

import numpy as np
import torch

import closure_core as core
from closure_designs import trajectory_strength3


BUDGETS: tuple[int | str, ...] = (20, 50, 100, 200, "convergence")
CANONICAL_COUNT = 64


def label(budget: int | str) -> str:
    return "convergence" if budget == "convergence" else f"epochs{budget}"


def rows_for_condition(data, design, full: bool) -> np.ndarray:
    cards = core.action_cards(data, design)
    if full:
        return np.asarray(
            [(*schema, seed) for schema in core.schema_actions(data, design) for seed in range(8)],
            dtype=np.int16,
        )
    return trajectory_strength3((*cards, 8))


def occurrence_keys(rows: np.ndarray) -> list[tuple[tuple[int, ...], int]]:
    """Distinguish replicated OA rows while retaining their factor levels."""
    counts: dict[tuple[int, ...], int] = {}
    keys = []
    for row in rows:
        levels = tuple(int(value) for value in row)
        occurrence = counts.get(levels, 0)
        keys.append((levels, occurrence))
        counts[levels] = occurrence + 1
    return keys


def render_action(data, design, row):
    fi, ci, li = [int(value) for value in row[:3]]
    class_map = design["class"][li]
    rendered = {
        part: core.completion.render(
            data, part, design["feature"][fi], design["category"][ci]
        )[0]
        for part in core.completion.PARTS
    }
    train_y = class_map[data.y["train"]] if data.task == "classification" else data.y["train"]
    validation_y = (
        class_map[data.y["validation"]]
        if data.task == "classification" else data.y["validation"]
    )
    return rendered, class_map, train_y, validation_y


def train_checkpoints(
    *, data, design, row, model_name: str, master_seed: int,
    config: dict[str, Any], device: torch.device,
    required: set[int | str],
    initial_state: dict[str, torch.Tensor] | None = None,
    minimum_epochs: int = 200,
) -> tuple[dict[int | str, tuple[np.ndarray, np.ndarray]], list[dict[str, float]], int, int, float, int]:
    rendered, class_map, train_y, validation_y = render_action(data, design, row)
    output_width = 2 if data.task == "classification" else 1
    subseeds = core.derive_subseeds(master_seed)
    model = core.initialize_model(
        model_name, rendered["train"].shape[1], output_width,
        subseeds["initialization"], config, device,
    )
    if initial_state is not None:
        model.load_state_dict(initial_state)
    training = config["training"]
    convergence = core.CONFIG["experiment_b"]["convergence"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    order_rng = np.random.default_rng(int(subseeds["dataloader"]))
    torch.manual_seed(int(subseeds["dropout"]))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(subseeds["dropout"]))
        torch.cuda.reset_peak_memory_stats(device)
    batch = int(training["batch_size"])
    patience = int(convergence["patience"])
    relative_minimum = float(convergence["relative_minimum_improvement"])
    maximum = int(convergence["maximum_epochs"])
    best_loss = float("inf"); best_epoch = 0; best_state = None; stale = 0
    convergence_epoch = None; convergence_best_epoch = None
    trajectory = []
    checkpoint_states: dict[int | str, dict[str, torch.Tensor]] = {}
    started = time.perf_counter()
    for epoch in range(1, maximum + 1):
        model.train(); order = order_rng.permutation(len(rendered["train"]))
        total_loss = 0.0; total_grad_squared = 0.0; batches = 0
        before = [parameter.detach().clone() for parameter in model.parameters()]
        for start in range(0, len(order), batch):
            chosen = order[start : start + batch]
            xb = torch.from_numpy(rendered["train"][chosen]).to(device)
            if data.task == "classification":
                yb = torch.from_numpy(train_y[chosen].astype(np.int64)).to(device)
            else:
                yb = torch.from_numpy(train_y[chosen].astype(np.float32)).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = core.completion.loss_value(
                core.completion.forward(model, xb, model_name), yb, data.task, model_name
            )
            loss.backward()
            grad_squared = sum(
                float(torch.sum(parameter.grad.detach() ** 2).cpu())
                for parameter in model.parameters() if parameter.grad is not None
            )
            total_grad_squared += grad_squared
            optimizer.step(); total_loss += float(loss.detach().cpu()) * len(xb); batches += 1
        update_squared = sum(
            float(torch.sum((current.detach() - previous) ** 2).cpu())
            for previous, current in zip(before, model.parameters())
        )
        validation_loss = core.evaluate_loss(
            model, rendered["validation"], validation_y,
            data.task, model_name, device,
        )
        trajectory.append({
            "epoch": float(epoch), "training_loss": total_loss / len(train_y),
            "validation_loss": validation_loss,
            "gradient_norm": (total_grad_squared / max(batches, 1)) ** 0.5,
            "parameter_update_norm": update_squared ** 0.5,
        })
        if epoch in (20, 50, 100, 200) and epoch in required:
            checkpoint_states[epoch] = copy.deepcopy(model.state_dict())
        if validation_loss < best_loss * (1 - relative_minimum):
            best_loss = validation_loss; best_epoch = epoch; stale = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if convergence_epoch is None and stale >= patience:
            convergence_epoch = epoch
            convergence_best_epoch = best_epoch
            if "convergence" in required:
                checkpoint_states["convergence"] = copy.deepcopy(best_state)
        if epoch >= minimum_epochs and convergence_epoch is not None:
            break
    if convergence_epoch is None:
        convergence_epoch = len(trajectory)
        convergence_best_epoch = best_epoch
        if "convergence" in required:
            checkpoint_states["convergence"] = copy.deepcopy(best_state)
    predictions = {}
    for budget in required:
        state = checkpoint_states.get(budget)
        if state is None:
            raise AssertionError(f"missing checkpoint {budget}")
        model.load_state_dict(state)
        validation = core.completion.predict(
            model, rendered["validation"], data.task, model_name, class_map, device
        )
        test = core.completion.predict(
            model, rendered["test"], data.task, model_name, class_map, device
        )
        core.validate_probabilities(test, data.task)
        predictions[budget] = (validation, test)
    elapsed = time.perf_counter() - started
    peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    return predictions, trajectory, int(convergence_best_epoch), convergence_epoch, elapsed, peak


def setup_output(dataset, model_name, size_label, data, design, rows_by_budget):
    output_width = 2 if data.task == "classification" else 1
    outputs = {}
    for budget, rows in rows_by_budget.items():
        path = core.RAW / "experiment_b" / f"{dataset}__{model_name}__n{size_label}__{label(budget)}"
        path.mkdir(parents=True, exist_ok=True)
        outputs[budget] = {
            "path": path, "rows": rows,
            "validation": core.open_memmap(
                path / "validation_predictions.npy",
                (len(rows), len(data.y["validation"]), output_width), np.float32,
            ),
            "test": core.open_memmap(
                path / "test_predictions.npy",
                (len(rows), len(data.y["test"]), output_width), np.float32,
            ),
            "complete": core.open_memmap(path / "complete.npy", (len(rows),), np.bool_),
            "canonical_validation": core.open_memmap(
                path / "canonical_validation_predictions.npy",
                (CANONICAL_COUNT, len(data.y["validation"]), output_width), np.float32,
            ),
            "canonical_test": core.open_memmap(
                path / "canonical_test_predictions.npy",
                (CANONICAL_COUNT, len(data.y["test"]), output_width), np.float32,
            ),
            "canonical_complete": core.open_memmap(
                path / "canonical_complete.npy", (CANONICAL_COUNT,), np.bool_
            ),
        }
        np.save(path / "design_rows.npy", rows)
        np.save(path / "validation_y.npy", data.y["validation"])
        np.save(path / "test_y.npy", data.y["test"])
    return outputs


def wait_for_complete_masks(outputs, key: str, *, timeout_seconds: float = 172800.0) -> None:
    """Barrier for deterministic multi-process path shards sharing fixed memmaps."""
    started = time.monotonic()
    while not all(bool(np.asarray(current[key]).all()) for current in outputs.values()):
        if time.monotonic() - started > timeout_seconds:
            raise TimeoutError(f"timed out waiting for B {key} shard barrier")
        time.sleep(0.2)


def owns_path(index: int, path_shard: int, path_shards: int) -> bool:
    return index % path_shards == path_shard


def a_reference(dataset, model_name, split_seed):
    path = core.RAW / "experiment_a" / f"{dataset}__{model_name}__split{split_seed}"
    if not (path / "manifest.json").exists():
        return None
    actions = np.load(path / "schema_actions.npy")
    return {
        "path": path,
        "lookup": {tuple(int(value) for value in row): index for index, row in enumerate(actions)},
        "joint_validation": np.load(path / "joint_validation.npy", mmap_mode="r"),
        "joint_test": np.load(path / "joint_test.npy", mmap_mode="r"),
        "joint_master": np.load(path / "joint_master_seeds.npy"),
        "canonical_validation": np.load(path / "canonical_validation.npy", mmap_mode="r"),
        "canonical_test": np.load(path / "canonical_test.npy", mmap_mode="r"),
        "canonical_master": np.load(path / "canonical_master_seeds.npy"),
    }


def run_size(
    dataset, model_name, size_label, data, smallest, largest, device, config,
    path_shard: int = 0, path_shards: int = 1,
):
    design = core.completion.views(data, config)
    full = rows_for_condition(data, design, True)
    trajectory_rows = rows_for_condition(data, design, False)
    corner = size_label in {smallest, largest}
    rows_by_budget = {
        budget: full if corner and budget in {20, "convergence"} else trajectory_rows
        for budget in BUDGETS
    }
    lock_dir = core.RAW / "experiment_b" / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{dataset}__{model_name}__n{size_label}.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        outputs = setup_output(dataset, model_name, size_label, data, design, rows_by_budget)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    keys_by_budget = {
        budget: occurrence_keys(rows) for budget, rows in rows_by_budget.items()
    }
    union_keys = sorted(set().union(*[set(keys) for keys in keys_by_budget.values()]))
    row_maps = {
        budget: {key: index for index, key in enumerate(keys_by_budget[budget])}
        for budget in BUDGETS
    }
    split_seed = int(core.CONFIG["experiment_b"]["split_seed"])
    aref = a_reference(dataset, model_name, split_seed) if size_label == smallest else None
    model_hashes = {
        budget: core.model_config_hash(
            model_name, {**config["training"], "budget": budget, "training_size": len(data.y["train"])}
        ) for budget in BUDGETS
    }
    wall_total = 0.0; peak_max = 0; paths_run = 0; reused_20 = 0
    for union_index, (key_row, occurrence) in enumerate(union_keys):
        if not owns_path(union_index, path_shard, path_shards):
            continue
        physical_key = (key_row, occurrence)
        row = np.asarray(key_row, dtype=np.int16)
        joint_required = {
            budget for budget in BUDGETS
            if physical_key in row_maps[budget]
            and not bool(outputs[budget]["complete"][row_maps[budget][physical_key]])
        }
        shared_canonical_index = (
            int(row[3])
            if aref is not None and occurrence == 0 and key_row[:3] == (0, 0, 0)
            else None
        )
        canonical_required = {
            budget for budget in BUDGETS
            if shared_canonical_index is not None
            and not bool(outputs[budget]["canonical_complete"][shared_canonical_index])
        }
        required = joint_required | canonical_required
        if not required:
            continue
        if aref is not None and occurrence == 0:
            action_id = aref["lookup"][tuple(int(value) for value in row[:3])]
            master_seed = int(aref["joint_master"][action_id, int(row[3])])
        else:
            master_seed = core.stable_seed(
                "B-bundle", dataset, model_name, size_label,
                *[int(value) for value in row], "occurrence", occurrence,
            )
        predictions, trajectory, best_epoch, stopped_epoch, elapsed, peak = train_checkpoints(
            data=data, design=design, row=row, model_name=model_name,
            master_seed=master_seed, config=config, device=device, required=required,
            minimum_epochs=max(
                (int(value) for value in required if value != "convergence"), default=1
            ),
        )
        wall_total += elapsed; peak_max = max(peak_max, peak); paths_run += 1
        for budget, (validation, test) in predictions.items():
            current = outputs[budget]
            if budget == 20 and aref is not None and occurrence == 0:
                action_id = aref["lookup"][key_row[:3]]; seed_level = key_row[3]
                validation_gap = float(np.max(np.abs(validation - aref["joint_validation"][action_id, seed_level])))
                test_gap = float(np.max(np.abs(test - aref["joint_test"][action_id, seed_level])))
                if max(validation_gap, test_gap) > 2e-6:
                    raise AssertionError(f"A/B checkpoint mismatch {validation_gap}/{test_gap}")
                validation = np.asarray(aref["joint_validation"][action_id, seed_level])
                test = np.asarray(aref["joint_test"][action_id, seed_level]); reused_20 += 1
            truncated = trajectory if budget == "convergence" else trajectory[: int(budget)]
            curve_best_epoch = (
                best_epoch if budget == "convergence"
                else int(min(truncated, key=lambda value: value["validation_loss"])["epoch"])
            )
            curve_stopped_epoch = stopped_epoch if budget == "convergence" else int(budget)
            if budget in joint_required:
                index = row_maps[budget][physical_key]
                current["validation"][index] = validation; current["test"][index] = test
                current["validation"].flush(); current["test"].flush()
                core.save_json_atomic(
                    current["path"] / "curves" / f"{index:04d}.json",
                    {
                        "fit_index": index, "master_seed": master_seed,
                        "schema_action": list(key_row[:3]), "seed_level": key_row[3],
                        "best_epoch": curve_best_epoch, "stopped_epoch": curve_stopped_epoch,
                        "trajectory": truncated, "checkpointed_bundle": True,
                    },
                )
            if budget in canonical_required:
                canonical_index = int(shared_canonical_index)
                current["canonical_validation"][canonical_index] = validation
                current["canonical_test"][canonical_index] = test
                current["canonical_validation"].flush(); current["canonical_test"].flush()
                core.save_json_atomic(
                    current["path"] / "canonical_curves" / f"{canonical_index:04d}.json",
                    {
                        "fit_index": canonical_index, "master_seed": master_seed,
                        "best_epoch": curve_best_epoch, "stopped_epoch": curve_stopped_epoch,
                        "trajectory": truncated, "checkpointed_bundle": True,
                        "shared_joint_canonical_path": True,
                    },
                )
            fitkey = core.fit_key(
                dataset=dataset, split=split_seed, model=model_name,
                model_hash=model_hashes[budget], schema_digest=core.schema_hash(dataset, row[:3]),
                master_seed=master_seed, training_size=len(data.y["train"]),
                training_budget=label(budget),
            )
            core.register_fit(
                key=fitkey, experiment="B", dataset=dataset, split_seed=split_seed,
                model=model_name, master_seed=master_seed, artifact=current["path"],
                array_index=(
                    f"prediction[{row_maps[budget][physical_key]}]"
                    if budget in joint_required
                    else f"canonical[{shared_canonical_index}]"
                ),
                validation=validation, test=test,
                wall_seconds=elapsed if budget == "convergence" else 0.0,
                peak_device_bytes=peak,
            )
            if budget in joint_required:
                current["complete"][row_maps[budget][physical_key]] = True
                current["complete"].flush()
            if budget in canonical_required:
                current["canonical_complete"][int(shared_canonical_index)] = True
                current["canonical_complete"].flush()
        if paths_run % 4 == 0:
            print(f"B-bundle {dataset}/{model_name}/N={size_label}: paths {paths_run}/{len(union_keys)}", flush=True)

    # All joint shards must finish before any shard enters the canonical loop;
    # the first eight small-N canonical paths are shared joint paths.
    wait_for_complete_masks(outputs, "complete")

    # Canonical paths share all five checkpoints.
    for canonical_index in range(CANONICAL_COUNT):
        if not owns_path(canonical_index, path_shard, path_shards):
            continue
        required = {
            budget for budget in BUDGETS
            if not bool(outputs[budget]["canonical_complete"][canonical_index])
        }
        if not required:
            continue
        if aref is not None:
            master_seed = int(aref["canonical_master"][canonical_index])
        else:
            master_seed = core.stable_seed(
                "B-bundle", dataset, model_name, size_label, "canonical", canonical_index
            )
        row = np.asarray([0, 0, 0, 0], dtype=np.int16)
        predictions, trajectory, best_epoch, stopped_epoch, elapsed, peak = train_checkpoints(
            data=data, design=design, row=row, model_name=model_name,
            master_seed=master_seed, config=config, device=device, required=required,
        )
        wall_total += elapsed; peak_max = max(peak_max, peak); paths_run += 1
        for budget, (validation, test) in predictions.items():
            current = outputs[budget]
            if budget == 20 and aref is not None:
                validation_gap = float(np.max(np.abs(validation - aref["canonical_validation"][canonical_index])))
                test_gap = float(np.max(np.abs(test - aref["canonical_test"][canonical_index])))
                if max(validation_gap, test_gap) > 2e-6:
                    raise AssertionError(f"A/B canonical checkpoint mismatch {validation_gap}/{test_gap}")
                validation = np.asarray(aref["canonical_validation"][canonical_index])
                test = np.asarray(aref["canonical_test"][canonical_index]); reused_20 += 1
            current["canonical_validation"][canonical_index] = validation
            current["canonical_test"][canonical_index] = test
            current["canonical_validation"].flush(); current["canonical_test"].flush()
            truncated = trajectory if budget == "convergence" else trajectory[: int(budget)]
            curve_best_epoch = (
                best_epoch if budget == "convergence"
                else int(min(truncated, key=lambda value: value["validation_loss"])["epoch"])
            )
            curve_stopped_epoch = stopped_epoch if budget == "convergence" else int(budget)
            core.save_json_atomic(
                current["path"] / "canonical_curves" / f"{canonical_index:04d}.json",
                {
                    "fit_index": canonical_index, "master_seed": master_seed,
                    "best_epoch": curve_best_epoch, "stopped_epoch": curve_stopped_epoch,
                    "trajectory": truncated,
                    "checkpointed_bundle": True,
                },
            )
            fitkey = core.fit_key(
                dataset=dataset, split=split_seed, model=model_name,
                model_hash=model_hashes[budget], schema_digest=core.schema_hash(dataset, row[:3]),
                master_seed=master_seed, training_size=len(data.y["train"]),
                training_budget=label(budget),
            )
            core.register_fit(
                key=fitkey, experiment="B", dataset=dataset, split_seed=split_seed,
                model=model_name, master_seed=master_seed, artifact=current["path"],
                array_index=f"canonical[{canonical_index}]", validation=validation,
                test=test, wall_seconds=elapsed if budget == "convergence" else 0.0,
                peak_device_bytes=peak,
            )
            current["canonical_complete"][canonical_index] = True
            current["canonical_complete"].flush()

    if path_shard != 0:
        return
    wait_for_complete_masks(outputs, "canonical_complete")

    # Complete manifests only after every mask validates.
    for budget, current in outputs.items():
        if not bool(np.asarray(current["complete"]).all()) or not bool(np.asarray(current["canonical_complete"]).all()):
            raise AssertionError(f"incomplete B bundle {current['path']}")
        rows = rows_by_budget[budget]
        masters = []
        for (levels, occurrence) in keys_by_budget[budget]:
            row = np.asarray(levels, dtype=np.int16)
            if aref is not None and occurrence == 0:
                masters.append(int(aref["joint_master"][aref["lookup"][tuple(row[:3])], int(row[3])]))
            else:
                masters.append(core.stable_seed(
                    "B-bundle", dataset, model_name, size_label, *row.tolist(),
                    "occurrence", occurrence,
                ))
        canonical_masters = [
            int(aref["canonical_master"][index]) if aref is not None
            else core.stable_seed("B-bundle", dataset, model_name, size_label, "canonical", index)
            for index in range(CANONICAL_COUNT)
        ]
        shared_count = len(set(masters) & set(canonical_masters))
        np.save(current["path"] / "master_seeds.npy", np.asarray(masters, dtype=np.int64))
        np.save(current["path"] / "canonical_master_seeds.npy", np.asarray(canonical_masters, dtype=np.int64))
        manifest = {
            "status": "complete", "experiment": "B", "dataset": dataset,
            "task": data.task, "split_seed": split_seed, "model": model_name,
            "training_size_label": size_label, "training_rows": len(data.y["train"]),
            "budget": budget, "full_product": bool(corner and budget in {20, "convergence"}),
            "design_rows": len(rows), "schema_cards": core.action_cards(data, design),
            "canonical_reference_fits": CANONICAL_COUNT,
            "shared_canonical_joint_fits": shared_count,
            "represented_fits": len(rows) + CANONICAL_COUNT - shared_count,
            "unique_master_seeds": len(set(masters)),
            "independent_seed_records": len(set(masters + canonical_masters)),
            "reused_experiment_a": bool(aref is not None and budget == 20),
            "checkpointed_trajectory": True, "new_paths_this_size_invocation": paths_run,
            "fit_wall_seconds_this_size_invocation": wall_total if budget == "convergence" else 0.0,
            "maximum_peak_device_bytes_this_size_invocation": peak_max,
            "model_config_sha256": model_hashes[budget],
            "protocol_sha256": core.sha256(core.HERE / "FINAL_CLOSURE_PROTOCOL.md"),
            "config_sha256": core.sha256(core.CONFIG_PATH),
        }
        core.save_json_atomic(current["path"] / "manifest.json", manifest)
        print(json.dumps(manifest, indent=2), flush=True)


def run_dataset_model(
    dataset: str, model_name: str, device_name: str,
    path_shard: int = 0, path_shards: int = 1,
):
    if dataset not in core.CONFIG["experiment_b"]["datasets"]:
        raise ValueError("dataset outside B")
    if model_name not in core.CONFIG["primary_models"]:
        raise ValueError("model outside B")
    torch.set_num_threads(1); device = torch.device(device_name)
    config = core.completion_config(); split = int(core.CONFIG["experiment_b"]["split_seed"])
    prepared, raw_indices = core.b_prepared_datasets(dataset, split, config)
    labels = list(prepared); smallest, largest = labels[0], labels[-1]
    sets = [set(raw_indices[current]) for current in labels]
    if any(not sets[index].issubset(sets[index + 1]) for index in range(len(sets) - 1)):
        raise AssertionError("B training sets are not nested")
    for size_label, data in prepared.items():
        run_size(
            dataset, model_name, size_label, data, smallest, largest, device, config,
            path_shard=path_shard, path_shards=path_shards,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True); parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--path-shard", type=int, default=0)
    parser.add_argument("--path-shards", type=int, default=1)
    args = parser.parse_args()
    if args.path_shards < 1 or not 0 <= args.path_shard < args.path_shards:
        raise ValueError("invalid path shard")
    run_dataset_model(
        args.dataset, args.model, args.device,
        path_shard=args.path_shard, path_shards=args.path_shards,
    )


if __name__ == "__main__":
    main()
