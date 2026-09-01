"""Checkpointed exact matched-function convergence subexperiment for B."""

from __future__ import annotations

import argparse
import copy
import json
import time
from typing import Any

import numpy as np
import torch

import closure_core as core
from run_experiment_b_bundle import train_checkpoints


BUDGETS: tuple[int | str, ...] = (20, 100, "convergence")


def label(budget: int | str) -> str:
    return "convergence" if budget == "convergence" else f"epochs{budget}"


def run_bundle(dataset: str, model_name: str, device_name: str) -> dict[str, Any]:
    bconfig = core.CONFIG["experiment_b"]
    if dataset not in bconfig["matched_datasets"]:
        raise ValueError("dataset outside frozen matched panel")
    if model_name not in core.CONFIG["primary_models"]:
        raise ValueError("model outside frozen matched panel")
    split_seed = int(bconfig["split_seed"])
    config = core.completion_config()
    prepared, _ = core.b_prepared_datasets(dataset, split_seed, config)
    data = prepared[list(prepared)[0]]
    design = core.completion.views(data, config)
    actions = core.schema_actions(data, design)
    device = torch.device(device_name)
    torch.set_num_threads(1)
    output_width = 2 if data.task == "classification" else 1
    master_seed = core.stable_seed("B-matched", dataset, model_name, "shared-trajectory")
    subseeds = core.derive_subseeds(master_seed)

    outputs = {}
    for budget in BUDGETS:
        stem = f"{dataset}__{model_name}__n{len(data.y['train'])}__{label(budget)}"
        output = core.RAW / "matched_convergence" / stem
        output.mkdir(parents=True, exist_ok=True)
        outputs[budget] = {
            "path": output,
            "validation": core.open_memmap(
                output / "validation_predictions.npy",
                (2, len(actions), len(data.y["validation"]), output_width), np.float32,
            ),
            "test": core.open_memmap(
                output / "test_predictions.npy",
                (2, len(actions), len(data.y["test"]), output_width), np.float32,
            ),
            "complete": core.open_memmap(
                output / "complete.npy", (2, len(actions)), np.bool_
            ),
            "gaps": core.open_memmap(
                output / "initial_gaps.npy", (len(actions),), np.float64
            ),
        }
        np.save(output / "schema_actions.npy", np.asarray(actions, dtype=np.int16))
        np.save(output / "test_y.npy", data.y["test"])

    canonical_x = core.completion.render(
        data, "train", design["feature"][0], design["category"][0]
    )[0]
    canonical_model = core.initialize_model(
        model_name, canonical_x.shape[1], output_width,
        subseeds["initialization"], config, device,
    )
    canonical_state = copy.deepcopy(canonical_model.state_dict())
    canonical_validation_x = core.completion.render(
        data, "validation", design["feature"][0], design["category"][0]
    )[0]
    canonical_initial = core.completion.predict(
        canonical_model, canonical_validation_x, data.task, model_name,
        design["class"][0], device,
    )

    started = time.perf_counter(); paths_run = 0; wall_total = 0.0; peak_max = 0
    for action_index, (fi, ci, li) in enumerate(actions):
        class_map = design["class"][li]
        rendered_validation = core.completion.render(
            data, "validation", design["feature"][fi], design["category"][ci]
        )[0]
        coordinate_map = core.completion.render(
            data, "train", design["feature"][fi], design["category"][ci]
        )[1]
        row = np.asarray([fi, ci, li, 0], dtype=np.int16)
        for arm_index, arm in enumerate(("ordinary", "matched")):
            required = {
                budget for budget in BUDGETS
                if not bool(outputs[budget]["complete"][arm_index, action_index])
            }
            if not required:
                continue
            initial_state = None
            gap = 0.0
            if arm == "matched":
                probe = core.initialize_model(
                    model_name, canonical_x.shape[1], output_width,
                    subseeds["initialization"], config, device,
                )
                initial_state = core.completion.matched_state(
                    model_name, canonical_state, coordinate_map, class_map
                )
                probe.load_state_dict(initial_state)
                current_initial = core.completion.predict(
                    probe, rendered_validation, data.task, model_name, class_map, device
                )
                gap = float(np.max(np.abs(canonical_initial - current_initial)))
                if gap > float(core.CONFIG["initial_match_tolerance"]):
                    raise AssertionError(f"matched initial gap {gap}")
                del probe
            predictions, trajectory, best_epoch, stopped_epoch, elapsed, peak = train_checkpoints(
                data=data, design=design, row=row, model_name=model_name,
                master_seed=master_seed, config=config, device=device, required=required,
                initial_state=initial_state, minimum_epochs=100,
            )
            paths_run += 1; wall_total += elapsed; peak_max = max(peak_max, peak)
            for budget, (validation, test) in predictions.items():
                current = outputs[budget]
                current["validation"][arm_index, action_index] = validation
                current["test"][arm_index, action_index] = test
                current["validation"].flush(); current["test"].flush()
                if arm == "matched":
                    current["gaps"][action_index] = gap; current["gaps"].flush()
                truncated = trajectory if budget == "convergence" else trajectory[: int(budget)]
                curve_best_epoch = (
                    best_epoch if budget == "convergence"
                    else int(min(truncated, key=lambda value: value["validation_loss"])["epoch"])
                )
                curve_stopped_epoch = stopped_epoch if budget == "convergence" else int(budget)
                core.save_json_atomic(
                    current["path"] / "curves" / f"{arm}_{action_index:03d}.json",
                    {
                        "arm": arm, "action_index": action_index,
                        "master_seed": master_seed, "best_epoch": curve_best_epoch,
                        "stopped_epoch": curve_stopped_epoch, "trajectory": truncated,
                        "checkpointed_bundle": True,
                    },
                )
                model_hash = core.model_config_hash(
                    model_name, {**config["training"], "budget": budget, "matched": arm}
                )
                key = core.fit_key(
                    dataset=dataset, split=split_seed, model=model_name,
                    model_hash=model_hash,
                    schema_digest=core.schema_hash(dataset, actions[action_index]),
                    master_seed=master_seed, training_size=len(data.y["train"]),
                    training_budget=label(budget), matched_arm=arm,
                )
                core.register_fit(
                    key=key, experiment="B-matched", dataset=dataset,
                    split_seed=split_seed, model=model_name, master_seed=master_seed,
                    artifact=current["path"], array_index=f"{arm}[{action_index}]",
                    validation=validation, test=test,
                    wall_seconds=elapsed if budget == "convergence" else 0.0,
                    peak_device_bytes=peak,
                )
                current["complete"][arm_index, action_index] = True
                current["complete"].flush()
            if paths_run % 8 == 0:
                print(
                    f"B-matched {dataset}/{model_name}: paths {paths_run}/{2 * len(actions)}",
                    flush=True,
                )

    manifests = {}
    for budget, current in outputs.items():
        if not bool(np.asarray(current["complete"]).all()):
            raise AssertionError(f"matched convergence condition incomplete: {current['path']}")
        manifest = {
            "status": "complete", "experiment": "B-matched",
            "dataset": dataset, "task": data.task, "split_seed": split_seed,
            "model": model_name, "training_rows": len(data.y["train"]),
            "budget": budget, "schema_actions": len(actions),
            "represented_fits": 2 * len(actions),
            "checkpointed_trajectory": True,
            "new_paths_this_bundle_invocation": paths_run,
            "maximum_initial_gap": float(np.max(np.asarray(current["gaps"]))),
            "fit_wall_seconds_this_bundle_invocation": (
                wall_total if budget == "convergence" else 0.0
            ),
            "maximum_peak_device_bytes_this_bundle_invocation": peak_max,
            "invocation_wall_seconds": time.perf_counter() - started,
            "controlled_master_seed": int(master_seed),
            "protocol_sha256": core.sha256(core.HERE / "FINAL_CLOSURE_PROTOCOL.md"),
            "config_sha256": core.sha256(core.CONFIG_PATH),
        }
        core.save_json_atomic(current["path"] / "manifest.json", manifest)
        print(json.dumps(manifest, indent=2), flush=True)
        manifests[label(budget)] = manifest
    return manifests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run_bundle(args.dataset, args.model, args.device)


if __name__ == "__main__":
    main()
