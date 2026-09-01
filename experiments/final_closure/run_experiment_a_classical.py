"""Secondary first-split CatBoost/XGBoost independent-seed scope for Experiment A."""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

import closure_core as core

sys.path.insert(0, str(core.DAY5))
import completion_classical_panel as classical  # noqa: E402


def run_cell(dataset: str, model_name: str) -> dict:
    if dataset not in core.CONFIG["all_datasets"]:
        raise ValueError("dataset outside frozen secondary A panel")
    if model_name not in core.CONFIG["secondary_a_models"]:
        raise ValueError("model outside frozen secondary A panel")
    split_seed = int(core.CONFIG["experiment_a"]["secondary_splits"][0])
    config = core.completion_config()
    data = core.completion.prepare(dataset, split_seed, config)
    design = core.completion.views(data, config)
    actions = core.schema_actions(data, design)
    seeds_per_schema = int(core.CONFIG["experiment_a"]["joint_master_seeds_per_schema"])
    canonical_count = int(core.CONFIG["experiment_a"]["canonical_master_seeds"])
    output_width = 2 if data.task == "classification" else 1
    stem = f"{dataset}__{model_name}__split{split_seed}"
    output = core.RAW / "experiment_a_classical" / stem
    output.mkdir(parents=True, exist_ok=True)
    joint_validation = core.open_memmap(
        output / "joint_validation.npy",
        (len(actions), seeds_per_schema, len(data.y["validation"]), output_width), np.float32,
    )
    joint_test = core.open_memmap(
        output / "joint_test.npy",
        (len(actions), seeds_per_schema, len(data.y["test"]), output_width), np.float32,
    )
    joint_complete = core.open_memmap(
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
    np.save(output / "schema_actions.npy", np.asarray(actions, dtype=np.int16))
    np.save(output / "test_y.npy", data.y["test"])
    joint_master = np.asarray([
        [core.stable_seed("A-classical", dataset, model_name, "joint", action, repeat)
         for repeat in range(seeds_per_schema)]
        for action in range(len(actions))
    ], dtype=np.int64)
    canonical_master = np.asarray([
        joint_master[0, repeat] if repeat < seeds_per_schema
        else core.stable_seed("A-classical", dataset, model_name, "canonical", repeat)
        for repeat in range(canonical_count)
    ], dtype=np.int64)
    np.save(output / "joint_master_seeds.npy", joint_master)
    np.save(output / "canonical_master_seeds.npy", canonical_master)
    rendered_cache = {}
    for action_index, (fi, ci, li) in enumerate(actions):
        rendered = {}; categorical = None
        for part in core.completion.PARTS:
            rendered[part], current = classical.render_ordinal(
                data, part, design["feature"][fi], design["category"][ci]
            )
            categorical = current if categorical is None else categorical
            if categorical != current:
                raise AssertionError("categorical coordinates changed across splits")
        rendered_cache[action_index] = (rendered, categorical, design["class"][li])
    model_hash = core.model_config_hash(model_name, {"iterations": 80})
    started = time.perf_counter(); fitted = 0; wall_total = 0.0

    def train_one(action_index: int, master_seed: int, array_index: str):
        nonlocal fitted, wall_total
        rendered, categorical, class_map = rendered_cache[action_index]
        train_y = class_map[data.y["train"]] if data.task == "classification" else data.y["train"]
        query = np.concatenate((rendered["validation"], rendered["test"]), axis=0)
        seed = core.derive_subseeds(master_seed)["model_operation"] % (2**31)
        tick = time.perf_counter()
        raw = classical.fit_predict(
            model_name, data.task, int(seed), rendered["train"], train_y,
            query, categorical, 80,
        )
        elapsed = time.perf_counter() - tick
        if data.task == "classification":
            raw = raw[:, class_map]
        validation = np.asarray(raw[: len(data.y["validation"])], dtype=np.float32)
        test = np.asarray(raw[len(data.y["validation"]) :], dtype=np.float32)
        core.validate_probabilities(test, data.task)
        key = core.fit_key(
            dataset=dataset, split=split_seed, model=model_name,
            model_hash=model_hash, schema_digest=core.schema_hash(dataset, actions[action_index]),
            master_seed=int(master_seed), training_size=len(data.y["train"]),
            training_budget="iterations=80",
        )
        core.register_fit(
            key=key, experiment="A-secondary", dataset=dataset,
            split_seed=split_seed, model=model_name, master_seed=int(master_seed),
            artifact=output, array_index=array_index, validation=validation,
            test=test, wall_seconds=elapsed, peak_device_bytes=0,
        )
        fitted += 1; wall_total += elapsed
        return validation, test, elapsed

    for repeat in range(canonical_count):
        if bool(canonical_complete[repeat]):
            try:
                core.validate_probabilities(canonical_test[repeat], data.task); continue
            except AssertionError:
                canonical_complete[repeat] = False
        if repeat < seeds_per_schema and bool(joint_complete[0, repeat]):
            validation = np.asarray(joint_validation[0, repeat]); test = np.asarray(joint_test[0, repeat]); elapsed = 0.0
        else:
            validation, test, elapsed = train_one(
                0, int(canonical_master[repeat]), f"canonical[{repeat}]"
            )
            if repeat < seeds_per_schema:
                joint_validation[0, repeat] = validation; joint_test[0, repeat] = test
                joint_complete[0, repeat] = True
                joint_validation.flush(); joint_test.flush(); joint_complete.flush()
        canonical_validation[repeat] = validation; canonical_test[repeat] = test
        canonical_complete[repeat] = True
        canonical_validation.flush(); canonical_test.flush(); canonical_complete.flush()
    for action_index in range(len(actions)):
        for repeat in range(seeds_per_schema):
            if bool(joint_complete[action_index, repeat]):
                try:
                    core.validate_probabilities(joint_test[action_index, repeat], data.task); continue
                except AssertionError:
                    joint_complete[action_index, repeat] = False
            validation, test, elapsed = train_one(
                action_index, int(joint_master[action_index, repeat]),
                f"joint[{action_index},{repeat}]",
            )
            joint_validation[action_index, repeat] = validation; joint_test[action_index, repeat] = test
            joint_complete[action_index, repeat] = True
            joint_validation.flush(); joint_test.flush(); joint_complete.flush()
            if fitted % 16 == 0:
                print(f"A-classical {stem}: fitted {fitted}", flush=True)
    manifest = {
        "status": "complete", "experiment": "A-secondary", "dataset": dataset,
        "task": data.task, "split_seed": split_seed, "model": model_name,
        "schema_cards": core.action_cards(data, design), "schema_actions": len(actions),
        "joint_pool_fits": len(actions) * seeds_per_schema,
        "canonical_reference_fits": canonical_count,
        "unique_represented_fits": len(actions) * seeds_per_schema + canonical_count - seeds_per_schema,
        "new_fits_this_invocation": fitted, "fit_wall_seconds_this_invocation": wall_total,
        "invocation_wall_seconds": time.perf_counter() - started,
        "model_config_sha256": model_hash,
        "protocol_sha256": core.sha256(core.HERE / "FINAL_CLOSURE_PROTOCOL.md"),
        "config_sha256": core.sha256(core.CONFIG_PATH),
    }
    core.save_json_atomic(output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    run_cell(args.dataset, args.model)


if __name__ == "__main__":
    main()
