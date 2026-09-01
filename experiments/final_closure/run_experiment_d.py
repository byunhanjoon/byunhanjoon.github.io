"""Run one full finite prediction tensor for coupling Experiment D."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

import closure_core as core


def run_cell(dataset: str, split_seed: int, model_name: str, device_name: str) -> dict:
    dconfig = core.CONFIG["experiment_d"]
    if dataset not in dconfig["datasets"] or split_seed not in core.CONFIG["split_seeds"]:
        raise ValueError("cell outside frozen Experiment D")
    if model_name not in core.CONFIG["primary_models"]:
        raise ValueError("model outside frozen Experiment D")
    config = core.completion_config()
    torch.set_num_threads(1)
    device = torch.device(device_name)
    data = core.completion.prepare(dataset, split_seed, config)
    design = core.completion.views(data, config)
    schema = core.schema_actions(data, design)
    init_seeds = [int(value) for value in dconfig["init_seeds"]]
    order_seeds = [int(value) for value in dconfig["order_seeds"]]
    rows = np.asarray(
        [(*action, init_index, order_index) for action in schema
         for init_index in range(4) for order_index in range(4)],
        dtype=np.int16,
    )
    output_width = 2 if data.task == "classification" else 1
    stem = f"{dataset}__{model_name}__split{split_seed}"
    output = core.RAW / "experiment_d" / stem
    output.mkdir(parents=True, exist_ok=True)
    validation_array = core.open_memmap(
        output / "validation_predictions.npy",
        (len(rows), len(data.y["validation"]), output_width), np.float32,
    )
    test_array = core.open_memmap(
        output / "test_predictions.npy",
        (len(rows), len(data.y["test"]), output_width), np.float32,
    )
    complete = core.open_memmap(output / "complete.npy", (len(rows),), np.bool_)
    np.save(output / "design_rows.npy", rows)
    np.save(output / "validation_y.npy", data.y["validation"])
    np.save(output / "test_y.npy", data.y["test"])
    model_hash = core.model_config_hash(
        model_name, {**config["training"], "finite_init": init_seeds, "finite_order": order_seeds}
    )
    fitted = 0; wall_total = 0.0; peak_max = 0; started = time.perf_counter()
    for index, row in enumerate(rows):
        if bool(complete[index]):
            try:
                core.validate_probabilities(test_array[index], data.task)
                continue
            except AssertionError:
                complete[index] = False
        fi, ci, li, ii, oi = [int(value) for value in row]
        class_map = design["class"][li]
        rendered = {
            part: core.completion.render(
                data, part, design["feature"][fi], design["category"][ci]
            )[0]
            for part in core.completion.PARTS
        }
        model = core.initialize_model(
            model_name, rendered["train"].shape[1], output_width,
            init_seeds[ii], config, device,
        )
        # D is a declared finite init/order menu, not an independent master-RNG
        # pool.  The order coordinate controls minibatch/dropout stochasticity.
        subseeds = {
            domain: order_seeds[oi] for domain in core.CONFIG["rng"]["domains"]
        }
        subseeds["initialization"] = init_seeds[ii]
        transformed_y = (
            class_map[data.y["train"]] if data.task == "classification" else data.y["train"]
        )
        elapsed, peak = core.fit_fixed(
            model, rendered["train"], transformed_y, data.task, model_name,
            subseeds, config["training"], device,
        )
        validation = core.completion.predict(
            model, rendered["validation"], data.task, model_name, class_map, device
        )
        test = core.completion.predict(
            model, rendered["test"], data.task, model_name, class_map, device
        )
        core.validate_probabilities(test, data.task)
        validation_array[index] = validation; test_array[index] = test
        validation_array.flush(); test_array.flush()
        key = core.fit_key(
            dataset=dataset, split=split_seed, model=model_name,
            model_hash=model_hash, schema_digest=core.schema_hash(dataset, row[:3]),
            master_seed=None, training_size=len(data.y["train"]),
            training_budget="epochs=20", finite_init_seed=init_seeds[ii],
            finite_order_seed=order_seeds[oi],
        )
        core.register_fit(
            key=key, experiment="D", dataset=dataset, split_seed=split_seed,
            model=model_name, master_seed=None, artifact=output,
            array_index=f"prediction[{index}]", validation=validation, test=test,
            wall_seconds=elapsed, peak_device_bytes=peak,
        )
        complete[index] = True; complete.flush()
        fitted += 1; wall_total += elapsed; peak_max = max(peak_max, peak)
        if fitted % 16 == 0:
            print(f"D {stem}: fitted {fitted}/{len(rows)}", flush=True)
    if not bool(np.asarray(complete).all()):
        raise AssertionError("Experiment-D cell incomplete")
    manifest = {
        "status": "complete", "experiment": "D", "dataset": dataset,
        "task": data.task, "split_seed": split_seed, "model": model_name,
        "schema_cards": core.action_cards(data, design),
        "finite_cards": [*core.action_cards(data, design), 4, 4],
        "represented_fits": len(rows), "new_fits_this_invocation": fitted,
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
