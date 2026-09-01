"""Run one field-wise and row-order OA-32 versus controls cell."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TIER1 = load_module("tier1_highdim_row", "tier1_orbit.py")
FIELD = load_module("field_highdim_row", "highdim_field_cover.py")


def marginal_design(binary_count: int, rng: np.random.Generator) -> np.ndarray:
    columns = [
        rng.permutation(np.repeat(np.arange(4), 8)),
        rng.permutation(np.repeat(np.arange(4), 8)),
    ]
    columns.extend(rng.permutation(np.repeat(np.arange(2), 16)) for _ in range(binary_count))
    return np.column_stack(columns)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=HERE / "highdim_row_config.json")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results" / "highdim_row_cover")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.dataset not in config["datasets"] or args.model not in config["models"]:
        raise ValueError("cell not in frozen config")

    data = TIER1.load_dataset(Path(config["data_root"]) / args.dataset, config)
    if data.task != "binclass":
        raise ValueError("high-dimensional row panel is classification only")
    encoded, cardinalities = TIER1.encode_categories(data)
    views = TIER1.make_views(data, config, cardinalities)
    mapping_rng = np.random.default_rng(6_120_827)
    field_maps = []
    for size in cardinalities:
        candidate = mapping_rng.permutation(size)
        if size > 1 and np.array_equal(candidate, np.arange(size)):
            candidate = np.roll(candidate, 1)
        field_maps.append((np.arange(size), candidate))

    # Binary columns are class, row order, then one per categorical field.
    binary_count = 2 + len(cardinalities)
    base = FIELD.base_oa(binary_count)
    factor_cardinalities = (4, 4) + (2,) * binary_count
    FIELD.assert_pairwise_balance(base, factor_cardinalities)
    row_rng = np.random.default_rng(int(config["row_permutation_seed"]))
    row_orders = (np.arange(len(data.train_y)), row_rng.permutation(len(data.train_y)))
    if np.array_equal(row_orders[0], row_orders[1]):
        raise AssertionError("row-order nuisance must be nonidentity")

    methods = ("oa32", "marginal32", "iid32")
    repetitions = int(config["repetitions"])
    validation_outputs = np.empty((3, repetitions, len(data.validation_y), 2), dtype=np.float32)
    test_outputs = np.empty((3, repetitions, len(data.test_y), 2), dtype=np.float32)
    rng = FIELD.cell_rng(int(config["design_seed"]), args.dataset, args.model)
    for method_index, method in enumerate(methods):
        for repetition in range(repetitions):
            if method == "oa32":
                design = FIELD.randomized_oa(base, rng)
            elif method == "marginal32":
                design = marginal_design(binary_count, rng)
            else:
                design = FIELD.iid_design(binary_count, rng)
            validation_sum = np.zeros((len(data.validation_y), 2), dtype=np.float64)
            test_sum = np.zeros((len(data.test_y), 2), dtype=np.float64)
            for run, row in enumerate(design):
                feature_level, seed_level, class_bit, row_bit = row[:4]
                category_bits = row[4:]
                maps = [field_maps[index][int(bit)] for index, bit in enumerate(category_bits)]
                train_x, categorical = TIER1.render(
                    data.train_n, encoded["train"], views["feature"][feature_level], maps
                )
                validation_x, _ = TIER1.render(
                    data.validation_n, encoded["validation"], views["feature"][feature_level], maps
                )
                test_x, _ = TIER1.render(
                    data.test_n, encoded["test"], views["feature"][feature_level], maps
                )
                order = row_orders[int(row_bit)]
                class_map = views["class"][class_bit]
                joined = np.concatenate((validation_x, test_x), axis=0)
                raw = TIER1.fit_predict(
                    args.model, data.task, int(config["seeds"][seed_level]),
                    train_x[order], class_map[data.train_y[order]], joined, categorical, config,
                )[:, class_map]
                validation_sum += raw[: len(validation_x)]
                test_sum += raw[len(validation_x) :]
                print(
                    f"{args.dataset} {args.model} {method} "
                    f"rep={repetition + 1}/{repetitions} run={run + 1}/32",
                    flush=True,
                )
            validation_outputs[method_index, repetition] = validation_sum / 32
            test_outputs[method_index, repetition] = test_sum / 32

    manifest = {
        "status": "complete", "dataset": args.dataset, "model": args.model,
        "methods": list(methods), "repetitions": repetitions,
        "binary_factors": binary_count, "categorical_fields": len(cardinalities),
        "row_order_factor": True, "full_joint_cell_count": int(np.prod(factor_cardinalities)),
        "ensemble_runs": 32, "pairwise_balance_verified": True,
        "rows": {"train": len(data.train_y), "validation": len(data.validation_y), "test": len(data.test_y)},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.dataset}__{args.model}"
    np.savez_compressed(
        args.output_dir / f"{stem}.npz",
        validation_predictions=validation_outputs, test_predictions=test_outputs,
        validation_y=data.validation_y, test_y=data.test_y,
    )
    (args.output_dir / f"{stem}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

