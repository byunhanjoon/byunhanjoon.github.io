"""Run one high-dimensional mixed strength-3 OA-128 cell and controls."""

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


TIER1 = load_module("tier1_highdim_strength3", "tier1_orbit.py")
FIELD = load_module("field_highdim_strength3", "highdim_field_cover.py")
ROW = load_module("row_highdim_strength3", "highdim_row_cover.py")
S2 = load_module("s2_highdim_strength3", "analyze_strength2_cover.py")

# Greedily frozen coefficient set in GF(2)^7. With feature span {1,2} and
# seed span {4,8}, every required mixed collection through three declared
# factors has full rank. The exhaustive balance assertion below is definitive.
BINARY_COEFFICIENTS = (
    124, 56, 94, 55, 29, 98, 44, 24, 35, 83, 41, 121, 101, 77,
    64, 89, 61, 84, 111, 50, 71, 18, 118, 38, 74, 115, 23, 104,
)


def parity(values: np.ndarray) -> np.ndarray:
    output = np.zeros(len(values), dtype=int)
    current = values.copy()
    while np.any(current):
        output ^= current & 1
        current >>= 1
    return output


def strength3_base(binary_count: int) -> np.ndarray:
    if binary_count > len(BINARY_COEFFICIENTS):
        raise ValueError("too many binary factors for frozen OA-128")
    inputs = np.arange(128, dtype=int)
    feature = parity(inputs & 1) + 2 * parity(inputs & 2)
    seed = parity(inputs & 4) + 2 * parity(inputs & 8)
    binaries = [parity(inputs & coefficient) for coefficient in BINARY_COEFFICIENTS[:binary_count]]
    base = np.column_stack([feature, seed, *binaries])
    S2.assert_strength(base, (4, 4) + (2,) * binary_count, 3)
    return base


def randomized_design(base: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    output = base[rng.permutation(len(base))].copy()
    output[:, 0] = rng.permutation(4)[output[:, 0]]
    output[:, 1] = rng.permutation(4)[output[:, 1]]
    for column in range(2, output.shape[1]):
        output[:, column] = rng.permutation(2)[output[:, column]]
    return output


def block_design(kind: str, binary_count: int, rng: np.random.Generator) -> np.ndarray:
    if kind == "strength3_oa128":
        return randomized_design(strength3_base(binary_count), rng)
    if kind == "four_strength2_oa32":
        base = FIELD.base_oa(binary_count)
        return np.concatenate([FIELD.randomized_oa(base, rng) for _ in range(4)])
    if kind == "four_marginal32":
        return np.concatenate([ROW.marginal_design(binary_count, rng) for _ in range(4)])
    if kind == "iid128":
        return np.column_stack((
            rng.integers(0, 4, size=128), rng.integers(0, 4, size=128),
            rng.integers(0, 2, size=(128, binary_count)),
        ))
    raise ValueError(kind)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=HERE / "highdim_strength3_config.json")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results" / "highdim_strength3_cover")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.dataset not in config["datasets"] or args.model not in config["models"]:
        raise ValueError("cell not in frozen config")
    data = TIER1.load_dataset(Path(config["data_root"]) / args.dataset, config)
    encoded, cardinalities = TIER1.encode_categories(data)
    views = TIER1.make_views(data, config, cardinalities)
    mapping_rng = np.random.default_rng(6_120_827)
    field_maps = []
    for size in cardinalities:
        candidate = mapping_rng.permutation(size)
        if size > 1 and np.array_equal(candidate, np.arange(size)):
            candidate = np.roll(candidate, 1)
        field_maps.append((np.arange(size), candidate))
    binary_count = 2 + len(cardinalities)
    base = strength3_base(binary_count)
    row_rng = np.random.default_rng(int(config["row_permutation_seed"]))
    row_orders = (np.arange(len(data.train_y)), row_rng.permutation(len(data.train_y)))
    methods = ("strength3_oa128", "four_strength2_oa32", "four_marginal32", "iid128")
    repetitions = int(config["repetitions"])
    validation_outputs = np.empty((len(methods), repetitions, len(data.validation_y), 2), dtype=np.float32)
    test_outputs = np.empty((len(methods), repetitions, len(data.test_y), 2), dtype=np.float32)
    rng = FIELD.cell_rng(int(config["design_seed"]), args.dataset, args.model)
    for method_index, method in enumerate(methods):
        for repetition in range(repetitions):
            design = block_design(method, binary_count, rng)
            validation_sum = np.zeros((len(data.validation_y), 2), dtype=np.float64)
            test_sum = np.zeros((len(data.test_y), 2), dtype=np.float64)
            for row in design:
                feature_level, seed_level, class_bit, row_bit = row[:4]
                maps = [field_maps[index][int(bit)] for index, bit in enumerate(row[4:])]
                train_x, categorical = TIER1.render(data.train_n, encoded["train"], views["feature"][feature_level], maps)
                validation_x, _ = TIER1.render(data.validation_n, encoded["validation"], views["feature"][feature_level], maps)
                test_x, _ = TIER1.render(data.test_n, encoded["test"], views["feature"][feature_level], maps)
                order = row_orders[int(row_bit)]
                class_map = views["class"][class_bit]
                joined = np.concatenate((validation_x, test_x), axis=0)
                raw = TIER1.fit_predict(
                    args.model, data.task, int(config["seeds"][seed_level]),
                    train_x[order], class_map[data.train_y[order]], joined, categorical, config,
                )[:, class_map]
                validation_sum += raw[: len(validation_x)]
                test_sum += raw[len(validation_x):]
            validation_outputs[method_index, repetition] = validation_sum / 128
            test_outputs[method_index, repetition] = test_sum / 128
            print(f"{args.dataset} {args.model} {method} repetition {repetition + 1}/{repetitions}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.dataset}__{args.model}"
    np.savez_compressed(
        args.output_dir / f"{stem}.npz", validation_predictions=validation_outputs,
        test_predictions=test_outputs, validation_y=data.validation_y, test_y=data.test_y,
    )
    manifest = {
        "status": "complete", "dataset": args.dataset, "model": args.model,
        "methods": list(methods), "repetitions": repetitions, "ensemble_runs": 128,
        "binary_factors": binary_count, "categorical_fields": len(cardinalities),
        "mixed_strength3_balance_verified": True, "base_runs": len(base),
    }
    (args.output_dir / f"{stem}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
