"""Run one high-dimensional field-wise OA-32 versus iid-32 cell."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def load_tier1():
    path = HERE / "tier1_orbit.py"
    spec = importlib.util.spec_from_file_location("tier1_highdim", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TIER1 = load_tier1()


def binary_forms(count: int) -> np.ndarray:
    candidates = []
    for value in range(1, 32):
        bits = np.asarray([(value >> index) & 1 for index in range(5)], dtype=int)
        # Exclude the feature span (bits 0,1 only) and seed span (bits 2,3 only).
        in_feature_span = not np.any(bits[2:])
        in_seed_span = not np.any(bits[[0, 1, 4]])
        if not in_feature_span and not in_seed_span:
            candidates.append(bits)
    if count > len(candidates):
        raise ValueError(f"requested {count} binary factors, maximum is {len(candidates)}")
    return np.asarray(candidates[:count])


def base_oa(binary_count: int) -> np.ndarray:
    bits = np.asarray([[(row >> index) & 1 for index in range(5)] for row in range(32)], dtype=int)
    feature = bits[:, 0] + 2 * bits[:, 1]
    seed = bits[:, 2] + 2 * bits[:, 3]
    forms = binary_forms(binary_count)
    binary = (bits @ forms.T) % 2
    return np.column_stack((feature, seed, binary))


def assert_pairwise_balance(design: np.ndarray, cardinalities: tuple[int, ...]) -> None:
    for first in range(len(cardinalities)):
        counts = np.bincount(design[:, first], minlength=cardinalities[first])
        if np.unique(counts).size != 1:
            raise AssertionError(f"unbalanced factor {first}")
        for second in range(first + 1, len(cardinalities)):
            pair = np.zeros((cardinalities[first], cardinalities[second]), dtype=int)
            for row in design:
                pair[row[first], row[second]] += 1
            if np.unique(pair).size != 1:
                raise AssertionError(f"unbalanced pair {first},{second}")


def randomized_oa(base: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    output = base.copy()
    output[:, 0] = rng.permutation(4)[output[:, 0]]
    output[:, 1] = rng.permutation(4)[output[:, 1]]
    output[:, 2:] ^= rng.integers(0, 2, size=output.shape[1] - 2)
    return output


def iid_design(binary_count: int, rng: np.random.Generator) -> np.ndarray:
    return np.column_stack((
        rng.integers(0, 4, size=32), rng.integers(0, 4, size=32),
        rng.integers(0, 2, size=(32, binary_count)),
    ))


def cell_rng(seed: int, dataset: str, model: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}:{dataset}:{model}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=HERE / "highdim_field_config.json")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results" / "highdim_field_cover")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.dataset not in config["datasets"] or args.model not in config["models"]:
        raise ValueError("cell not in frozen config")
    data_root = Path(config["data_root"]) / args.dataset
    data = TIER1.load_dataset(data_root, config)
    if data.task != "binclass":
        raise ValueError("high-dimensional panel is classification only")
    encoded, cardinalities = TIER1.encode_categories(data)
    views = TIER1.make_views(data, config, cardinalities)
    field_maps = []
    mapping_rng = np.random.default_rng(6_120_827)
    for size in cardinalities:
        candidate = mapping_rng.permutation(size)
        if size > 1 and np.array_equal(candidate, np.arange(size)):
            candidate = np.roll(candidate, 1)
        field_maps.append((np.arange(size), candidate))
    binary_count = 1 + len(cardinalities)  # class plus one factor per category field
    base = base_oa(binary_count)
    factor_cardinalities = (4, 4) + (2,) * binary_count
    assert_pairwise_balance(base, factor_cardinalities)
    repetitions = int(config["repetitions"])
    validation_outputs = np.empty((2, repetitions, len(data.validation_y), 2), dtype=np.float32)
    test_outputs = np.empty((2, repetitions, len(data.test_y), 2), dtype=np.float32)
    rng = cell_rng(int(config["design_seed"]), args.dataset, args.model)
    for method_index, method in enumerate(("oa32", "iid32")):
        for repetition in range(repetitions):
            design = randomized_oa(base, rng) if method == "oa32" else iid_design(binary_count, rng)
            validation_sum = np.zeros((len(data.validation_y), 2), dtype=np.float64)
            test_sum = np.zeros((len(data.test_y), 2), dtype=np.float64)
            for run, row in enumerate(design):
                feature_level, seed_level, class_bit = row[:3]
                category_bits = row[3:]
                maps = [field_maps[index][int(bit)] for index, bit in enumerate(category_bits)]
                train_x, categorical = TIER1.render(data.train_n, encoded["train"], views["feature"][feature_level], maps)
                validation_x, _ = TIER1.render(data.validation_n, encoded["validation"], views["feature"][feature_level], maps)
                test_x, _ = TIER1.render(data.test_n, encoded["test"], views["feature"][feature_level], maps)
                class_map = views["class"][class_bit]
                joined = np.concatenate((validation_x, test_x), axis=0)
                raw = TIER1.fit_predict(
                    args.model, data.task, int(config["seeds"][seed_level]),
                    train_x, class_map[data.train_y], joined, categorical, config,
                )[:, class_map]
                validation_sum += raw[: len(validation_x)]
                test_sum += raw[len(validation_x) :]
                print(f"{args.dataset} {args.model} {method} rep={repetition + 1}/{repetitions} run={run + 1}/32", flush=True)
            validation_outputs[method_index, repetition] = validation_sum / 32
            test_outputs[method_index, repetition] = test_sum / 32
    manifest = {
        "status": "complete", "dataset": args.dataset, "model": args.model,
        "methods": ["oa32", "iid32"], "repetitions": repetitions,
        "binary_factors": binary_count, "categorical_fields": len(cardinalities),
        "full_joint_cell_count": int(np.prod(factor_cardinalities)),
        "oa_runs": 32, "pairwise_balance_verified": True,
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

