"""Deterministic orbit map and frozen canonical performance runner."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent


def load_tier1_module():
    path = HERE / "tier1_orbit.py"
    spec = importlib.util.spec_from_file_location("day5_tier1_for_canonical", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load tier1_orbit.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TIER1 = load_tier1_module()


def membership_key(values: np.ndarray, level: int) -> bytes:
    return np.packbits(np.asarray(values == level, dtype=np.uint8), bitorder="little").tobytes()


def canonicalize_target(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    levels = np.unique(y).astype(int)
    ordered = sorted((membership_key(y, int(level)), int(level)) for level in levels)
    old_to_new = np.full(int(levels.max()) + 1, -1, dtype=int)
    for new, (_, old) in enumerate(ordered):
        old_to_new[old] = new
    return old_to_new[y.astype(int)], old_to_new


def canonicalize_tables(
    train_x: np.ndarray,
    queries: list[np.ndarray],
    categorical: tuple[int, ...],
) -> tuple[np.ndarray, list[np.ndarray], tuple[int, ...]]:
    train = np.asarray(train_x, dtype=np.float64).copy()
    output_queries = [np.asarray(query, dtype=np.float64).copy() for query in queries]
    categorical_set = set(categorical)
    for column in categorical:
        known = train[:, column] >= 0
        levels = np.unique(train[known, column]).astype(int)
        ordered = sorted((membership_key(train[:, column], int(level)), int(level)) for level in levels)
        mapping = {old: new for new, (_, old) in enumerate(ordered)}
        for array in [train, *output_queries]:
            original = array[:, column].copy()
            array[:, column] = -1
            for old, new in mapping.items():
                array[original == old, column] = new

    def feature_key(column: int) -> tuple[bytes, bytes, bytes]:
        values = train[:, column]
        missing = np.isnan(values)
        normalized = np.where(missing, 0.0, values).astype("<f8", copy=False)
        marker = b"C" if column in categorical_set else b"N"
        return marker, np.packbits(missing, bitorder="little").tobytes(), normalized.tobytes()

    order = sorted(range(train.shape[1]), key=feature_key)
    transformed_categorical = tuple(new for new, old in enumerate(order) if old in categorical_set)
    return train[:, order], [query[:, order] for query in output_queries], transformed_categorical


def array_digest(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.shape).encode())
        digest.update(contiguous.dtype.str.encode())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def run_cell(dataset_name: str, model_name: str, config: dict[str, Any]):
    data_root = Path(config["data_root"]) / dataset_name
    data = TIER1.load_dataset(data_root, config)
    encoded, cardinalities = TIER1.encode_categories(data)
    views = TIER1.make_views(data, config, cardinalities)
    reference = None
    view_digests = []
    semantic_to_canonical = None
    for feature in views["feature"]:
        for category in views["category"]:
            train_x, categorical = TIER1.render(data.train_n, encoded["train"], feature, category)
            validation_x, _ = TIER1.render(data.validation_n, encoded["validation"], feature, category)
            test_x, _ = TIER1.render(data.test_n, encoded["test"], feature, category)
            canonical_train, canonical_queries, canonical_categorical = canonicalize_tables(
                train_x, [validation_x, test_x], categorical
            )
            for class_map in views["class"]:
                rendered_y = class_map[data.train_y] if data.task == "binclass" else data.train_y
                canonical_y, rendered_to_canonical = canonicalize_target(rendered_y.astype(int)) if data.task == "binclass" else (rendered_y, np.asarray([0]))
                alignment = rendered_to_canonical[class_map] if data.task == "binclass" else np.asarray([0])
                bundle = (
                    canonical_train, canonical_queries[0], canonical_queries[1],
                    canonical_y, canonical_categorical, alignment,
                )
                digest = array_digest(*bundle[:4], np.asarray(bundle[4]), np.asarray(bundle[5]))
                view_digests.append(digest)
                if reference is None:
                    reference = bundle
                    semantic_to_canonical = alignment
                else:
                    for current, expected in zip(bundle[:4], reference[:4]):
                        if not np.array_equal(current, expected, equal_nan=True):
                            raise AssertionError("canonical orbit did not close")
                    if bundle[4] != reference[4] or not np.array_equal(alignment, semantic_to_canonical):
                        raise AssertionError("canonical metadata did not close")
    assert reference is not None and semantic_to_canonical is not None
    train_x, validation_x, test_x, train_y, categorical, _ = reference
    joined = np.concatenate((validation_x, test_x), axis=0)
    output_dim = 2 if data.task == "binclass" else 1
    validation_predictions = np.empty((len(config["seeds"]), len(data.validation_y), output_dim), dtype=np.float32)
    test_predictions = np.empty((len(config["seeds"]), len(data.test_y), output_dim), dtype=np.float32)
    for seed_index, seed in enumerate(config["seeds"]):
        raw = TIER1.fit_predict(
            model_name, data.task, int(seed), train_x, train_y,
            joined, categorical, config,
        )
        aligned = raw[:, semantic_to_canonical] if data.task == "binclass" else raw
        validation_predictions[seed_index] = aligned[: len(validation_x)]
        test_predictions[seed_index] = aligned[len(validation_x) :]
    manifest = {
        "status": "complete",
        "dataset": dataset_name,
        "model": model_name,
        "task": data.task,
        "seeds": config["seeds"],
        "checked_orbit_views": len(view_digests),
        "unique_canonical_input_digests": len(set(view_digests)),
        "canonical_input_digest": view_digests[0],
        "rows": {"train": len(data.train_y), "validation": len(data.validation_y), "test": len(data.test_y)},
    }
    return manifest, {
        "validation_predictions": validation_predictions,
        "test_predictions": test_predictions,
        "validation_y": data.validation_y,
        "test_y": data.test_y,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=HERE / "tier1_config.json")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results" / "canonical_orbit")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    allowed = ("ordinal_forest", "native_histgb", "catboost_native", "onehot_adam_mlp")
    if args.dataset not in config["datasets"] or args.model not in allowed:
        raise ValueError("cell not in frozen canonical panel")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest, arrays = run_cell(args.dataset, args.model, config)
    stem = f"{args.dataset}__{args.model}"
    np.savez_compressed(args.output_dir / f"{stem}.npz", **arrays)
    (args.output_dir / f"{stem}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

