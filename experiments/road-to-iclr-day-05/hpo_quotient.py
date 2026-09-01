"""Frozen prospective quotient-HPO cell runner."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline


HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "hpo_quotient_config.json"


def load_tier1_module():
    path = HERE / "tier1_orbit.py"
    spec = importlib.util.spec_from_file_location("day5_tier1_orbit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load tier1_orbit.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TIER1 = load_tier1_module()


def fit_predict(
    family: str,
    candidate: dict[str, Any],
    seed: int,
    train_x: np.ndarray,
    train_y: np.ndarray,
    query_x: np.ndarray,
    categorical: tuple[int, ...],
    config: dict[str, Any],
) -> np.ndarray:
    training = config["training"]
    threads = int(training["threads_per_fit"])
    if family == "ordinal_forest":
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=int(training["forest_estimators"]),
                n_jobs=threads,
                random_state=seed,
                **candidate,
            ),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(train_x, train_y)
        return np.asarray(model.predict_proba(query_x), dtype=np.float64)
    if family == "catboost_native":
        from catboost import CatBoostClassifier

        train_frame = pd.DataFrame(train_x)
        query_frame = pd.DataFrame(query_x)
        for column in categorical:
            train_frame[column] = train_frame[column].fillna(-1).astype(int).astype(str)
            query_frame[column] = query_frame[column].fillna(-1).astype(int).astype(str)
        model = CatBoostClassifier(
            iterations=int(training["boosting_iterations"]),
            learning_rate=0.08,
            loss_function="Logloss",
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
            thread_count=threads,
            **candidate,
        )
        model.fit(train_frame, train_y, cat_features=list(categorical))
        return np.asarray(model.predict_proba(query_frame), dtype=np.float64)
    raise ValueError(family)


def run_cell(dataset_name: str, family: str, config: dict[str, Any]):
    data_root = Path(config["data_root"]) / dataset_name
    data = TIER1.load_dataset(data_root, config)
    if data.task != "binclass":
        raise ValueError("prospective HPO panel is binary classification only")
    encoded, cardinalities = TIER1.encode_categories(data)
    views = TIER1.make_views(data, config, cardinalities)
    candidates = config["candidates"][family]
    factor_shape = (len(views["feature"]), len(views["category"]), len(views["class"]))
    shape = (len(candidates),) + factor_shape + (len(config["seeds"]),)
    validation = np.empty(shape + (len(data.validation_y), 2), dtype=np.float32)
    test = np.empty(shape + (len(data.test_y), 2), dtype=np.float32)
    total = math.prod(shape)
    completed = 0
    for fi, feature in enumerate(views["feature"]):
        for ci, category in enumerate(views["category"]):
            train_x, categorical = TIER1.render(data.train_n, encoded["train"], feature, category)
            validation_x, _ = TIER1.render(data.validation_n, encoded["validation"], feature, category)
            test_x, _ = TIER1.render(data.test_n, encoded["test"], feature, category)
            joined = np.concatenate((validation_x, test_x), axis=0)
            for li, class_map in enumerate(views["class"]):
                train_y = class_map[data.train_y]
                for hi, candidate in enumerate(candidates):
                    for si, seed in enumerate(config["seeds"]):
                        raw = fit_predict(
                            family, candidate, int(seed), train_x, train_y,
                            joined, categorical, config,
                        )[:, class_map]
                        validation[hi, fi, ci, li, si] = raw[: len(validation_x)]
                        test[hi, fi, ci, li, si] = raw[len(validation_x) :]
                        completed += 1
                        print(f"{dataset_name} {family} {completed}/{total}", flush=True)
    manifest = {
        "status": "complete",
        "dataset": dataset_name,
        "family": family,
        "task": data.task,
        "factor_names": ["feature", "category", "class"],
        "factor_shape": list(factor_shape),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "seeds": config["seeds"],
        "rows": {"train": len(data.train_y), "validation": len(data.validation_y), "test": len(data.test_y)},
        "source_hashes": {
            name: TIER1.sha256(data_root / name)
            for name in ("N_train.npy", "N_val.npy", "N_test.npy", "y_train.npy", "y_val.npy", "y_test.npy")
        },
    }
    return manifest, {
        "validation_predictions": validation,
        "test_predictions": test,
        "validation_y": data.validation_y,
        "test_y": data.test_y,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results" / "hpo_quotient")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.dataset not in config["datasets"] or args.family not in config["families"]:
        raise ValueError("dataset/family not in frozen config")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest, arrays = run_cell(args.dataset, args.family, config)
    stem = f"{args.dataset}__{args.family}"
    np.savez_compressed(args.output_dir / f"{stem}.npz", **arrays)
    (args.output_dir / f"{stem}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

