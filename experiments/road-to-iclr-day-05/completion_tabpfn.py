"""Frozen TabPFN default/internal/external nuisance completion panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from tabpfn import TabPFNClassifier

from completion_neural_panel import CONFIG, PARTS, digest, prepare, views


HERE = Path(__file__).resolve().parent
CHECKPOINT = Path("/home/byunhanjoon/.cache/tabpfn/tabpfn-v2.5-classifier-v2.5_default.ckpt")
SETTINGS = ((1, "none"), (1, "default"), (8, "default"))


def render_tabpfn(data, part: str, feature: np.ndarray, category: list[np.ndarray]) -> tuple[np.ndarray, tuple[int, ...]]:
    n_num = data.x_num[part].shape[1]
    blocks = []
    categorical_positions = []
    for position, field in enumerate(feature):
        field = int(field)
        if field < n_num:
            blocks.append(data.x_num[part][:, field : field + 1])
        else:
            cat = field - n_num
            mapping = category[cat]
            values = np.full(len(data.x_cat[part]), -1, dtype=np.float32)
            known = data.x_cat[part][:, cat] >= 0
            values[known] = mapping[data.x_cat[part][known, cat]]
            blocks.append(values[:, None])
            categorical_positions.append(position)
    return np.ascontiguousarray(np.concatenate(blocks, axis=1), dtype=np.float32), tuple(categorical_positions)


def run(dataset: str, split_seed: int, device: str, config: dict[str, Any], output: Path) -> None:
    data = prepare(dataset, split_seed, config)
    if data.task != "classification":
        raise ValueError("TabPFN completion is classification-only")
    design = views(data, config)
    actions = list(np.ndindex(len(design["feature"]), len(design["category"]), len(design["class"])))
    arrays: dict[str, np.ndarray] = {}
    telemetry = []
    for estimators, policy in SETTINGS:
        validation = np.empty((len(actions), len(data.y["validation"]), 2), dtype=np.float32)
        test = np.empty((len(actions), len(data.y["test"]), 2), dtype=np.float32)
        for index, (fi, ci, li) in enumerate(actions):
            rendered = {}
            categorical = None
            for part in PARTS:
                rendered[part], current = render_tabpfn(data, part, design["feature"][fi], design["category"][ci])
                if categorical is None:
                    categorical = current
                elif categorical != current:
                    raise AssertionError("categorical positions changed across splits")
            class_map = design["class"][li]
            inference_config = None if policy == "default" else {
                "FEATURE_SHIFT_METHOD": None, "CLASS_SHIFT_METHOD": None,
            }
            model = TabPFNClassifier(
                n_estimators=estimators, categorical_features_indices=categorical,
                model_path=CHECKPOINT, device=device, random_state=4201,
                inference_config=inference_config, fit_mode="fit_preprocessors",
            )
            started = time.perf_counter()
            model.fit(rendered["train"], class_map[data.y["train"]])
            joined = np.concatenate((rendered["validation"], rendered["test"]), axis=0)
            raw = model.predict_proba(joined)[:, class_map]
            elapsed = time.perf_counter() - started
            validation[index] = raw[: len(data.y["validation"])]
            test[index] = raw[len(data.y["validation"]) :]
            telemetry.append({
                "estimators": estimators, "policy": policy, "action": index,
                "feature": fi, "category": ci, "class": li,
                "wall_seconds": elapsed, "forward_ensemble_members": estimators,
            })
            print(f"{dataset} split={split_seed} tabpfn {estimators}:{policy} {index + 1}/{len(actions)}", flush=True)
        arrays[f"validation__{estimators}__{policy}"] = validation
        arrays[f"test__{estimators}__{policy}"] = test
    arrays["validation_y"] = data.y["validation"]
    arrays["test_y"] = data.y["test"]
    arrays["actions"] = np.asarray(actions, dtype=np.int16)
    stem = f"{dataset}__split{split_seed}"
    np.savez_compressed(output / f"{stem}.npz", **arrays)
    manifest = {
        "status": "complete", "dataset": dataset, "split_seed": split_seed,
        "actions": len(actions), "settings": [list(item) for item in SETTINGS],
        "tabpfn_calls": len(actions) * len(SETTINGS),
        "forward_ensemble_members": int(sum(item[0] for item in SETTINGS) * len(actions)),
        "checkpoint": str(CHECKPOINT), "checkpoint_sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
        "device": device,
        "rows": {part: len(data.y[part]) for part in PARTS},
        "wall_seconds": float(sum(item["wall_seconds"] for item in telemetry)),
        "protocol_sha256": config["protocol_sha256"], "telemetry": telemetry,
    }
    (output / f"{stem}.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split-seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results" / "completion_tabpfn")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if digest(HERE / config["protocol"]) != config["protocol_sha256"]:
        raise AssertionError("completion protocol hash mismatch")
    if config["dataset_tasks"].get(args.dataset) != "classification":
        raise ValueError("dataset is outside frozen classification panel")
    if args.split_seed not in config["split_seeds"]:
        raise ValueError("split seed outside frozen panel")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run(args.dataset, args.split_seed, args.device, config, args.output_dir)


if __name__ == "__main__":
    main()
