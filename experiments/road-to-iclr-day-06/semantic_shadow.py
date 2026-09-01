"""H4 short-horizon semantic shadow forecasting bundles."""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import os
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

import semantic_arithmetic as h1

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "hypothesis_04_config.json"
PARTS = ("train", "validation", "test")


def optimizer_configs(config: dict) -> list[dict]:
    rows = []
    for lr, wd, batch in itertools.product(
        config["learning_rates"], config["weight_decays"], config["batch_sizes"]
    ):
        rows.append({
            "id": f"lr{lr:g}__wd{wd:g}__b{batch}",
            "learning_rate": float(lr), "weight_decay": float(wd),
            "batch_size": int(batch),
        })
    return rows


def run_bundle(
    dataset: str, model_name: str, seed: int, config_id: str,
    device_name: str, output_dir: Path, force: bool = False,
) -> Path:
    config = json.loads(CONFIG_PATH.read_text())
    day5_config = json.loads(h1.DAY5_CONFIG_PATH.read_text())
    choices = {row["id"]: row for row in optimizer_configs(config)}
    if dataset not in config["datasets"] or model_name not in config["models"]:
        raise ValueError("bundle outside frozen H4 matrix")
    if seed not in config["seeds"] or config_id not in choices:
        raise ValueError("seed/config outside frozen H4 menu")
    current = choices[config_id]
    run_config = copy.deepcopy(config)
    run_config["training"].update(current)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{dataset}__{model_name}__{config_id}__seed{seed}"
    artifact = output_dir / f"{stem}.npz"
    manifest_path = output_dir / f"{stem}.json"
    if artifact.exists() and manifest_path.exists() and not force:
        if json.loads(manifest_path.read_text()).get("status") == "complete":
            print(f"complete: {stem}", flush=True)
            return artifact

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(device_name)
    data = h1.completion.prepare(dataset, int(config["split_seed"]), day5_config)
    design = h1.completion.views(data, day5_config)
    class_map = design["class"][0]
    canonical_rendered = {
        part: h1.completion.render(
            data, part, design["feature"][0], design["category"][0]
        )[0] for part in PARTS
    }
    width = canonical_rendered["train"].shape[1]
    output_width = 2 if data.task == "classification" else 1
    transformed_target = class_map[data.y["train"]] if data.task == "classification" else data.y["train"]
    actions = [(0, 0)] + [
        (index, index if len(design["category"]) > 1 else 0)
        for index in range(1, int(config["nonidentity_views"]) + 1)
    ]
    h1.seed_all(seed)
    canonical_model = h1.initialize_model(
        model_name, width, output_width, seed, day5_config, device, "fp32"
    )
    canonical_state = copy.deepcopy(canonical_model.state_dict())
    all_validation, all_test, labels, elapsed, initial_gaps = [], [], [], [], []
    for action_index, (feature_index, category_index) in enumerate(actions):
        rendered = {
            part: h1.completion.render(
                data, part, design["feature"][feature_index],
                design["category"][category_index],
            )[0] for part in PARTS
        }
        coordinate_map = h1.completion.render(
            data, "train", design["feature"][feature_index],
            design["category"][category_index],
        )[1]
        model = h1.initialize_model(
            model_name, width, output_width, seed, day5_config, device, "fp32"
        )
        state = canonical_state if action_index == 0 else h1.completion.matched_state(
            model_name, canonical_state, coordinate_map, class_map
        )
        model.load_state_dict(state)
        canonical_initial = h1.completion.predict(
            canonical_model, canonical_rendered["validation"], data.task,
            model_name, class_map, device,
        )
        current_initial = h1.completion.predict(
            model, rendered["validation"], data.task, model_name, class_map, device
        )
        gap = float(np.max(np.abs(canonical_initial - current_initial)))
        if gap > float(config["initial_match_tolerance"]):
            raise AssertionError(f"initial gap {gap}")
        validation, test, seconds = h1.trajectory(
            model=model, rendered=rendered, target=transformed_target, data=data,
            model_name=model_name, class_map=class_map, seed=seed,
            config=run_config, device=device,
        )
        all_validation.append(validation); all_test.append(test)
        labels.append((action_index, feature_index, category_index))
        elapsed.append(seconds); initial_gaps.append(gap)
    np.savez_compressed(
        artifact, validation_predictions=np.asarray(all_validation, dtype=np.float32),
        test_predictions=np.asarray(all_test, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int16),
        checkpoints=np.asarray(config["checkpoints"], dtype=np.int16),
        validation_y=data.y["validation"], test_y=data.y["test"],
    )
    manifest = {
        "status": "complete", "hypothesis": "H4_SEMANTIC_SHADOW",
        "dataset": dataset, "task": data.task, "model": model_name,
        "seed": seed, "optimizer": current, "paths": len(labels),
        "maximum_initial_gap": float(max(initial_gaps)),
        "total_fit_seconds": float(sum(elapsed)),
        "config_sha256": hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(),
        "protocol_sha256": hashlib.sha256(
            (HERE / "HYPOTHESIS_04_PROTOCOL.md").read_bytes()
        ).hexdigest(),
        "torch": torch.__version__, "device": device_name,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"H4 complete {stem}: {sum(elapsed):.2f}s", flush=True)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--config-id", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results" / "h4")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_bundle(
        args.dataset, args.model, args.seed, args.config_id,
        args.device, args.output_dir, args.force,
    )


if __name__ == "__main__":
    main()
