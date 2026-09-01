"""Paired path experiment for semantic arithmetic amplification."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn

import sys

HERE = Path(__file__).resolve().parent
DAY5 = HERE.parent / "road-to-iclr-day-05"
sys.path.insert(0, str(DAY5))
import completion_neural_panel as completion  # noqa: E402

CONFIG_PATH = HERE / "hypothesis_01_config.json"
DAY5_CONFIG_PATH = DAY5 / "completion_config.json"
PARTS = ("train", "validation", "test")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ExactAccumLinear(nn.Linear):
    """Float32 parameters with a float64 dot-product accumulator."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        result = nn.functional.linear(
            value.to(torch.float64),
            self.weight.to(torch.float64),
            None if self.bias is None else self.bias.to(torch.float64),
        )
        return result.to(torch.float32)


def convert_interface(model: nn.Module, model_name: str) -> nn.Module:
    old = model.first
    replacement = ExactAccumLinear(
        old.in_features, old.out_features, bias=old.bias is not None,
        device=old.weight.device, dtype=old.weight.dtype,
    )
    with torch.no_grad():
        replacement.weight.copy_(old.weight)
        if old.bias is not None:
            replacement.bias.copy_(old.bias)
    model.first = replacement
    if model_name == "mlp":
        model.network[0] = replacement
    return model


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def proper_loss(prediction: np.ndarray, target: np.ndarray, task: str) -> float:
    values = np.asarray(prediction, dtype=np.float64)
    if task == "classification":
        probabilities = np.clip(values, 1e-12, 1.0)
        return float(-np.log(probabilities[np.arange(len(target)), target.astype(int)]).mean())
    return float(np.mean((values.reshape(-1) - target.reshape(-1)) ** 2))


def initialize_model(
    model_name: str, width: int, output_width: int, seed: int,
    day5_config: dict[str, Any], device: torch.device, precision: str,
) -> nn.Module:
    model = completion.initialize(model_name, width, output_width, seed, day5_config, device)
    if precision == "iea64":
        model = convert_interface(model, model_name)
    return model


def trajectory(
    *, model: nn.Module, rendered: dict[str, np.ndarray], target: np.ndarray,
    data: completion.Prepared, model_name: str, class_map: np.ndarray,
    seed: int, config: dict[str, Any], device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    training = config["training"]
    checkpoints = [int(value) for value in config["checkpoints"]]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    batch = int(training["batch_size"])
    seed_all(seed + 1_000_003)
    order_rng = np.random.default_rng(seed + 2_000_003)
    validation_predictions, test_predictions = [], []

    def record() -> None:
        validation_predictions.append(completion.predict(
            model, rendered["validation"], data.task, model_name, class_map, device
        ))
        test_predictions.append(completion.predict(
            model, rendered["test"], data.task, model_name, class_map, device
        ))

    record()
    started = time.perf_counter()
    for epoch in range(1, int(training["epochs"]) + 1):
        model.train()
        order = order_rng.permutation(len(rendered["train"]))
        for start in range(0, len(order), batch):
            chosen = order[start : start + batch]
            xb = torch.from_numpy(rendered["train"][chosen]).to(device)
            if data.task == "classification":
                yb = torch.from_numpy(target[chosen].astype(np.int64)).to(device)
            else:
                yb = torch.from_numpy(target[chosen].astype(np.float32)).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = completion.loss_value(
                completion.forward(model, xb, model_name), yb, data.task, model_name
            )
            loss.backward()
            optimizer.step()
        if epoch in checkpoints[1:]:
            record()
    if len(validation_predictions) != len(checkpoints):
        raise AssertionError("checkpoint schedule was not fully recorded")
    return (
        np.asarray(validation_predictions, dtype=np.float32),
        np.asarray(test_predictions, dtype=np.float32),
        time.perf_counter() - started,
    )


def run_bundle(
    dataset: str, model_name: str, seed: int, device_name: str,
    output_dir: Path, force: bool = False,
) -> Path:
    config = json.loads(CONFIG_PATH.read_text())
    day5_config = json.loads(DAY5_CONFIG_PATH.read_text())
    if dataset not in config["datasets"] or model_name not in config["models"]:
        raise ValueError("bundle outside frozen matrix")
    if seed not in config["pilot_seeds"] + config["confirmation_seeds"]:
        raise ValueError("seed outside frozen menu")
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{dataset}__{model_name}__seed{seed}"
    artifact = output_dir / f"{stem}.npz"
    manifest_path = output_dir / f"{stem}.json"
    if artifact.exists() and manifest_path.exists() and not force:
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") == "complete":
            print(f"complete: {stem}", flush=True)
            return artifact

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(device_name)
    split_seed = int(config["split_seed"])
    data = completion.prepare(dataset, split_seed, day5_config)
    design = completion.views(data, day5_config)
    canonical_feature = design["feature"][0]
    canonical_category = design["category"][0]
    class_map = design["class"][0]
    output_width = 2 if data.task == "classification" else 1
    canonical_rendered = {
        part: completion.render(
            data, part, canonical_feature, canonical_category
        )[0] for part in PARTS
    }
    width = canonical_rendered["train"].shape[1]
    actions = [(0, 0)] + [
        (index, index if len(design["category"]) > 1 else 0)
        for index in range(1, int(config["nonidentity_views"]) + 1)
    ]
    all_validation, all_test, labels, elapsed = [], [], [], []
    maximum_initial_gap = 0.0
    transformed_target = class_map[data.y["train"]] if data.task == "classification" else data.y["train"]

    for precision in ("fp32", "iea64"):
        seed_all(seed)
        canonical_initial_model = initialize_model(
            model_name, width, output_width, seed, day5_config, device, precision
        )
        canonical_state = copy.deepcopy(canonical_initial_model.state_dict())
        for action_index, (feature_index, category_index) in enumerate(actions):
            rendered = {
                part: completion.render(
                    data, part, design["feature"][feature_index],
                    design["category"][category_index],
                )[0] for part in PARTS
            }
            coordinate_map = completion.render(
                data, "train", design["feature"][feature_index],
                design["category"][category_index],
            )[1]
            model = initialize_model(
                model_name, width, output_width, seed, day5_config, device, precision
            )
            if action_index == 0:
                model.load_state_dict(canonical_state)
            else:
                model.load_state_dict(completion.matched_state(
                    model_name, canonical_state, coordinate_map, class_map
                ))
            initial_canonical = completion.predict(
                canonical_initial_model, canonical_rendered["validation"], data.task,
                model_name, class_map, device,
            )
            initial_current = completion.predict(
                model, rendered["validation"], data.task, model_name, class_map, device
            )
            gap = float(np.max(np.abs(initial_canonical - initial_current)))
            maximum_initial_gap = max(maximum_initial_gap, gap)
            if gap > float(config["initial_match_tolerance"]):
                raise AssertionError(f"initial matched-function gap {gap} exceeds tolerance")
            validation, test, seconds = trajectory(
                model=model, rendered=rendered, target=transformed_target, data=data,
                model_name=model_name, class_map=class_map, seed=seed, config=config,
                device=device,
            )
            all_validation.append(validation)
            all_test.append(test)
            labels.append((precision, action_index, feature_index, category_index))
            elapsed.append(seconds)
            print(
                f"H1 {dataset}/{model_name}/seed={seed} {precision} "
                f"view={action_index}: {seconds:.2f}s gap0={gap:.3e}", flush=True,
            )

    np.savez_compressed(
        artifact,
        validation_predictions=np.asarray(all_validation, dtype=np.float32),
        test_predictions=np.asarray(all_test, dtype=np.float32),
        labels=np.asarray(labels, dtype="U24"),
        checkpoints=np.asarray(config["checkpoints"], dtype=np.int16),
        validation_y=data.y["validation"], test_y=data.y["test"],
    )
    manifest = {
        "status": "complete", "hypothesis": "H1_SAA", "dataset": dataset,
        "task": data.task, "model": model_name, "seed": seed,
        "split_seed": split_seed, "paths": len(labels),
        "epochs_per_path": int(config["training"]["epochs"]),
        "maximum_initial_gap": maximum_initial_gap,
        "path_wall_seconds": elapsed, "total_fit_seconds": float(sum(elapsed)),
        "config_sha256": sha256(CONFIG_PATH),
        "protocol_sha256": sha256(HERE / "HYPOTHESIS_01_PROTOCOL.md"),
        "day5_config_sha256": sha256(DAY5_CONFIG_PATH),
        "torch": torch.__version__, "device": device_name,
        "tf32": False, "deterministic_algorithms": True,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results" / "h1")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_bundle(args.dataset, args.model, args.seed, args.device, args.output_dir, args.force)


if __name__ == "__main__":
    main()
