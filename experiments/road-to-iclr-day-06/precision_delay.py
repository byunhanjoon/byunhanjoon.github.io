"""H2 precision-indexed semantic trajectory experiment."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn

import semantic_arithmetic as h1

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "hypothesis_02_config.json"
PARTS = ("train", "validation", "test")
DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float64": torch.float64,
}


class InterfaceLinear(nn.Linear):
    def __init__(self, *args, accumulation_dtype: torch.dtype, **kwargs):
        super().__init__(*args, **kwargs)
        self.accumulation_dtype = accumulation_dtype

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        result = nn.functional.linear(
            value.to(self.accumulation_dtype),
            self.weight.to(self.accumulation_dtype),
            None if self.bias is None else self.bias.to(self.accumulation_dtype),
        )
        return result.to(torch.float32)


def convert(model: nn.Module, model_name: str, precision: str) -> nn.Module:
    if precision == "float32":
        return model
    old = model.first
    replacement = InterfaceLinear(
        old.in_features, old.out_features, bias=old.bias is not None,
        device=old.weight.device, dtype=old.weight.dtype,
        accumulation_dtype=DTYPES[precision],
    )
    with torch.no_grad():
        replacement.weight.copy_(old.weight)
        if old.bias is not None:
            replacement.bias.copy_(old.bias)
    model.first = replacement
    if model_name == "mlp":
        model.network[0] = replacement
    return model


def initialize(
    model_name: str, width: int, output_width: int, seed: int,
    day5_config: dict[str, Any], device: torch.device, precision: str,
) -> nn.Module:
    model = h1.completion.initialize(
        model_name, width, output_width, seed, day5_config, device
    )
    return convert(model, model_name, precision)


def run_bundle(
    dataset: str, model_name: str, seed: int, device_name: str,
    output_dir: Path, force: bool = False,
) -> Path:
    config = json.loads(CONFIG_PATH.read_text())
    day5_config = json.loads(h1.DAY5_CONFIG_PATH.read_text())
    if dataset not in config["datasets"] or model_name not in config["models"]:
        raise ValueError("bundle outside frozen H2 matrix")
    if seed not in config["seeds"]:
        raise ValueError("seed outside frozen H2 menu")
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{dataset}__{model_name}__seed{seed}"
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
    all_validation, all_test, labels, elapsed = [], [], [], []
    for precision in config["precisions"]:
        h1.seed_all(seed)
        canonical_initial = initialize(
            model_name, width, output_width, seed, day5_config, device, precision
        )
        canonical_state = copy.deepcopy(canonical_initial.state_dict())
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
            model = initialize(
                model_name, width, output_width, seed, day5_config, device, precision
            )
            if action_index == 0:
                model.load_state_dict(canonical_state)
            else:
                model.load_state_dict(h1.completion.matched_state(
                    model_name, canonical_state, coordinate_map, class_map
                ))
            validation, test, seconds = h1.trajectory(
                model=model, rendered=rendered, target=transformed_target, data=data,
                model_name=model_name, class_map=class_map, seed=seed, config=config,
                device=device,
            )
            all_validation.append(validation); all_test.append(test)
            labels.append((precision, action_index, feature_index, category_index))
            elapsed.append(seconds)
            print(
                f"H2 {dataset}/{model_name}/seed={seed} {precision} "
                f"view={action_index}: {seconds:.2f}s", flush=True,
            )
    np.savez_compressed(
        artifact, validation_predictions=np.asarray(all_validation, dtype=np.float32),
        test_predictions=np.asarray(all_test, dtype=np.float32),
        labels=np.asarray(labels, dtype="U24"),
        checkpoints=np.asarray(config["checkpoints"], dtype=np.int16),
        validation_y=data.y["validation"], test_y=data.y["test"],
    )
    manifest = {
        "status": "complete", "hypothesis": "H2_PRECISION_DELAY",
        "dataset": dataset, "task": data.task, "model": model_name,
        "seed": seed, "paths": len(labels),
        "total_fit_seconds": float(sum(elapsed)), "path_wall_seconds": elapsed,
        "config_sha256": hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(),
        "protocol_sha256": hashlib.sha256(
            (HERE / "HYPOTHESIS_02_PROTOCOL.md").read_bytes()
        ).hexdigest(),
        "torch": torch.__version__, "device": device_name,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results" / "h2")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_bundle(args.dataset, args.model, args.seed, args.device, args.output_dir, args.force)


if __name__ == "__main__":
    main()
