"""Frozen modern-neural completion panel for the Day-5 OrbitCover program."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tabm
import torch
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from torch import nn

import sys

HERE = Path(__file__).resolve().parent
DAY3 = HERE.parent / "road-to-iclr-day-03"
sys.path.insert(0, str(DAY3))

from experiments.day3.broad_models import DenseStemFTTransformer  # noqa: E402
from experiments.day3.core import MLP, ResNet  # noqa: E402


CONFIG = HERE / "completion_config.json"
PARTS = ("train", "validation", "test")


@dataclass
class Prepared:
    name: str
    task: str
    x_num: dict[str, np.ndarray]
    x_cat: dict[str, np.ndarray]
    y: dict[str, np.ndarray]
    cardinalities: list[int]
    y_mean: float
    y_scale: float


class CompletionTabM(nn.Module):
    def __init__(self, input_size: int, output_size: int, latent_size: int, members: int) -> None:
        super().__init__()
        self.first = nn.Linear(input_size, latent_size)
        self.backbone = tabm.TabM.make(
            n_num_features=latent_size, cat_cardinalities=[], d_out=output_size,
            num_embeddings=None, n_blocks=2, d_block=192, dropout=0.1, k=members,
        )

    def forward_members(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.first(x), None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_members(x).mean(dim=1)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def optional_array(path: Path, rows: int) -> np.ndarray:
    if not path.exists():
        return np.empty((rows, 0), dtype=object)
    return np.asarray(np.load(path, allow_pickle=True), dtype=object)


def raw_local(name: str, config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = Path(config["data_root"]) / name
    numeric, categorical, targets = [], [], []
    for part in ("train", "val", "test"):
        y = np.asarray(np.load(root / f"y_{part}.npy"))
        numeric.append(np.asarray(np.load(root / f"N_{part}.npy"), dtype=np.float64))
        categorical.append(optional_array(root / f"C_{part}.npy", len(y)))
        targets.append(y)
    return np.concatenate(numeric), np.concatenate(categorical), np.concatenate(targets)


def raw_openml(name: str, config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bunch = fetch_openml(data_id=int(config["openml_ids"][name]), as_frame=True, parser="auto")
    frame = bunch.data.copy()
    numerical_columns = [column for column in frame if pd.api.types.is_numeric_dtype(frame[column].dtype)]
    categorical_columns = [column for column in frame if column not in numerical_columns]
    numeric = frame[numerical_columns].to_numpy(dtype=np.float64)
    categorical = np.empty((len(frame), len(categorical_columns)), dtype=object)
    for index, column in enumerate(categorical_columns):
        categorical[:, index] = frame[column].astype("string").fillna("__MISSING__").to_numpy(dtype=str)
    return numeric, categorical, np.asarray(bunch.target)


def capped(indices: np.ndarray, target: np.ndarray, maximum: int, task: str, seed: int) -> np.ndarray:
    if len(indices) <= maximum:
        return np.sort(indices)
    stratify = target[indices] if task == "classification" else None
    chosen, _ = train_test_split(
        indices, train_size=maximum, random_state=seed, shuffle=True, stratify=stratify
    )
    return np.sort(chosen)


def split_indices(target: np.ndarray, task: str, seed: int, caps: dict[str, int]) -> dict[str, np.ndarray]:
    rows = np.arange(len(target))
    stratify = target if task == "classification" else None
    train_val, test = train_test_split(rows, test_size=0.2, random_state=seed, stratify=stratify)
    second = target[train_val] if task == "classification" else None
    train, validation = train_test_split(
        train_val, test_size=0.25, random_state=seed + 1, stratify=second
    )
    return {
        "train": capped(train, target, int(caps["train"]), task, seed + 11),
        "validation": capped(validation, target, int(caps["validation"]), task, seed + 12),
        "test": capped(test, target, int(caps["test"]), task, seed + 13),
    }


def prepare(name: str, split_seed: int, config: dict[str, Any]) -> Prepared:
    task = config["dataset_tasks"][name]
    numeric, categorical, raw_target = (
        raw_openml(name, config) if name.startswith("openml-") else raw_local(name, config)
    )
    if task == "classification":
        _, target = np.unique(raw_target.astype(str), return_inverse=True)
        if len(np.unique(target)) != 2:
            raise ValueError(f"{name}: completion panel currently requires binary classification")
        target = target.astype(np.int64)
    else:
        target = raw_target.astype(np.float64)
    indices = split_indices(target, task, split_seed, config["subsample"])
    train_numeric = numeric[indices["train"]]
    medians = np.nanmedian(train_numeric, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    imputed = {}
    for part in PARTS:
        values = numeric[indices[part]].copy()
        bad = ~np.isfinite(values)
        if bad.any():
            values[bad] = medians[np.where(bad)[1]]
        imputed[part] = values
    means = imputed["train"].mean(axis=0)
    scales = imputed["train"].std(axis=0)
    scales = np.where(scales > 0, scales, 1.0)
    x_num = {part: np.ascontiguousarray((imputed[part] - means) / scales, dtype=np.float32) for part in PARTS}
    x_cat = {part: np.empty((len(indices[part]), categorical.shape[1]), dtype=np.int64) for part in PARTS}
    cardinalities = []
    for column in range(categorical.shape[1]):
        training = pd.Series(categorical[indices["train"], column]).fillna("__MISSING__").astype(str)
        levels = sorted(training.unique().tolist())
        mapping = {value: index for index, value in enumerate(levels)}
        cardinalities.append(len(levels))
        for part in PARTS:
            values = pd.Series(categorical[indices[part], column]).fillna("__MISSING__").astype(str)
            x_cat[part][:, column] = np.asarray([mapping.get(value, -1) for value in values], dtype=np.int64)
    if task == "regression":
        y_mean = float(target[indices["train"]].mean())
        y_scale = float(target[indices["train"]].std()) or 1.0
        y = {part: np.ascontiguousarray((target[indices[part]] - y_mean) / y_scale, dtype=np.float32) for part in PARTS}
    else:
        y_mean, y_scale = 0.0, 1.0
        y = {part: np.ascontiguousarray(target[indices[part]], dtype=np.int64) for part in PARTS}
    return Prepared(name, task, x_num, x_cat, y, cardinalities, y_mean, y_scale)


def views(data: Prepared, config: dict[str, Any]) -> dict[str, list[Any]]:
    seed = int(config["view_seed"]) + sum(data.name.encode())
    rng = np.random.default_rng(seed)
    fields = data.x_num["train"].shape[1] + data.x_cat["train"].shape[1]
    feature = [np.arange(fields)]
    while len(feature) < int(config["factor_levels"]["feature"]):
        candidate = rng.permutation(fields)
        if not any(np.array_equal(candidate, old) for old in feature):
            feature.append(candidate)
    category = [[np.arange(size) for size in data.cardinalities]]
    available_category_maps = math.prod(math.factorial(size) for size in data.cardinalities)
    target_category_levels = min(
        int(config["factor_levels"]["category"]), available_category_maps
    ) if data.cardinalities else 1
    while len(category) < target_category_levels:
        candidate = [rng.permutation(size) for size in data.cardinalities]
        if not any(all(np.array_equal(a, b) for a, b in zip(candidate, old)) for old in category):
            category.append(candidate)
    classes = [np.arange(2), np.asarray([1, 0])] if data.task == "classification" else [np.asarray([0])]
    return {"feature": feature, "category": category, "class": classes}


def canonical_offsets(data: Prepared) -> list[int]:
    widths = [1] * data.x_num["train"].shape[1] + [size + 1 for size in data.cardinalities]
    return list(np.cumsum([0, *widths]))


def render(data: Prepared, part: str, feature: np.ndarray, category: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    n_num = data.x_num[part].shape[1]
    offsets = canonical_offsets(data)
    blocks, coordinate_map = [], []
    for field in feature:
        field = int(field)
        if field < n_num:
            blocks.append(data.x_num[part][:, field : field + 1])
            coordinate_map.append(offsets[field])
        else:
            cat = field - n_num
            cardinality = data.cardinalities[cat]
            mapping = category[cat]
            transformed = np.full(len(data.x_cat[part]), cardinality, dtype=np.int64)
            known = data.x_cat[part][:, cat] >= 0
            transformed[known] = mapping[data.x_cat[part][known, cat]]
            block = np.zeros((len(transformed), cardinality + 1), dtype=np.float32)
            block[np.arange(len(transformed)), transformed] = 1.0
            blocks.append(block)
            inverse = np.argsort(mapping)
            coordinate_map.extend((offsets[field] + inverse).tolist())
            coordinate_map.append(offsets[field] + cardinality)
    return np.ascontiguousarray(np.concatenate(blocks, axis=1), dtype=np.float32), np.asarray(coordinate_map)


def make_model(name: str, width: int, output: int, config: dict[str, Any]) -> nn.Module:
    training = config["training"]
    if name == "mlp":
        return MLP(width, output, int(training["mlp_resnet_width"]), int(training["depth"]), float(training["dropout"]))
    if name == "resnet":
        return ResNet(width, output, int(training["mlp_resnet_width"]), int(training["depth"]), float(training["dropout"]))
    if name == "ft_transformer":
        return DenseStemFTTransformer(width, output, int(training["ft_d_token"]), int(training["ft_tokens"]))
    if name == "tabm":
        return CompletionTabM(
            width, output, int(training["tabm_latent"]), int(training["tabm_members"])
        )
    raise ValueError(name)


def initialize(name: str, width: int, output: int, init_seed: int, config: dict[str, Any], device: torch.device) -> nn.Module:
    random.seed(init_seed)
    np.random.seed(init_seed)
    torch.manual_seed(init_seed)
    torch.cuda.manual_seed_all(init_seed)
    return make_model(name, width, output, config).to(device)


def output_keys(name: str, state: dict[str, torch.Tensor]) -> tuple[str, str]:
    if name == "mlp":
        weights = [key for key, value in state.items() if key.startswith("network.") and key.endswith(".weight") and value.ndim == 2]
        weight = weights[-1]
        return weight, weight.replace("weight", "bias")
    if name == "resnet":
        return "output.2.weight", "output.2.bias"
    if name == "ft_transformer":
        return "backbone.output.linear.weight", "backbone.output.linear.bias"
    if name == "tabm":
        return "backbone.output.weight", "backbone.output.bias"
    raise ValueError(name)


def matched_state(
    name: str,
    canonical: dict[str, torch.Tensor],
    coordinate_map: np.ndarray,
    class_map: np.ndarray,
) -> dict[str, torch.Tensor]:
    state = copy.deepcopy(canonical)
    index = torch.as_tensor(coordinate_map, dtype=torch.long, device=canonical["first.weight"].device)
    state["first.weight"] = canonical["first.weight"][:, index].clone()
    if name == "mlp":
        state["network.0.weight"] = state["first.weight"]
    if len(class_map) == 2:
        weight_key, bias_key = output_keys(name, state)
        target = torch.as_tensor(class_map, dtype=torch.long, device=canonical[bias_key].device)
        if name == "tabm":
            state[weight_key][..., target] = canonical[weight_key]
            state[bias_key][..., target] = canonical[bias_key]
        else:
            state[weight_key][target] = canonical[weight_key]
            state[bias_key][target] = canonical[bias_key]
    return state


def forward(model: nn.Module, x: torch.Tensor, name: str) -> torch.Tensor:
    if name == "tabm":
        return model.forward_members(x)
    return model(x)


def loss_value(prediction: torch.Tensor, target: torch.Tensor, task: str, name: str) -> torch.Tensor:
    if name == "tabm":
        if task == "classification":
            expanded = target[:, None].expand(-1, prediction.shape[1])
            return nn.functional.cross_entropy(prediction.flatten(0, 1), expanded.flatten())
        expanded = target[:, None].expand(-1, prediction.shape[1])
        return nn.functional.mse_loss(prediction.squeeze(-1), expanded)
    if task == "classification":
        return nn.functional.cross_entropy(prediction, target)
    return nn.functional.mse_loss(prediction.squeeze(-1), target)


def predict(model: nn.Module, x: np.ndarray, task: str, name: str, class_map: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(x), 1024):
            raw = forward(model, torch.from_numpy(x[start : start + 1024]).to(device), name)
            if name == "tabm":
                raw = raw.softmax(-1).mean(1) if task == "classification" else raw.mean(1)
            elif task == "classification":
                raw = raw.softmax(-1)
            outputs.append(raw.cpu().numpy())
    values = np.concatenate(outputs)
    return values[:, class_map] if task == "classification" else values.reshape(-1, 1)


def fit(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    task: str,
    name: str,
    order_seed: int,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[float, int]:
    training = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"])
    )
    batch = int(training["batch_size"])
    torch.manual_seed(order_seed)
    torch.cuda.manual_seed_all(order_seed)
    generator = np.random.default_rng(order_seed)
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(int(training["epochs"])):
        model.train()
        order = generator.permutation(len(x))
        for start in range(0, len(x), batch):
            chosen = order[start : start + batch]
            xb = torch.from_numpy(x[chosen]).to(device)
            if task == "classification":
                yb = torch.from_numpy(y[chosen].astype(np.int64)).to(device)
            else:
                yb = torch.from_numpy(y[chosen].astype(np.float32)).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_value(forward(model, xb, name), yb, task, name)
            loss.backward()
            optimizer.step()
    elapsed = time.perf_counter() - started
    peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    return elapsed, peak


def all_actions(data: Prepared, config: dict[str, Any]) -> list[tuple[int, int, int, int, int]]:
    current = views(data, config)
    return list(np.ndindex(
        len(current["feature"]), len(current["category"]), len(current["class"]),
        len(config["init_seeds"]), len(config["order_seeds"]),
    ))


def selected_actions(data: Prepared, config: dict[str, Any], mode: str, split_seed: int, model: str) -> list[tuple[int, int, int, int, int]]:
    actions = all_actions(data, config)
    if mode in {"exact", "broad"}:
        return actions
    raise ValueError(mode)


def run_cell(dataset: str, model_name: str, split_seed: int, mode: str, device_name: str, config: dict[str, Any], output: Path) -> None:
    if mode == "exact" and (dataset not in config["exact_datasets"] or split_seed != config["split_seeds"][0]):
        raise ValueError("exact mode restricted to frozen subset/first split")
    device = torch.device(device_name)
    data = prepare(dataset, split_seed, config)
    design = views(data, config)
    actions = selected_actions(data, config, mode, split_seed, model_name)
    first_x, _ = render(data, "train", design["feature"][0], design["category"][0])
    output_dim = 2 if data.task == "classification" else 1
    validation = np.empty((len(actions), len(data.y["validation"]), output_dim), dtype=np.float32)
    test = np.empty((len(actions), len(data.y["test"]), output_dim), dtype=np.float32)
    telemetry = []
    for action_index, (fi, ci, li, ii, oi) in enumerate(actions):
        class_map = design["class"][li]
        rendered = {}
        coordinate_map = None
        for part in PARTS:
            rendered[part], current_map = render(data, part, design["feature"][fi], design["category"][ci])
            if coordinate_map is None:
                coordinate_map = current_map
            elif not np.array_equal(coordinate_map, current_map):
                raise AssertionError("coordinate map changed across splits")
        init_seed = int(config["init_seeds"][ii])
        order_seed = int(config["order_seeds"][oi])
        model = initialize(model_name, first_x.shape[1], output_dim, init_seed, config, device)
        transformed_y = class_map[data.y["train"]] if data.task == "classification" else data.y["train"]
        elapsed, peak = fit(model, rendered["train"], transformed_y, data.task, model_name, order_seed, config, device)
        validation[action_index] = predict(model, rendered["validation"], data.task, model_name, class_map, device)
        test[action_index] = predict(model, rendered["test"], data.task, model_name, class_map, device)
        telemetry.append({
            "action": action_index, "feature": fi, "category": ci, "class": li,
            "init": ii, "order": oi, "wall_seconds": elapsed, "peak_device_bytes": peak,
        })
        print(f"{dataset} {model_name} split={split_seed} {mode} {action_index + 1}/{len(actions)}", flush=True)
    stem = f"{dataset}__{model_name}__split{split_seed}__{mode}"
    np.savez_compressed(
        output / f"{stem}.npz", validation_predictions=validation, test_predictions=test,
        validation_y=data.y["validation"], test_y=data.y["test"],
        actions=np.asarray(actions, dtype=np.int16), y_mean=data.y_mean, y_scale=data.y_scale,
    )
    manifest = {
        "status": "complete", "dataset": dataset, "task": data.task, "model": model_name,
        "split_seed": split_seed, "mode": mode, "actions": len(actions),
        "represented_fits": len(actions), "rows": {part: len(data.y[part]) for part in PARTS},
        "features": {"numerical": data.x_num["train"].shape[1], "categorical": data.x_cat["train"].shape[1], "dense": first_x.shape[1]},
        "cardinalities": data.cardinalities, "device": device_name,
        "wall_seconds": float(sum(item["wall_seconds"] for item in telemetry)),
        "maximum_peak_device_bytes": max(item["peak_device_bytes"] for item in telemetry),
        "protocol_sha256": config["protocol_sha256"],
        "package_versions": {"torch": torch.__version__},
        "telemetry": telemetry,
    }
    (output / f"{stem}.json").write_text(json.dumps(manifest, indent=2) + "\n")


def run_matched(dataset: str, model_name: str, device_name: str, config: dict[str, Any], output: Path) -> None:
    split_seed = int(config["split_seeds"][0])
    device = torch.device(device_name)
    data = prepare(dataset, split_seed, config)
    design = views(data, config)
    first_x, _ = render(data, "train", design["feature"][0], design["category"][0])
    output_dim = 2 if data.task == "classification" else 1
    init_seed = int(config["init_seeds"][0]); order_seed = int(config["order_seeds"][0])
    canonical_model = initialize(model_name, first_x.shape[1], output_dim, init_seed, config, device)
    canonical_state = copy.deepcopy(canonical_model.state_dict())
    schema_actions = list(np.ndindex(4, 4, len(design["class"])))
    predictions = np.empty((2, len(schema_actions), len(data.y["test"]), output_dim), dtype=np.float32)
    initial_gaps = []
    telemetry = []
    for action_index, (fi, ci, li) in enumerate(schema_actions):
        class_map = design["class"][li]
        rendered = {part: render(data, part, design["feature"][fi], design["category"][ci])[0] for part in PARTS}
        coordinate_map = render(data, "train", design["feature"][fi], design["category"][ci])[1]
        for arm_index, arm in enumerate(("ordinary", "matched")):
            model = initialize(model_name, first_x.shape[1], output_dim, init_seed, config, device)
            if arm == "matched":
                model.load_state_dict(matched_state(model_name, canonical_state, coordinate_map, class_map))
            canonical_initial_model = initialize(model_name, first_x.shape[1], output_dim, init_seed, config, device)
            canonical_input = render(data, "validation", design["feature"][0], design["category"][0])[0]
            canonical_initial = predict(canonical_initial_model, canonical_input, data.task, model_name, design["class"][0], device)
            current_initial = predict(model, rendered["validation"], data.task, model_name, class_map, device)
            gap = float(np.max(np.abs(canonical_initial - current_initial))) if arm == "matched" else float("nan")
            if arm == "matched" and gap > float(config["initial_match_tolerance"]):
                raise AssertionError(f"matched initial gap {gap} exceeds tolerance")
            initial_gaps.append(gap)
            transformed_y = class_map[data.y["train"]] if data.task == "classification" else data.y["train"]
            elapsed, peak = fit(
                model, rendered["train"], transformed_y, data.task, model_name,
                order_seed, config, device,
            )
            predictions[arm_index, action_index] = predict(model, rendered["test"], data.task, model_name, class_map, device)
            telemetry.append({
                "action": action_index, "feature": fi, "category": ci, "class": li,
                "arm": arm, "wall_seconds": elapsed, "peak_device_bytes": peak,
                "initial_gap": gap,
            })
        print(f"matched {dataset} {model_name} {action_index + 1}/{len(schema_actions)}", flush=True)
    stem = f"{dataset}__{model_name}__matched"
    np.savez_compressed(output / f"{stem}.npz", test_predictions=predictions, test_y=data.y["test"], actions=np.asarray(schema_actions))
    manifest = {
        "status": "complete", "dataset": dataset, "task": data.task, "model": model_name,
        "schema_actions": len(schema_actions), "represented_fits": 2 * len(schema_actions),
        "maximum_matched_initial_gap": float(np.nanmax(initial_gaps)), "protocol_sha256": config["protocol_sha256"],
        "wall_seconds": float(sum(item["wall_seconds"] for item in telemetry)),
        "maximum_peak_device_bytes": max(item["peak_device_bytes"] for item in telemetry),
        "telemetry": telemetry,
    }
    (output / f"{stem}.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--split-seed", type=int)
    parser.add_argument("--mode", choices=("broad", "exact", "matched"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results" / "completion_neural")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if digest(HERE / config["protocol"]) != config["protocol_sha256"]:
        raise AssertionError("completion protocol hash mismatch")
    if args.dataset not in config["datasets"] or args.model not in config["models"]:
        raise ValueError("dataset/model outside frozen config")
    torch.set_num_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "matched":
        if args.dataset not in config["exact_datasets"]:
            raise ValueError("matched mode restricted to exact subset")
        run_matched(args.dataset, args.model, args.device, config, args.output_dir)
    else:
        split_seed = args.split_seed if args.split_seed is not None else int(config["split_seeds"][0])
        if split_seed not in config["split_seeds"]:
            raise ValueError("split seed outside frozen config")
        run_cell(args.dataset, args.model, split_seed, args.mode, args.device, config, args.output_dir)


if __name__ == "__main__":
    main()
