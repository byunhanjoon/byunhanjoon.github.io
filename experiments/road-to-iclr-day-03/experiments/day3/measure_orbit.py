"""Parameter-matched TabM ensemble over mixed-measure numerical views."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import tabm
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .core import Prepared, loss_numpy, metric
from .mixed_measure_ple import (
    DAY1,
    fixed_ple_block,
    mixed_measure_block,
    subsample_dataset,
)
from .orbit_ensemble import _ensemble_numpy, _evaluate


ROOT = Path(__file__).resolve().parents[2]
DAY2 = ROOT.parent / "road-to-iclr-day-02"
sys.path.insert(0, str(DAY2))
sys.path.insert(0, str(DAY1))

import cross_dataset_models as legacy_models  # noqa: E402
import real_data_benchmark as benchmark  # noqa: E402


CONFIG_PATH = ROOT / "experiments/day3/configs/measure_orbit_preregistered.json"
RESULTS = ROOT / "results/day3/measure_orbit"
VIEW_NAMES = ("baseline_fixed_ple", "tail_reallocated_ple", "mixed_measure_ple")
PARTS = ("train", "val", "test")


def config() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text())


def _read(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def adaptive_budget(numeric_columns: int) -> int:
    cfg = config()["encoding"]
    if numeric_columns <= 0:
        return 0
    return max(
        int(cfg["minimum_budget_per_numeric_column"]),
        min(
            int(cfg["maximum_budget_per_numeric_column"]),
            int(cfg["maximum_total_ple_width"]) // numeric_columns,
        ),
    )


def build_views(dataset: benchmark.Dataset, seed: int) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, object], benchmark.EncodedDataset]:
    cfg = config()["encoding"]
    cache: dict[str, object] = {}
    schema = benchmark.encode_dataset(dataset, "schema", seed, 8, 128, 20.0, 1e-3, cache)
    if dataset.x_num is None:
        return ({name: schema.x for name in VIEW_NAMES}, {"budget": 0, "columns": {}}, schema)
    clean = benchmark._clean_numeric(dataset.x_num)
    budget = adaptive_budget(clean["train"].shape[1])
    blocks: dict[str, list[dict[str, np.ndarray]]] = {name: [] for name in VIEW_NAMES}
    metadata: dict[str, object] = {"budget": budget, "columns": {}}
    for column in range(clean["train"].shape[1]):
        base, base_meta = fixed_ple_block(clean, column, budget)
        tail, tail_meta = mixed_measure_block(
            clean,
            column,
            budget,
            int(cfg["maximum_atoms_per_column"]),
            int(cfg["minimum_nonatom_rows"]),
            include_atom_coordinates=False,
        )
        mixed, mixed_meta = mixed_measure_block(
            clean,
            column,
            budget,
            int(cfg["maximum_atoms_per_column"]),
            int(cfg["minimum_nonatom_rows"]),
            include_atom_coordinates=True,
        )
        blocks["baseline_fixed_ple"].append(base)
        blocks["tail_reallocated_ple"].append(tail)
        blocks["mixed_measure_ple"].append(mixed)
        metadata["columns"][column] = {"baseline": base_meta, "tail": tail_meta, "mixed": mixed_meta}
    views: dict[str, dict[str, np.ndarray]] = {}
    for name in VIEW_NAMES:
        numerical = {part: np.column_stack([block[part] for block in blocks[name]]).astype(np.float32) for part in PARTS}
        views[name] = {
            part: np.ascontiguousarray(np.column_stack((schema.x[part], numerical[part])), dtype=np.float32)
            for part in PARTS
        }
    widths = {name: value["train"].shape[1] for name, value in views.items()}
    if len(set(widths.values())) != 1:
        raise AssertionError(f"View widths differ: {widths}")
    return views, metadata, schema


class MeasureViewTabM(nn.Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        cfg = config()["training"]
        self.first = nn.Linear(input_size, int(cfg["latent_size"]))
        self.backbone = tabm.TabM.make(
            n_num_features=int(cfg["latent_size"]),
            cat_cardinalities=[],
            d_out=output_size,
            num_embeddings=None,
            n_blocks=int(cfg["n_blocks"]),
            d_block=int(cfg["d_block"]),
            dropout=float(cfg["dropout"]),
            k=len(config()["member_assignment"]),
        )

    def forward_members(self, member_features: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.first(member_features), None)


def _member_tensor(features: dict[str, torch.Tensor], assignment: list[str]) -> torch.Tensor:
    return torch.stack([features[name] for name in assignment], dim=1)


def _predict(
    model: MeasureViewTabM,
    views: dict[str, np.ndarray],
    assignment: list[str],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    rows = len(next(iter(views.values())))
    output = []
    with torch.inference_mode():
        for start in range(0, rows, batch_size):
            batch = {name: torch.from_numpy(values[start : start + batch_size]).to(device) for name, values in views.items()}
            output.append(model.forward_members(_member_tensor(batch, assignment)).float().cpu().numpy())
    return np.concatenate(output)


def train_cell(
    prepared: Prepared,
    views: dict[str, dict[str, np.ndarray]],
    arm: str,
    seed: int,
    device: str,
) -> dict[str, object]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cfg = config()["training"]
    resolved = torch.device(device)
    assignment = (
        ["baseline_fixed_ple"] * len(config()["member_assignment"])
        if arm == "baseline_tabm"
        else list(config()["member_assignment"])
    )
    model = MeasureViewTabM(prepared.x["train"].shape[1], prepared.n_classes if prepared.task == "multiclass" else 1).to(resolved)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    batch_size = int(cfg["large_batch_size"] if len(prepared.y["train"]) >= int(cfg["large_dataset_threshold"]) else cfg["batch_size"])
    tensors = [torch.from_numpy(views[name]["train"]) for name in VIEW_NAMES]
    loader = DataLoader(
        TensorDataset(*tensors, torch.from_numpy(prepared.y["train"])),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed + 70000),
        pin_memory=resolved.type == "cuda",
    )
    best_loss, best_epoch, stale, best_state = math.inf, 0, 0, None
    started = time.perf_counter()
    for epoch in range(1, int(cfg["max_epochs"]) + 1):
        model.train()
        for *feature_values, target in loader:
            batch = {name: value.to(resolved, non_blocking=True) for name, value in zip(VIEW_NAMES, feature_values)}
            target = target.to(resolved, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            members = model.forward_members(_member_tensor(batch, assignment))
            if prepared.task == "binclass":
                binary_target = target.float()
                loss = nn.functional.binary_cross_entropy_with_logits(members.squeeze(-1), binary_target[:, None].expand_as(members.squeeze(-1)))
            elif prepared.task == "multiclass":
                expanded = target.long()[:, None].expand(-1, members.shape[1])
                loss = nn.functional.cross_entropy(members.flatten(0, 1), expanded.flatten())
            else:
                continuous_target = target.float()
                loss = nn.functional.mse_loss(members.squeeze(-1), continuous_target[:, None].expand_as(members.squeeze(-1)))
            loss.backward()
            optimizer.step()
        val_members = _predict(model, {name: value["val"] for name, value in views.items()}, assignment, resolved, batch_size * 2)
        val_loss = loss_numpy(prepared.task, _ensemble_numpy(val_members, prepared.task), prepared.y["val"])
        if val_loss < best_loss:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if stale > int(cfg["patience"]):
            break
    if best_state is None:
        raise RuntimeError("No checkpoint")
    model.load_state_dict(best_state)
    result: dict[str, object] = {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch": best_epoch,
        "epochs_trained": epoch,
        "train_seconds": time.perf_counter() - started,
    }
    for part in ("val", "test"):
        members = _predict(model, {name: value[part] for name, value in views.items()}, assignment, resolved, batch_size * 2)
        result.update(_evaluate(prepared, members, prepared.y[part], part))
    return result


def run(args: argparse.Namespace) -> None:
    cfg = config()
    datasets = args.datasets or list(cfg["screen"]["datasets"])
    seeds = args.seeds or list(cfg["screen"]["seeds"])
    arms = args.arms or list(cfg["screen"]["arms"])
    rows: list[dict[str, object]] = list(_read(args.output))
    complete = {(row["dataset"], int(row["seed"]), row["arm"]) for row in rows}
    selected = [name for index, name in enumerate(datasets) if index % args.num_shards == args.shard]
    for dataset_name in selected:
        dataset = benchmark.load_dataset(DAY1 / "data", dataset_name)
        dataset = subsample_dataset(dataset, int(cfg["data"]["max_train_rows"]), int(cfg["data"]["max_eval_rows"]), int(cfg["data"]["sample_seed"]))
        for seed in seeds:
            views, metadata, schema = build_views(dataset, seed)
            prepared = Prepared(
                x=views["baseline_fixed_ple"],
                y=schema.y,
                task=dataset.task,
                n_classes=2 if dataset.task == "binclass" else 1,
                y_mean=schema.y_mean,
                y_scale=schema.y_scale,
                metadata={},
            )
            for arm in arms:
                key = (dataset_name, seed, arm)
                if key in complete:
                    continue
                result = train_cell(prepared, views, arm, seed, args.device)
                rows.append({
                    "hypothesis": "measure_orbit",
                    "dataset": dataset_name,
                    "task": dataset.task,
                    "seed": seed,
                    "arm": arm,
                    "budget_per_numeric": metadata["budget"],
                    "input_features": prepared.x["train"].shape[1],
                    "encoding_metadata": json.dumps(metadata, sort_keys=True, separators=(",", ":")),
                    **result,
                })
                complete.add(key)
                _write(args.output, rows)
                print(f"{dataset_name} s{seed} {arm} loss={float(result['test_proper_loss']):.6f} metric={float(result['test_metric']):.6f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--arms", nargs="+")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=RESULTS / "runs.csv")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
