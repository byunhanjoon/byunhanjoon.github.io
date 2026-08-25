"""Prospective broad confirmation of validation-selected Measure-Orbit."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .broad_data import load_broad_dataset
from .broad_extension_data import load_extension_dataset
from .core import PARTS, base_schema, clean_numeric, make_prepared
from .measure_orbit import (
    RESULTS as SCREEN_RESULTS,
    VIEW_NAMES,
    adaptive_budget,
    config as method_config,
    fixed_ple_block,
    mixed_measure_block,
    train_cell,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "experiments/day3/configs/selective_measure_orbit_preregistered.json"
RESULTS = ROOT / "results/day3/selective_measure_orbit"


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


def load(name: str):
    return load_extension_dataset(name) if name.endswith("_extension") else load_broad_dataset(name)


def build_views(dataset, seed: int) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, object]]:
    schema = base_schema(dataset, seed)
    if dataset.x_num is None or dataset.x_num["train"].shape[1] == 0:
        return {name: schema for name in VIEW_NAMES}, {"budget": 0, "columns": {}}
    cfg = method_config()["encoding"]
    clean = clean_numeric(dataset.x_num)
    budget = adaptive_budget(clean["train"].shape[1])
    blocks: dict[str, list[dict[str, np.ndarray]]] = {name: [] for name in VIEW_NAMES}
    metadata: dict[str, object] = {"budget": budget, "columns": {}}
    for column in range(clean["train"].shape[1]):
        base, base_meta = fixed_ple_block(clean, column, budget)
        tail, tail_meta = mixed_measure_block(clean, column, budget, int(cfg["maximum_atoms_per_column"]), int(cfg["minimum_nonatom_rows"]), include_atom_coordinates=False)
        mixed, mixed_meta = mixed_measure_block(clean, column, budget, int(cfg["maximum_atoms_per_column"]), int(cfg["minimum_nonatom_rows"]), include_atom_coordinates=True)
        blocks["baseline_fixed_ple"].append(base)
        blocks["tail_reallocated_ple"].append(tail)
        blocks["mixed_measure_ple"].append(mixed)
        metadata["columns"][column] = {"baseline": base_meta, "tail": tail_meta, "mixed": mixed_meta}
    views = {}
    for name in VIEW_NAMES:
        numerical = {part: np.column_stack([block[part] for block in blocks[name]]).astype(np.float32) for part in PARTS}
        views[name] = {part: np.ascontiguousarray(np.column_stack((schema[part], numerical[part])), dtype=np.float32) for part in PARTS}
    if len({view["train"].shape[1] for view in views.values()}) != 1:
        raise AssertionError("Confirmation view widths differ")
    return views, metadata


def run(args: argparse.Namespace) -> None:
    cfg = config()
    all_datasets = list(cfg["confirmation"]["broad_datasets"]) + list(cfg["confirmation"]["extension_datasets"])
    datasets = args.datasets or all_datasets
    seeds = args.seeds or list(cfg["confirmation"]["seeds"])
    arms = args.arms or list(cfg["confirmation"]["candidate_arms"])
    rows: list[dict[str, object]] = list(_read(args.output))
    complete = {(row["dataset"], int(row["seed"]), row["arm"]) for row in rows}
    selected = [name for index, name in enumerate(datasets) if index % args.num_shards == args.shard]
    for dataset_name in selected:
        dataset = load(dataset_name)
        for seed in seeds:
            views, metadata = build_views(dataset, seed)
            prepared = make_prepared(dataset, views["baseline_fixed_ple"], {})
            for arm in arms:
                key = (dataset_name, seed, arm)
                if key in complete:
                    continue
                result = train_cell(prepared, views, arm, seed, args.device)
                rows.append({
                    "hypothesis": "selective_measure_orbit",
                    "dataset": dataset_name,
                    "task": dataset.task,
                    "seed": seed,
                    "arm": arm,
                    "budget_per_numeric": metadata["budget"],
                    "input_features": prepared.x["train"].shape[1],
                    "split_fingerprint": dataset.split_fingerprint,
                    **result,
                })
                complete.add(key)
                _write(args.output, rows)
                print(f"{dataset_name} s{seed} {arm} val={float(result['val_proper_loss']):.6f} test={float(result['test_proper_loss']):.6f}", flush=True)


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
