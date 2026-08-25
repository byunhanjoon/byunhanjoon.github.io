"""Fixed-budget PLE for numerical columns with atoms plus continuous support.

The baseline and proposed representation allocate exactly ``budget`` columns
per numerical feature.  Baseline PLE pads collapsed quantile intervals with
zeros.  Mixed-measure PLE uses target-free training frequencies to allocate
coordinates to empirical atoms, a non-atom gate, and a PLE fitted to the
remaining empirical measure.  A tail-reallocated arm isolates the effect of
moving knots away from atoms without adding atom indicators.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
DAY2 = ROOT.parent / "road-to-iclr-day-02"
DAY1 = ROOT.parent / "road-to-iclr-day-01"
sys.path.insert(0, str(DAY2))
sys.path.insert(0, str(DAY1))

import cross_dataset_models as models  # noqa: E402
import real_data_benchmark as benchmark  # noqa: E402
from residual_map_benchmark import subsample_dataset  # noqa: E402


CONFIG_PATH = ROOT / "experiments/day3/configs/mixed_measure_ple_preregistered.json"
RESULTS = ROOT / "results/day3/mixed_measure_ple"
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


def _ple(values: np.ndarray, knots: np.ndarray) -> np.ndarray:
    if len(knots) < 2:
        return np.empty((len(values), 0), dtype=np.float64)
    left, right = knots[:-1], knots[1:]
    widths = np.maximum(right - left, 1e-12)
    return np.clip((values[:, None] - left) / widths, 0.0, 1.0)


def _pad(block: np.ndarray, budget: int) -> np.ndarray:
    if block.shape[1] > budget:
        raise ValueError(f"Block width {block.shape[1]} exceeds budget {budget}")
    if block.shape[1] == budget:
        return block
    return np.column_stack((block, np.zeros((len(block), budget - block.shape[1]))))


def atom_values(
    train: np.ndarray,
    budget: int,
    maximum_atoms: int,
    minimum_nonatom_rows: int,
) -> np.ndarray:
    """Select target-free point masses that consume at least one quantile slot."""

    values, counts = np.unique(train, return_counts=True)
    threshold = max(2, math.ceil(len(train) / budget))
    order = np.lexsort((values, -counts))
    selected: list[float] = []
    removed = 0
    for index in order:
        count = int(counts[index])
        if count < threshold or len(selected) >= maximum_atoms:
            continue
        if len(train) - removed - count < minimum_nonatom_rows:
            continue
        selected.append(float(values[index]))
        removed += count
    return np.sort(np.asarray(selected, dtype=np.float64))


def fixed_ple_block(parts: dict[str, np.ndarray], column: int, budget: int) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    train = parts["train"][:, column]
    knots = np.unique(np.quantile(train, np.linspace(0.0, 1.0, budget + 1)))
    if len(knots) < 2:
        knots = np.asarray([knots[0], knots[0] + 1.0])
    output = {part: _pad(_ple(values[:, column], knots), budget).astype(np.float32) for part, values in parts.items()}
    return output, {"column": column, "atoms": [], "knots": int(len(knots)), "active_width": int(len(knots) - 1)}


def mixed_measure_block(
    parts: dict[str, np.ndarray],
    column: int,
    budget: int,
    maximum_atoms: int,
    minimum_nonatom_rows: int,
    *,
    include_atom_coordinates: bool,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    train = parts["train"][:, column]
    atoms = atom_values(train, budget, maximum_atoms, minimum_nonatom_rows)
    if not len(atoms):
        return fixed_ple_block(parts, column, budget)
    nonatom = ~np.isin(train, atoms)
    conditional_train = train[nonatom]
    if include_atom_coordinates:
        spline_budget = max(1, budget - len(atoms) - 1)
    else:
        spline_budget = budget
    knots = np.unique(np.quantile(conditional_train, np.linspace(0.0, 1.0, spline_budget + 1)))
    if len(knots) < 2:
        knots = np.asarray([knots[0], knots[0] + 1.0])
    output: dict[str, np.ndarray] = {}
    for part, matrix in parts.items():
        values = matrix[:, column]
        spline = _ple(values, knots)
        if include_atom_coordinates:
            atom_matrix = (values[:, None] == atoms[None, :]).astype(np.float64)
            is_nonatom = ~np.any(atom_matrix.astype(bool), axis=1)
            spline[~is_nonatom] = 0.0
            block = np.column_stack((atom_matrix, is_nonatom.astype(np.float64), spline))
        else:
            block = spline
        output[part] = _pad(block, budget).astype(np.float32)
    return output, {
        "column": column,
        "atoms": atoms.tolist(),
        "atom_rows": int((~nonatom).sum()),
        "nonatom_rows": int(nonatom.sum()),
        "knots": int(len(knots)),
        "active_width": int((len(atoms) + 1 if include_atom_coordinates else 0) + len(knots) - 1),
    }


def representations(dataset: benchmark.Dataset, seed: int) -> tuple[dict[str, benchmark.EncodedDataset], dict[str, object]]:
    cfg = config()["encoding"]
    budget = int(cfg["coordinate_budget_per_numeric_column"])
    maximum_atoms = int(cfg["maximum_atoms_per_column"])
    minimum_nonatom_rows = int(cfg["minimum_nonatom_rows"])
    cache: dict[str, object] = {}
    schema = benchmark.encode_dataset(dataset, "schema", seed, budget, 128, 20.0, 1e-3, cache)
    if dataset.x_num is None:
        return {name: schema for name in config()["representations"]}, {"numeric_columns": 0, "columns": {}}
    clean = benchmark._clean_numeric(dataset.x_num)
    arms: dict[str, list[dict[str, np.ndarray]]] = {name: [] for name in config()["representations"]}
    metadata: dict[str, object] = {"numeric_columns": clean["train"].shape[1], "columns": {}}
    for column in range(clean["train"].shape[1]):
        base, base_meta = fixed_ple_block(clean, column, budget)
        tail, tail_meta = mixed_measure_block(clean, column, budget, maximum_atoms, minimum_nonatom_rows, include_atom_coordinates=False)
        mixed, mixed_meta = mixed_measure_block(clean, column, budget, maximum_atoms, minimum_nonatom_rows, include_atom_coordinates=True)
        arms["baseline_fixed_ple"].append(base)
        arms["tail_reallocated_ple"].append(tail)
        arms["mixed_measure_ple"].append(mixed)
        metadata["columns"][column] = {"baseline": base_meta, "tail": tail_meta, "mixed": mixed_meta}
    encoded: dict[str, benchmark.EncodedDataset] = {}
    for name, blocks in arms.items():
        joined = {part: np.column_stack([block[part] for block in blocks]).astype(np.float32) for part in PARTS}
        encoded[name] = models._append_view(schema, joined, name, tuple(range(clean["train"].shape[1])))
    return encoded, metadata


def run(args: argparse.Namespace) -> None:
    cfg = config()
    datasets = args.datasets or list(cfg["datasets"])
    model_names = args.models or list(cfg["models"])
    seeds = args.seeds or list(cfg["seeds"])
    arms = args.arms or list(cfg["representations"])
    path = args.output
    rows: list[dict[str, object]] = list(_read(path))
    complete = {(row["dataset"], row["model"], int(row["seed"]), row["representation"]) for row in rows}
    selected = [name for index, name in enumerate(datasets) if index % args.num_shards == args.shard]
    for dataset_name in selected:
        dataset = benchmark.load_dataset(DAY1 / "data", dataset_name)
        dataset = subsample_dataset(
            dataset,
            int(cfg["data"]["max_train_rows"]),
            int(cfg["data"]["max_eval_rows"]),
            int(cfg["data"]["sample_seed"]),
        )
        for seed in seeds:
            variants, metadata = representations(dataset, seed)
            atom_columns = [
                str(column) for column, values in metadata["columns"].items()
                if values["mixed"]["atoms"]
            ]
            print(f"{dataset_name} seed={seed} atom_columns={';'.join(atom_columns) or '-'}", flush=True)
            for model_name in model_names:
                model_cfg = models.MODEL_CONFIGS[model_name]
                parameter_budget = models.baseline_parameter_count(
                    variants["baseline_fixed_ple"], model_name, model_cfg, int(cfg["training"]["ensemble_size"])
                )
                for arm in arms:
                    key = (dataset_name, model_name, seed, arm)
                    if key in complete:
                        continue
                    encoded = variants[arm]
                    output = models.train_model(
                        encoded,
                        model_name,
                        seed,
                        models.batch_size(dataset_name),
                        torch.device(args.device),
                        model_cfg,
                        int(cfg["training"]["ensemble_size"]),
                        int(cfg["training"]["max_epochs"]),
                        int(cfg["training"]["patience"]),
                        parameter_budget,
                    )
                    row: dict[str, object] = {
                        "hypothesis": "mixed_measure_ple",
                        "dataset": dataset_name,
                        "task": dataset.task,
                        "model": model_name,
                        "seed": seed,
                        "representation": arm,
                        "split_sample_seed": int(cfg["data"]["sample_seed"]),
                        "encoding_metadata": json.dumps(metadata, sort_keys=True, separators=(",", ":")),
                        **output.result,
                        **models._extra_metrics(dataset, encoded, output),
                    }
                    rows.append(row)
                    complete.add(key)
                    _write(path, rows)
                    print(f"  {model_name:<6} {arm:<24} test={float(output.result['test_score']):.6f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--arms", nargs="+")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=RESULTS / "runs.csv")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
