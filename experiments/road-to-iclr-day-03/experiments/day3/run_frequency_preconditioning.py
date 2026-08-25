"""Frequency-only exact categorical reparameterization screen."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from .core import PARTS, base_schema, category_codes, combine, geometry, helmert, load_dataset, make_prepared, one_hot_codes, train_model


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "day3" / "frequency_preconditioning.csv"
STATS = ROOT / "results" / "day3" / "frequency_update_statistics.csv"


def read(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def blocks(dataset, gamma: float):
    assert dataset.x_cat is not None
    output, statistics = [], []
    for j in range(dataset.x_cat["train"].shape[1]):
        codes, levels = category_codes(dataset.x_cat, j)
        counts = np.bincount(codes["train"], minlength=len(levels)).astype(float)
        probability = counts / counts.sum()
        reference = float(np.exp(np.log(np.maximum(probability, 1e-12)).mean()))
        multiplier = np.clip((reference / np.maximum(probability, 1e-12)) ** (gamma / 2), 0.25, 10.0)
        state_basis = np.diag(multiplier) @ helmert(len(levels))
        block = {p: one_hot_codes(codes, len(levels))[p] @ state_basis for p in PARTS}
        # Global RMS matching isolates anisotropy from overall activation scale.
        rms = np.sqrt(np.mean(np.sum(block["train"] ** 2, axis=1)))
        baseline = {p: one_hot_codes(codes, len(levels))[p] @ helmert(len(levels)) for p in PARTS}
        baseline_rms = np.sqrt(np.mean(np.sum(baseline["train"] ** 2, axis=1)))
        block = {p: values * baseline_rms / max(rms, 1e-12) for p, values in block.items()}
        output.append(block)
        statistics.extend({"dataset": dataset.name, "column": j, "level": level, "count": int(count), "probability": float(prob), "gamma": gamma, "lr_or_activation_multiplier": float(mult), "analytical_update_count_per_epoch": int(count)} for level, count, prob, mult in zip(levels, counts, probability, multiplier))
    return output, statistics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["adult", "diamond"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    rows = read(OUT)
    complete = {(r["dataset"], int(r["seed"]), float(r["gamma"])) for r in rows}
    stats = []
    for name in args.datasets:
        dataset = load_dataset(name)
        schema = base_schema(dataset, include_cat=False)
        for gamma in (0.0, 0.25, 0.5):
            categorical, category_stats = blocks(dataset, gamma)
            stats.extend(category_stats)
            x = combine([schema, *categorical])
            for seed in args.seeds:
                if (name, seed, gamma) in complete:
                    continue
                fit, _ = train_model(make_prepared(dataset, x, {}), seed, args.device)
                row = {"experiment": "frequency_preconditioning", "intervention_class": "B", "dataset": name, "task": dataset.task, "model": "mlp", "optimizer": "AdamW", "weight_decay": 1e-4, "seed": seed, "representation": f"frequency_gamma_{gamma:g}", "regularizer": "standard", "gamma": gamma, "clip_min": 0.25, "clip_max": 10.0, "global_rms_matched": True, "split_fingerprint": dataset.split_fingerprint, **geometry(x["train"]), **fit}
                rows.append(row)
                write(OUT, rows)
                print(f"frequency {name} s{seed} gamma={gamma:g} metric={fit['test_metric']:.6f}", flush=True)
    write(STATS, stats)


if __name__ == "__main__":
    main()
