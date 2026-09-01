#!/usr/bin/env python3
"""Zero-shot natural-label bridge for the static projective checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import run_projective as base  # noqa: E402


OUT = HERE / "results" / "natural_anchor"
CONFIG = json.loads((ROOT / "experiments" / "final_closure" / "final_closure_config.json").read_text())
DATASETS = (
    "fremtpl_claim_count", "kdd17_stock_return", "openml-abalone-183",
    "openml-kin8nm-189", "openml-pol-201", "openml-puma32h-308",
)
SPLITS = tuple(int(value) for value in CONFIG["split_seeds"])
QUERY_FAMILIES = ("point", "dense", "scaled_dense")
EPISODES = 1_536


def protocol_hash() -> str:
    return hashlib.sha256((HERE / "NATURAL_ANCHOR_PROTOCOL.md").read_bytes()).hexdigest()


def raw_data(name: str) -> tuple[np.ndarray, np.ndarray]:
    if name.startswith("openml-"):
        bunch = fetch_openml(
            data_id=int(CONFIG["openml_ids"][name]), as_frame=True, parser="auto"
        )
        frame = bunch.data
        numerical = [column for column in frame if pd.api.types.is_numeric_dtype(frame[column].dtype)]
        x = frame[numerical].to_numpy(dtype=np.float64)
        y = pd.to_numeric(pd.Series(np.asarray(bunch.target)), errors="raise").to_numpy(np.float64)
        return x, y
    root = Path(CONFIG["data_root"]) / name
    x = np.concatenate(
        [np.asarray(np.load(root / f"N_{part}.npy"), dtype=np.float64) for part in ("train", "val", "test")]
    )
    y = np.concatenate(
        [np.asarray(np.load(root / f"y_{part}.npy"), dtype=np.float64) for part in ("train", "val", "test")]
    )
    return x, y


def capped(indices: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    if len(indices) <= maximum:
        return np.sort(indices)
    chosen, _ = train_test_split(indices, train_size=maximum, random_state=seed, shuffle=True)
    return np.sort(chosen)


def prepare(name: str, split: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw_x, raw_y = raw_data(name)
    rows = np.arange(len(raw_y))
    train_validation, test = train_test_split(rows, test_size=0.2, random_state=split)
    train, _ = train_test_split(train_validation, test_size=0.25, random_state=split + 1)
    train = capped(train, 2_048, split + 11)
    test = capped(test, 512, split + 13)
    median = np.nanmedian(raw_x[train], axis=0)
    median = np.where(np.isfinite(median), median, 0.0)

    def impute(values: np.ndarray) -> np.ndarray:
        values = values.copy()
        bad = ~np.isfinite(values)
        if bad.any():
            values[bad] = median[np.where(bad)[1]]
        return values

    train_x = impute(raw_x[train])
    test_x = impute(raw_x[test])
    mean = train_x.mean(axis=0)
    scale = np.where(train_x.std(axis=0) > 0, train_x.std(axis=0), 1.0)
    train_x = (train_x - mean) / scale
    test_x = (test_x - mean) / scale
    if train_x.shape[1] > base.DIM:
        pca = PCA(n_components=base.DIM, svd_solver="full").fit(train_x)
        train_x = pca.transform(train_x)
        test_x = pca.transform(test_x)
    elif train_x.shape[1] < base.DIM:
        padding = base.DIM - train_x.shape[1]
        train_x = np.pad(train_x, ((0, 0), (0, padding)))
        test_x = np.pad(test_x, ((0, 0), (0, padding)))
    coordinate_scale = np.where(train_x.std(axis=0) > 0, train_x.std(axis=0), 1.0)
    train_x = np.clip(train_x / coordinate_scale, -5.0, 5.0).astype(np.float32)
    test_x = np.clip(test_x / coordinate_scale, -5.0, 5.0).astype(np.float32)
    y_mean = raw_y[train].mean()
    y_scale = raw_y[train].std() or 1.0
    train_y = ((raw_y[train] - y_mean) / y_scale).astype(np.float32)
    test_y = ((raw_y[test] - y_mean) / y_scale).astype(np.float32)
    return train_x, train_y, test_x, test_y


@torch.no_grad()
def evaluate(
    projective: base.ProjectiveModel,
    direct: base.DirectModel,
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    dataset: str,
    split: int,
    seed: int,
    query_family: str,
    device: torch.device,
) -> list[dict]:
    train_x, train_y, test_x, test_y = [torch.from_numpy(value).to(device) for value in arrays]
    generator_seed = int(
        hashlib.sha256(f"natural|{dataset}|{split}|{seed}|{query_family}".encode()).hexdigest()[:15], 16
    )
    generator = torch.Generator(device=device).manual_seed(generator_seed)
    sums = {
        "projective": {"nll": 0.0, "sq": 0.0, "cover": 0.0},
        "direct": {"nll": 0.0, "sq": 0.0, "cover": 0.0},
        "ridge": {"sq": 0.0},
    }
    for start in range(0, EPISODES, 256):
        size = min(256, EPISODES - start)
        ci = torch.randint(len(train_x), (size, base.CONTEXT), generator=generator, device=device)
        qi = torch.randint(len(test_x), (size, base.QUERIES), generator=generator, device=device)
        xc, yc = train_x[ci], train_y[ci]
        xq, yq = test_x[qi], test_y[qi]
        a = base.coefficients(size, query_family, generator, device)
        target = (a * yq).sum(dim=1)
        for name, model in (("projective", projective), ("direct", direct)):
            mean, variance = model(xc, yc, xq, a)
            sums[name]["nll"] += float(base.gaussian_nll(mean, variance, target).sum())
            sums[name]["sq"] += float((mean - target).square().sum())
            radius = 1.6448536269514722 * torch.sqrt(variance)
            sums[name]["cover"] += float(((target >= mean - radius) & (target <= mean + radius)).sum())

        ones_c = torch.ones(size, base.CONTEXT, 1, device=device)
        ones_q = torch.ones(size, base.QUERIES, 1, device=device)
        design_c = torch.cat((xc, ones_c), dim=2)
        design_q = torch.cat((xq, ones_q), dim=2)
        gram = design_c.transpose(1, 2) @ design_c
        penalty = torch.eye(base.DIM + 1, device=device)[None].expand(size, -1, -1).clone()
        penalty[:, -1, -1] = 1e-3
        rhs = design_c.transpose(1, 2) @ yc[:, :, None]
        weights = torch.linalg.solve(gram + penalty, rhs)
        ridge_prediction = (design_q @ weights).squeeze(2)
        ridge_target = (a * ridge_prediction).sum(dim=1)
        sums["ridge"]["sq"] += float((ridge_target - target).square().sum())

    rows = []
    for name in ("projective", "direct", "ridge"):
        row = {
            "dataset": dataset,
            "split_seed": split,
            "seed": seed,
            "query_family": query_family,
            "model": name,
            "rmse": math.sqrt(sums[name]["sq"] / EPISODES),
            "nll": np.nan,
            "coverage90": np.nan,
        }
        if name != "ridge":
            row["nll"] = sums[name]["nll"] / EPISODES
            row["coverage90"] = sums[name]["cover"] / EPISODES
        rows.append(row)
    return rows


def run_seed(seed: int, device: torch.device) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    projective = base.ProjectiveModel().to(device)
    direct = base.DirectModel().to(device)
    projective.load_state_dict(torch.load(base.OUT / f"projective_seed{seed}.pt", map_location=device, weights_only=True))
    direct.load_state_dict(torch.load(base.OUT / f"direct_seed{seed}.pt", map_location=device, weights_only=True))
    projective.eval(); direct.eval()
    rows = []
    started = time.perf_counter()
    for dataset in DATASETS:
        for split in SPLITS:
            arrays = prepare(dataset, split)
            for query_family in QUERY_FAMILIES:
                rows.extend(
                    evaluate(projective, direct, arrays, dataset, split, seed, query_family, device)
                )
    pd.DataFrame(rows).to_csv(OUT / f"cells_seed{seed}.csv", index=False)
    audit = {
        "seed": seed,
        "protocol_sha256": protocol_hash(),
        "wall_seconds": time.perf_counter() - started,
        "rows": len(rows),
    }
    (OUT / f"audit_seed{seed}.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))


def analyze() -> dict:
    paths = [OUT / f"cells_seed{seed}.csv" for seed in base.SEEDS]
    audit_paths = [OUT / f"audit_seed{seed}.json" for seed in base.SEEDS]
    missing = [str(path) for path in paths + audit_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    cells = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    cells.to_csv(OUT / "cells.csv", index=False)
    keys = ["dataset", "split_seed", "seed", "query_family"]
    nll = cells[cells.model != "ridge"].pivot(index=keys, columns="model", values="nll").reset_index()
    nll["projective_advantage"] = nll.direct - nll.projective
    rmse = cells.pivot(index=keys, columns="model", values="rmse").reset_index()
    point = rmse[rmse.query_family == "point"]
    dataset_point = point.groupby("dataset")[["projective", "direct", "ridge"]].mean()
    dataset_point["projective_vs_ridge_ratio"] = dataset_point.projective / dataset_point.ridge
    dense = nll[nll.query_family == "dense"]
    audits = [json.loads(path.read_text()) for path in audit_paths]
    metrics = {
        "point_datasets_projective_beats_direct": int((dataset_point.projective < dataset_point.direct).sum()),
        "point_dataset_rmse": {
            dataset: {key: float(value) for key, value in row.items()}
            for dataset, row in dataset_point.iterrows()
        },
        "dense_mean_nll_advantage": float(dense.projective_advantage.mean()),
        "dense_cell_win_rate": float((dense.projective_advantage > 0).mean()),
        "point_datasets_within_25pct_of_ridge": int((dataset_point.projective_vs_ridge_ratio <= 1.25).sum()),
        "total_wall_seconds": float(sum(audit["wall_seconds"] for audit in audits)),
    }
    integrity = bool(
        len(cells) == len(base.SEEDS) * len(DATASETS) * len(SPLITS) * len(QUERY_FAMILIES) * 3
        and np.isfinite(cells.rmse).all()
        and np.isfinite(cells[cells.model != "ridge"][["nll", "coverage90"]]).all().all()
        and all(audit["protocol_sha256"] == protocol_hash() for audit in audits)
    )
    gates = {
        "integrity": integrity,
        "point_beats_direct_on_at_least_4_datasets": metrics["point_datasets_projective_beats_direct"] >= 4,
        "dense_mean_nll_advantage_positive": metrics["dense_mean_nll_advantage"] > 0,
        "dense_cell_win_rate_at_least_60pct": metrics["dense_cell_win_rate"] >= 0.60,
        "within_25pct_of_ridge_on_at_least_4_datasets": metrics["point_datasets_within_25pct_of_ridge"] >= 4,
    }
    result = {
        "status": "complete_zero_shot_natural_anchor",
        "protocol_sha256": protocol_hash(),
        "metrics": metrics,
        "gates": gates,
        "natural_bridge_positive": all(gates.values()),
    }
    (OUT / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=base.SEEDS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    if args.analyze:
        analyze()
    elif args.seed is not None:
        run_seed(args.seed, torch.device(args.device))
    else:
        parser.error("choose --seed or --analyze")


if __name__ == "__main__":
    main()
