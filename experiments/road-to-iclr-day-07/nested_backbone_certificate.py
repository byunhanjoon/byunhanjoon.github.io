#!/usr/bin/env python3
"""Disjoint-state test of a backbone-robust optional geometry certificate."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from scipy.stats import norm


HERE = Path(__file__).resolve().parent
GEOMETRY = HERE.parent / "geometry_transfer"
MPE = HERE.parent / "mpe_iclr"
for path in (GEOMETRY, MPE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from geometry_transfer import operator_family, stable_seed, state_means  # noqa: E402
from representations import load_task, split_state_indices  # noqa: E402
from ridge_benchmark import ordinary_design  # noqa: E402
from neural_transfer import (  # noqa: E402
    TASKS, SPLITS, fit_predict,
)


BACKBONES = ["mlp", "resnet", "ft_transformer", "tabm"]
OUT = HERE / "results" / "nested_backbone"
ALPHA = 0.05


def rows_for_states(row_state: np.ndarray, states: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.isin(row_state, states))


def state_gain_rows(
    residual: np.ndarray,
    row_state: np.ndarray,
    states: np.ndarray,
    state_prediction: np.ndarray,
) -> np.ndarray:
    result = []
    for position, state in enumerate(states):
        values = residual[row_state == state]
        prediction = state_prediction[position]
        result.append(float(np.mean(values**2 - (values - prediction) ** 2)))
    return np.asarray(result, dtype=np.float64)


def cache_path(backbone: str, task_name: str, split: int) -> Path:
    return OUT / backbone / "base_cache" / f"{task_name}__split{split}.npz"


def make_cache(
    backbone: str, task_name: str, split: int, device: torch.device,
) -> dict[str, np.ndarray]:
    path = cache_path(backbone, task_name, split)
    if path.exists():
        with np.load(path) as payload:
            return {key: payload[key] for key in payload.files}

    task = load_task(task_name)
    parts = split_state_indices(task, split)
    construction = parts["train"]
    validation = parts["validation"]
    test = parts["test"]
    row_state = task.row_state_indices()
    construction_rows = rows_for_states(row_state, construction)
    validation_rows = rows_for_states(row_state, validation)
    test_rows = rows_for_states(row_state, test)

    raw_target = pd.to_numeric(task.rows.target, errors="raise").to_numpy(np.float64)
    center = raw_target[construction_rows].mean()
    scale = raw_target[construction_rows].std() or 1.0
    target_values = (raw_target - center) / scale
    design_sparse = ordinary_design(task, construction_rows)
    if design_sparse.shape[1] == 0:
        design_sparse = sparse.csr_matrix(
            np.ones((len(target_values), 1), dtype=np.float32)
        )
    design = torch.from_numpy(
        design_sparse.toarray().astype(np.float32, copy=False)
    ).to(device)
    target = torch.from_numpy(target_values.astype(np.float32)).to(device)

    order = construction_rows.copy()
    np.random.default_rng(
        stable_seed("day7-nested-rowfold", backbone, task_name, split)
    ).shuffle(order)
    row_folds = np.array_split(order, 3)
    oof = np.empty(len(construction_rows), dtype=np.float64)
    position = {row: index for index, row in enumerate(construction_rows.tolist())}
    for fold, held in enumerate(row_folds):
        fit_rows = np.setdiff1d(construction_rows, held)
        prediction = fit_predict(
            design, target, row_state, fit_rows, held,
            stable_seed("day7-nested-oof", backbone, task_name, split, fold),
            device, backbone,
        )
        oof[[position[row] for row in held]] = prediction

    future_rows = np.concatenate([validation_rows, test_rows])
    future_prediction = fit_predict(
        design, target, row_state, construction_rows, future_rows,
        stable_seed("day7-nested-full", backbone, task_name, split),
        device, backbone,
    )
    n_validation = len(validation_rows)
    payload = {
        "construction_states": construction,
        "validation_states": validation,
        "test_states": test,
        "row_state_construction": row_state[construction_rows],
        "row_state_validation": row_state[validation_rows],
        "row_state_test": row_state[test_rows],
        "residual_construction": target_values[construction_rows] - oof,
        "residual_validation": target_values[validation_rows] - future_prediction[:n_validation],
        "residual_test": target_values[test_rows] - future_prediction[n_validation:],
        "ordinary_dimensions": np.asarray([design.shape[1]]),
        "oof_finite": np.asarray([np.isfinite(oof).all()]),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return payload


def run_cell(
    backbone: str, task_name: str, split: int, device: torch.device,
) -> list[dict]:
    task = load_task(task_name)
    cache = make_cache(backbone, task_name, split, device)
    means = state_means(
        cache["residual_construction"],
        cache["row_state_construction"],
        cache["construction_states"],
    )
    validation_operators = operator_family(
        task.distance, cache["construction_states"], cache["validation_states"]
    )
    test_operators = operator_family(
        task.distance, cache["construction_states"], cache["test_states"]
    )
    rows = []
    for operator in sorted(validation_operators):
        validation_gain = state_gain_rows(
            cache["residual_validation"], cache["row_state_validation"],
            cache["validation_states"], validation_operators[operator] @ means,
        )
        test_gain = state_gain_rows(
            cache["residual_test"], cache["row_state_test"],
            cache["test_states"], test_operators[operator] @ means,
        )
        rows.append(
            {
                "backbone": backbone,
                "source": task.manifest["source_unit"],
                "task": task_name,
                "split": split,
                "operator": operator,
                "validation_gain": float(validation_gain.mean()),
                "validation_se": float(validation_gain.std(ddof=1) / np.sqrt(len(validation_gain))),
                "test_gain": float(test_gain.mean()),
                "validation_states": len(validation_gain),
                "test_states": len(test_gain),
                "ordinary_dimensions": int(cache["ordinary_dimensions"][0]),
                "oof_finite": bool(cache["oof_finite"][0]),
            }
        )
    return rows


def per_backbone_decisions(cells: pd.DataFrame, use_lcb: bool) -> pd.DataFrame:
    z_value = float(norm.ppf(1.0 - ALPHA / (2.0 * cells.operator.nunique())))
    rows = []
    keys = ["backbone", "source", "task", "split"]
    for _, group in cells.groupby(keys):
        score = group.validation_gain.copy()
        if use_lcb:
            score = score - z_value * group.validation_se
        winner = group.loc[score.idxmax()]
        selected = bool(score.loc[winner.name] > 0)
        rows.append(
            {
                **{key: winner[key] for key in keys},
                "operator": winner.operator,
                "selected": selected,
                "test_gain": float(winner.test_gain) if selected else 0.0,
                "decision_score": float(score.loc[winner.name]),
            }
        )
    return pd.DataFrame(rows)


def robust_decisions(cells: pd.DataFrame) -> pd.DataFrame:
    n_comparisons = cells.operator.nunique() * cells.backbone.nunique()
    z_value = float(norm.ppf(1.0 - ALPHA / (2.0 * n_comparisons)))
    scored = cells.copy()
    scored["lcb"] = scored.validation_gain - z_value * scored.validation_se
    rows = []
    for (source, task, split), group in scored.groupby(["source", "task", "split"]):
        operator_score = group.groupby("operator").lcb.min()
        operator = str(operator_score.idxmax())
        decision_score = float(operator_score.loc[operator])
        selected = decision_score > 0
        for row in group[group.operator == operator].itertuples(index=False):
            rows.append(
                {
                    "backbone": row.backbone,
                    "source": source,
                    "task": task,
                    "split": split,
                    "operator": operator,
                    "selected": selected,
                    "test_gain": float(row.test_gain) if selected else 0.0,
                    "decision_score": decision_score,
                }
            )
    return pd.DataFrame(rows)


def decision_summary(frame: pd.DataFrame) -> dict:
    selected = frame[frame.selected]
    return {
        "mean_test_gain": float(frame.test_gain.mean()),
        "harmful_cells": int((frame.test_gain < 0).sum()),
        "practically_harmful_cells": int((frame.test_gain < -0.002).sum()),
        "selected_backbone_cells": int(frame.selected.sum()),
        "selected_source_splits": int(selected[["source", "split"]].drop_duplicates().shape[0]),
        "selected_sources": int(selected.source.nunique()),
    }


def analyze() -> dict:
    paths = [OUT / backbone / "cells.csv" for backbone in BACKBONES]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing completed backbone cells: {missing}")
    cells = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    cells = cells.sort_values(["backbone", "task", "split", "operator"])
    cells.to_csv(OUT / "cells.csv", index=False)
    decisions = {
        "mean": per_backbone_decisions(cells, use_lcb=False),
        "per_backbone_lcb": per_backbone_decisions(cells, use_lcb=True),
        "backbone_worst": robust_decisions(cells),
    }
    pd.concat(
        [frame.assign(rule=name) for name, frame in decisions.items()],
        ignore_index=True,
    ).to_csv(OUT / "decisions.csv", index=False)
    summary = {
        "status": "complete_development_replay",
        "rules": {name: decision_summary(frame) for name, frame in decisions.items()},
        "integrity": bool(
            cells.oof_finite.all()
            and np.isfinite(cells[["validation_gain", "validation_se", "test_gain"]]).all().all()
        ),
    }
    robust = summary["rules"]["backbone_worst"]
    summary["gates"] = {
        "no_practical_harm": robust["practically_harmful_cells"] == 0,
        "positive_mean_gain": robust["mean_test_gain"] > 0,
        "at_least_two_source_splits_and_sources": (
            robust["selected_source_splits"] >= 2 and robust["selected_sources"] >= 2
        ),
        "fewer_harms_than_mean": (
            robust["harmful_cells"] < summary["rules"]["mean"]["harmful_cells"]
        ),
        "no_more_harms_than_per_backbone_lcb": (
            robust["harmful_cells"] <= summary["rules"]["per_backbone_lcb"]["harmful_cells"]
        ),
        "integrity": summary["integrity"],
    }
    summary["passes"] = all(summary["gates"].values())
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=BACKBONES)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    if args.analyze:
        analyze()
        return
    if args.backbone is None:
        parser.error("--backbone is required unless --analyze is used")
    device = torch.device(args.device)
    out = OUT / args.backbone
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    matrix = [(task, split) for task in TASKS for split in SPLITS]
    for index, (task_name, split) in enumerate(matrix):
        if index % args.shards != args.shard:
            continue
        started = time.time()
        rows.extend(run_cell(args.backbone, task_name, split, device))
        print(args.backbone, task_name, split, round(time.time() - started, 1), flush=True)
    shard_path = out / f"cells_shard{args.shard}of{args.shards}.csv"
    pd.DataFrame(rows).to_csv(shard_path, index=False)
    paths = [out / f"cells_shard{index}of{args.shards}.csv" for index in range(args.shards)]
    if all(path.exists() for path in paths):
        cells = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
        cells.sort_values(["task", "split", "operator"]).to_csv(out / "cells.csv", index=False)


if __name__ == "__main__":
    main()
