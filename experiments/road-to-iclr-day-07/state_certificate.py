#!/usr/bin/env python3
"""Post-hoc state-level uncertainty audit for Day-7 neural residual caches.

This script does not refit or inspect a neural base.  It replaces the coarse
five-fold operator assessment with twenty deterministic state folds, retains a
gain observation per held-out state, and applies uncertainty at the fold
cluster level.  Its outcomes are exploratory because the refinement was
motivated by the sealed MLP result; see STATE_CERTIFICATE_PROTOCOL.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr


HERE = Path(__file__).resolve().parent
GEOMETRY = HERE.parent / "geometry_transfer"
MPE = HERE.parent / "mpe_iclr"
for path in (GEOMETRY, MPE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from geometry_transfer import empirical_gain, operator_family, stable_seed, state_means  # noqa: E402
from representations import load_task  # noqa: E402
from neural_transfer import TASKS, SPLITS, backbone_output  # noqa: E402


BACKBONES = ["mlp", "resnet", "ft_transformer", "tabm"]
OUT = HERE / "results" / "state_certificate"


def local_index(rows: np.ndarray, ordered: np.ndarray) -> np.ndarray:
    lookup = {state: index for index, state in enumerate(ordered.tolist())}
    return np.asarray([lookup[state] for state in rows], dtype=np.int64)


def load_cache(backbone: str, task_name: str, split: int) -> dict[str, np.ndarray]:
    path = backbone_output(backbone) / "base_cache" / f"{task_name}__split{split}.npz"
    with np.load(path) as payload:
        return {key: payload[key] for key in payload.files}


def state_fold_scores(
    backbone: str, task_name: str, split: int, folds: int,
) -> pd.DataFrame:
    task = load_task(task_name)
    cache = load_cache(backbone, task_name, split)
    states = cache["t_states"].copy()
    np.random.default_rng(
        stable_seed("day7-certificate-state-fold", task_name, split)
    ).shuffle(states)
    held_folds = [
        np.sort(part)
        for part in np.array_split(states, min(folds, len(states)))
        if len(part)
    ]

    rows: list[dict] = []
    for fold, held in enumerate(held_folds):
        train = np.setdiff1d(states, held)
        means = state_means(cache["residual_t"], cache["row_state_t"], train)
        held_operators = operator_family(task.distance, train, held)
        for operator, matrix in held_operators.items():
            prediction = matrix @ means
            for position, state in enumerate(held):
                mask = cache["row_state_t"] == state
                residual = cache["residual_t"][mask]
                gain = float(np.mean(residual**2 - (residual - prediction[position]) ** 2))
                rows.append(
                    {
                        "backbone": backbone,
                        "source": task.manifest["source_unit"],
                        "task": task_name,
                        "split": split,
                        "operator": operator,
                        "fold": fold,
                        "state": int(state),
                        "gain": gain,
                    }
                )
    return pd.DataFrame(rows)


def outer_gains(backbone: str, task_name: str, split: int) -> dict[str, float]:
    task = load_task(task_name)
    cache = load_cache(backbone, task_name, split)
    means = state_means(cache["residual_t"], cache["row_state_t"], cache["t_states"])
    index = local_index(cache["row_state_u"], cache["u_states"])
    return {
        operator: empirical_gain(cache["residual_u"], index, matrix @ means)
        for operator, matrix in operator_family(
            task.distance, cache["t_states"], cache["u_states"]
        ).items()
    }


def summarize_cell(
    state_rows: pd.DataFrame, backbone: str, task_name: str, split: int,
) -> list[dict]:
    fold_means = state_rows.groupby(["operator", "fold"], as_index=False).gain.mean()
    summary = fold_means.groupby("operator").gain.agg(["mean", "std", "count"])
    summary["se"] = summary["std"] / np.sqrt(summary["count"])
    actual = outer_gains(backbone, task_name, split)
    winner = summary.sort_values(["mean"], ascending=False, kind="stable").iloc[0]
    winner_name = str(winner.name)
    z_familywise = float(norm.ppf(1.0 - 0.05 / (2.0 * len(summary))))

    rows = []
    for operator, row in summary.iterrows():
        chosen = operator == winner_name
        rows.append(
            {
                "backbone": backbone,
                "source": state_rows.source.iloc[0],
                "task": task_name,
                "split": split,
                "operator": operator,
                "predicted_gain": float(row["mean"]),
                "predicted_se": float(row["se"]),
                "actual_gain": actual[operator],
                "selected_mean": bool(chosen and winner["mean"] > 0),
                "selected_pointwise": bool(
                    chosen and winner["mean"] - 1.96 * winner["se"] > 0
                ),
                "selected_familywise": bool(
                    chosen and winner["mean"] - z_familywise * winner["se"] > 0
                ),
                "z_familywise": z_familywise,
                "state_folds": int(row["count"]),
            }
        )
    return rows


def selector_summary(cells: pd.DataFrame, selector: str) -> dict:
    keys = ["backbone", "source", "task", "split"]
    chosen = cells[cells[selector]].groupby(keys, as_index=False).actual_gain.sum()
    panel = cells[keys].drop_duplicates().merge(chosen, how="left").fillna({"actual_gain": 0.0})
    source_backbone = panel.groupby(["backbone", "source"]).actual_gain.mean()
    by_backbone = {}
    for backbone, frame in panel.groupby("backbone"):
        source = frame.groupby("source").actual_gain.mean()
        by_backbone[backbone] = {
            "source_balanced_gain": float(source.mean()),
            "selected_cells": int((frame.actual_gain != 0).sum()),
            "harmful_cells": int((frame.actual_gain < 0).sum()),
            "practically_harmful_cells": int((frame.actual_gain < -0.002).sum()),
            "min_source_gain": float(source.min()),
        }
    return {
        "source_backbone_balanced_gain": float(source_backbone.mean()),
        "harmful_cells": int((panel.actual_gain < 0).sum()),
        "practically_harmful_cells": int((panel.actual_gain < -0.002).sum()),
        "selected_cells": int((panel.actual_gain != 0).sum()),
        "by_backbone": by_backbone,
    }


def analyze(cells: pd.DataFrame) -> dict:
    aggregates = cells.groupby(
        ["source", "backbone", "operator"], as_index=False
    )[["predicted_gain", "actual_gain"]].mean()
    rho = float(spearmanr(aggregates.predicted_gain, aggregates.actual_gain).statistic)
    sign = float(
        np.mean((aggregates.predicted_gain > 0) == (aggregates.actual_gain > 0))
    )
    selectors = {
        name: selector_summary(cells, name)
        for name in ("selected_mean", "selected_pointwise", "selected_familywise")
    }
    return {
        "status": "complete_posthoc_development_audit",
        "aggregate_spearman": rho,
        "aggregate_sign_accuracy": sign,
        "selectors": selectors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=20)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    state_frames = []
    cell_rows = []
    for backbone in BACKBONES:
        for task_name in TASKS:
            for split in SPLITS:
                state = state_fold_scores(backbone, task_name, split, args.folds)
                state_frames.append(state)
                cell_rows.extend(summarize_cell(state, backbone, task_name, split))
                print(backbone, task_name, split, flush=True)
    states = pd.concat(state_frames, ignore_index=True)
    cells = pd.DataFrame(cell_rows).sort_values(
        ["backbone", "task", "split", "operator"]
    )
    states.to_csv(OUT / "state_gains.csv", index=False)
    cells.to_csv(OUT / "cells.csv", index=False)
    summary = analyze(cells)
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
