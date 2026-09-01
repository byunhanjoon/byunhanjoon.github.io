#!/usr/bin/env python3
"""Fresh split-2 replay of the frozen neural geometry certificate."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DAY7 = ROOT / "experiments" / "road-to-iclr-day-07"
sys.path.insert(0, str(DAY7))

import nested_backbone_certificate as frozen  # noqa: E402


OUT = HERE / "results"
BACKBONES = tuple(frozen.BACKBONES)
TASKS = tuple(frozen.TASKS)
SPLIT = 2
MAX_SECONDS = 60 * 60

# Redirect the imported implementation's only mutable output root.  All model,
# split, operator, and decision code remains unchanged.
frozen.OUT = OUT


def protocol_hash() -> str:
    return hashlib.sha256((HERE / "PROTOCOL.md").read_bytes()).hexdigest()


def run_backbone(backbone: str, device: str) -> None:
    started = time.perf_counter()
    rows: list[dict] = []
    for task in TASKS:
        cell_started = time.perf_counter()
        rows.extend(frozen.run_cell(backbone, task, SPLIT, torch.device(device)))
        elapsed = time.perf_counter() - started
        print(
            json.dumps(
                {
                    "backbone": backbone,
                    "task": task,
                    "split": SPLIT,
                    "cell_seconds": time.perf_counter() - cell_started,
                    "elapsed_seconds": elapsed,
                }
            ),
            flush=True,
        )
        if elapsed > MAX_SECONDS:
            raise TimeoutError(f"geometry sprint exceeded one hour for {backbone}")
    target = OUT / backbone
    target.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(["task", "operator"]).to_csv(
        target / "cells.csv", index=False
    )


def state_balanced_mse(residual: np.ndarray, row_state: np.ndarray) -> float:
    return float(
        np.mean(
            [
                np.mean(residual[row_state == state] ** 2)
                for state in np.unique(row_state)
            ]
        )
    )


def analyze() -> dict:
    paths = [OUT / backbone / "cells.csv" for backbone in BACKBONES]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing fresh cells: {missing}")
    cells = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    decisions = frozen.per_backbone_decisions(cells, use_lcb=True)

    baseline = []
    for backbone in BACKBONES:
        for task in TASKS:
            with np.load(frozen.cache_path(backbone, task, SPLIT)) as payload:
                baseline.append(
                    {
                        "backbone": backbone,
                        "task": task,
                        "split": SPLIT,
                        "baseline_mse": state_balanced_mse(
                            payload["residual_test"], payload["row_state_test"]
                        ),
                    }
                )
    decisions = decisions.merge(pd.DataFrame(baseline), validate="one_to_one")
    decisions["relative_gain"] = decisions["test_gain"] / decisions["baseline_mse"]
    decisions.to_csv(OUT / "decisions.csv", index=False)
    cells.to_csv(OUT / "cells.csv", index=False)

    selected = decisions[decisions.selected]
    backbone_means = decisions.groupby("backbone").test_gain.mean()
    integrity = bool(
        cells.oof_finite.all()
        and np.isfinite(cells[["validation_gain", "validation_se", "test_gain"]]).all().all()
        and np.isfinite(decisions[["baseline_mse", "relative_gain"]]).all().all()
    )
    metrics = {
        "mean_absolute_gain": float(decisions.test_gain.mean()),
        "mean_relative_gain": float(decisions.relative_gain.mean()),
        "median_selected_relative_gain": (
            float(selected.relative_gain.median()) if len(selected) else 0.0
        ),
        "harmful_selected_cells": int((decisions.test_gain < 0).sum()),
        "practically_harmful_selected_cells": int((decisions.test_gain < -0.002).sum()),
        "selected_cells": int(decisions.selected.sum()),
        "selected_sources": int(selected.source.nunique()),
        "selected_backbones": int(selected.backbone.nunique()),
        "positive_mean_backbones": int((backbone_means > 0).sum()),
        "by_backbone_mean_gain": {key: float(value) for key, value in backbone_means.items()},
    }
    gates = {
        "integrity": integrity,
        "mean_absolute_gain_at_least_0_005": metrics["mean_absolute_gain"] >= 0.005,
        "no_practical_harm": metrics["practically_harmful_selected_cells"] == 0,
        "at_least_6_selected_cells": metrics["selected_cells"] >= 6,
        "at_least_2_sources": metrics["selected_sources"] >= 2,
        "at_least_3_backbones": metrics["selected_backbones"] >= 3,
        "at_least_3_positive_backbones": metrics["positive_mean_backbones"] >= 3,
    }
    result = {
        "status": "complete_fresh_split_replay",
        "protocol_sha256": protocol_hash(),
        "split": SPLIT,
        "cells": len(decisions),
        "metrics": metrics,
        "gates": gates,
        "passed": all(gates.values()),
    }
    (OUT / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=BACKBONES)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    if args.analyze:
        analyze()
    elif args.backbone:
        run_backbone(args.backbone, args.device)
    else:
        parser.error("choose --backbone or --analyze")


if __name__ == "__main__":
    main()
