#!/usr/bin/env python3
"""Backbone-robust optional-bias decision on the Day-7 development panel."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm


HERE = Path(__file__).resolve().parent
INPUT = HERE / "results" / "state_certificate" / "cells.csv"
OUT = HERE / "results" / "backbone_robust"
ALPHA = 0.05


def materialize_decisions(cells: pd.DataFrame, rule: str) -> pd.DataFrame:
    backbones = sorted(cells.backbone.unique())
    operators = sorted(cells.operator.unique())
    if rule == "backbone_mean":
        z_value = float(norm.ppf(1.0 - ALPHA / (2.0 * len(operators))))
    elif rule == "backbone_worst":
        z_value = float(
            norm.ppf(1.0 - ALPHA / (2.0 * len(operators) * len(backbones)))
        )
    else:
        raise KeyError(rule)

    rows = []
    for (source, task, split), group in cells.groupby(["source", "task", "split"]):
        scored = group.copy()
        scored["lcb"] = scored.predicted_gain - z_value * scored.predicted_se
        if rule == "backbone_worst":
            score = scored.groupby("operator").lcb.min()
        else:
            aggregate = scored.groupby("operator").agg(
                mean=("predicted_gain", "mean"),
                se=("predicted_se", lambda values: np.sqrt(np.sum(values**2)) / len(values)),
            )
            score = aggregate["mean"] - z_value * aggregate["se"]
        operator = str(score.idxmax())
        robust_lcb = float(score.loc[operator])
        selected = robust_lcb > 0
        for row in group[group.operator == operator].itertuples(index=False):
            rows.append(
                {
                    "rule": rule,
                    "source": source,
                    "task": task,
                    "split": split,
                    "backbone": row.backbone,
                    "operator": operator,
                    "certificate_lcb": robust_lcb,
                    "selected": selected,
                    "actual_gain": float(row.actual_gain),
                    "deployed_gain": float(row.actual_gain) if selected else 0.0,
                    "z_value": z_value,
                }
            )
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame) -> dict:
    by_backbone = {}
    for backbone, group in frame.groupby("backbone"):
        by_backbone[backbone] = {
            "mean_gain": float(group.deployed_gain.mean()),
            "harmful_cells": int((group.deployed_gain < 0).sum()),
            "practically_harmful_cells": int((group.deployed_gain < -0.002).sum()),
            "selected_cells": int(group.selected.sum()),
        }
    selected = frame[frame.selected]
    return {
        "mean_gain": float(frame.deployed_gain.mean()),
        "harmful_cells": int((frame.deployed_gain < 0).sum()),
        "practically_harmful_cells": int((frame.deployed_gain < -0.002).sum()),
        "selected_task_splits": int(frame[frame.selected][["task", "split"]].drop_duplicates().shape[0]),
        "selected_backbone_cells": int(frame.selected.sum()),
        "selected_sources": int(selected.source.nunique()),
        "by_backbone": by_backbone,
    }


def per_backbone_familywise(cells: pd.DataFrame) -> dict:
    chosen = cells[cells.selected_familywise].copy()
    keys = ["backbone", "source", "task", "split"]
    panel = cells[keys].drop_duplicates().merge(
        chosen[keys + ["actual_gain"]], how="left"
    ).fillna({"actual_gain": 0.0})
    return {
        "mean_gain": float(panel.actual_gain.mean()),
        "harmful_cells": int((panel.actual_gain < 0).sum()),
        "practically_harmful_cells": int((panel.actual_gain < -0.002).sum()),
        "selected_backbone_cells": int((panel.actual_gain != 0).sum()),
    }


def main() -> None:
    cells = pd.read_csv(INPUT)
    frames = [
        materialize_decisions(cells, "backbone_mean"),
        materialize_decisions(cells, "backbone_worst"),
    ]
    decisions = pd.concat(frames, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    decisions.to_csv(OUT / "decisions.csv", index=False)
    summary = {
        "status": "complete_posthoc_development_audit",
        "per_backbone_familywise": per_backbone_familywise(cells),
        "backbone_mean": summarize(frames[0]),
        "backbone_worst": summarize(frames[1]),
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
