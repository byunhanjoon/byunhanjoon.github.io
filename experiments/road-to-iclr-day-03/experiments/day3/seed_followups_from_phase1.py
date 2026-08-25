"""Reuse exact phase-1 cells in confirmation and robustness follow-ups.

The reused cells have identical datasets, splits, representations, models,
seeds, remedies, and frozen hyperparameters.  Copying them avoids rerunning 696
scientifically identical training cells while preserving their provenance.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .analyze_broad_phase1 import load_phase1
from .broad_data import config


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/day3/broad_benchmark"


def _setting(remedy: str, selected: dict[str, dict[str, object]]) -> dict[str, object]:
    return selected.get(
        remedy, selected.get("anchor_whiten_adamw", selected["adamw"])
    )


def _matching_setting(row: pd.Series, setting: dict[str, object]) -> bool:
    return bool(
        np.isclose(row.learning_rate_requested, float(setting["learning_rate"]))
        and np.isclose(row.ridge_requested, float(setting.get("ridge", 1e-8)))
        and int(row.precondition_frequency_requested)
        == int(setting.get("precondition_frequency", 10))
    )


def _merge_write(path: Path, reused: pd.DataFrame, keys: list[str]) -> None:
    if path.exists():
        existing = pd.read_csv(path)
        frame = pd.concat([existing, reused], ignore_index=True, sort=False)
    else:
        frame = reused
    frame = frame.drop_duplicates(keys, keep="first")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def seed_confirmation(phase: pd.DataFrame, selected: dict[str, dict[str, object]]) -> int:
    cfg = config()
    selection = json.loads((RESULTS / "confirmation_selection.json").read_text())
    remedies = (
        selection["always_confirmed_controls"]
        + selection["selected_deployable_comparisons"]
    )
    valid = phase[
        phase.failure.fillna("").eq("")
        & phase.representation.eq("controlled")
        & phase.dataset.isin(cfg["architecture_confirmation_datasets"])
        & phase.remedy.isin(remedies)
        & phase.seed.isin(cfg["broad_seeds"])
        & (phase.remedy.eq("adamw") | phase.model.eq("mlp"))
    ].copy()
    matches = valid.apply(
        lambda row: _matching_setting(row, _setting(row.remedy, selected)), axis=1
    )
    if not bool(matches.all()):
        raise RuntimeError("A reusable phase-1 confirmation cell has mismatched settings")
    valid["reused_from_phase1"] = True
    keys = ["dataset", "target_kappa", "model", "remedy", "seed"]
    expected = (
        len(cfg["architecture_confirmation_datasets"])
        * len(cfg["kappas"])
        * len(cfg["broad_seeds"])
        * (len(cfg["models"]) + len(remedies) - 1)
    )
    if len(valid.drop_duplicates(keys)) != expected:
        raise RuntimeError(
            f"Expected {expected} reusable confirmation cells, got "
            f"{len(valid.drop_duplicates(keys))}"
        )
    order = {name: index for index, name in enumerate(cfg["architecture_confirmation_datasets"])}
    for shard in (0, 1):
        part = valid[valid.dataset.map(order).mod(2).eq(shard)]
        _merge_write(RESULTS / f"confirmation_shard{shard}.csv", part, keys)
    return len(valid)


def seed_robustness(phase: pd.DataFrame, selected: dict[str, dict[str, object]]) -> int:
    cfg = config()
    remedies = [
        "adamw",
        "anchor_whiten_adamw",
        "sketch_anchor_whiten_adamw",
        "input_natural",
    ]
    valid = phase[
        phase.failure.fillna("").eq("")
        & phase.representation.eq("controlled")
        & phase.dataset.isin(cfg["robustness"]["datasets"])
        & phase.remedy.isin(remedies)
        & phase.model.eq("mlp")
        & phase.seed.isin(cfg["broad_seeds"])
    ].copy()
    matches = valid.apply(
        lambda row: _matching_setting(row, _setting(row.remedy, selected)), axis=1
    )
    if not bool(matches.all()):
        raise RuntimeError("A reusable phase-1 robustness cell has mismatched settings")
    valid["experiment"] = "rank_scalability_robustness"
    valid["duplicate_fraction"] = 0.0
    valid["ridge"] = valid.ridge_requested.astype(float)
    valid["reused_from_phase1"] = True
    keys = [
        "dataset",
        "duplicate_fraction",
        "target_kappa",
        "remedy",
        "ridge",
        "seed",
    ]
    expected = (
        len(cfg["robustness"]["datasets"])
        * len(cfg["kappas"])
        * len(remedies)
        * len(cfg["broad_seeds"])
    )
    if len(valid.drop_duplicates(keys)) != expected:
        raise RuntimeError(
            f"Expected {expected} reusable robustness cells, got "
            f"{len(valid.drop_duplicates(keys))}"
        )
    order = {name: index for index, name in enumerate(cfg["robustness"]["datasets"])}
    for shard in (0, 1):
        part = valid[valid.dataset.map(order).mod(2).eq(shard)]
        _merge_write(RESULTS / f"robustness_shard{shard}.csv", part, keys)
    return len(valid)


def main() -> None:
    phase = load_phase1()
    selected = json.loads((RESULTS / "selected_hyperparameters.json").read_text())[
        "selected"
    ]
    payload = {
        "confirmation_cells_reused": seed_confirmation(phase, selected),
        "robustness_cells_reused": seed_robustness(phase, selected),
        "source": "successful frozen phase1 cells with identical configurations",
    }
    (RESULTS / "followup_reuse.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
