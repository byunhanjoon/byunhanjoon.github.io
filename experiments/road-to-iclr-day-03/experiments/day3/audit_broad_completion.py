"""Fail-closed completion audit for the preregistered broad benchmark.

This script is intentionally separate from scientific analysis.  It checks that
every frozen cell exists, that failures and duplicate successes are visible, and
that the pre-outcome freeze still matches the protected files.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .broad_data import config


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/day3/broad_benchmark"


def _load(pattern: str) -> pd.DataFrame:
    paths = sorted(path for path in RESULTS.glob(pattern) if not path.stem.endswith("_curves"))
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def _success(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    return frame[frame.failure.fillna("").eq("")]


def _keyset(frame: pd.DataFrame, columns: list[str]) -> set[tuple]:
    return set(frame[columns].itertuples(index=False, name=None)) if len(frame) else set()


def expected_phase1() -> set[tuple]:
    cfg = config()
    cells: set[tuple] = set()
    for dataset in cfg["datasets"]:
        for kappa in cfg["kappas"]:
            for model in cfg["models"]:
                for seed in cfg["broad_seeds"]:
                    cells.add((dataset, "controlled", float(kappa), model, "adamw", seed))
            for remedy in cfg["remedies"]:
                if remedy == "adamw":
                    continue
                for seed in cfg["broad_seeds"]:
                    cells.add((dataset, "controlled", float(kappa), "mlp", remedy, seed))
        for representation in (
            "cumulative_helmert",
            "local_adjacent",
            "raw_standard",
            "quantile_standard",
        ):
            for model in ("mlp", "resnet"):
                for seed in cfg["broad_seeds"]:
                    cells.add((dataset, representation, 1.0, model, "adamw", seed))
    return cells


def expected_confirmation() -> set[tuple]:
    cfg = config()
    selection_path = RESULTS / "confirmation_selection.json"
    if not selection_path.exists():
        return set()
    selection = json.loads(selection_path.read_text())
    remedies = (
        selection["always_confirmed_controls"]
        + selection["selected_deployable_comparisons"]
    )
    return {
        (dataset, float(kappa), model, remedy, seed)
        for dataset in cfg["architecture_confirmation_datasets"]
        for kappa in cfg["kappas"]
        for model in cfg["models"]
        for remedy in remedies
        for seed in cfg["confirmation_seeds"]
    }


def expected_robustness() -> set[tuple]:
    cfg = config()
    remedies = (
        "adamw",
        "anchor_whiten_adamw",
        "sketch_anchor_whiten_adamw",
        "input_natural",
    )
    cells = set()
    for dataset in cfg["robustness"]["datasets"]:
        for fraction in cfg["robustness"]["duplicate_fractions"]:
            for kappa in cfg["kappas"]:
                for remedy in remedies:
                    ridges = (
                        cfg["robustness"]["ridge_relative"]
                        if remedy == "input_natural"
                        else [1e-8]
                    )
                    for ridge in ridges:
                        for seed in cfg["broad_seeds"]:
                            cells.add(
                                (
                                    dataset,
                                    float(fraction),
                                    float(kappa),
                                    remedy,
                                    float(ridge),
                                    seed,
                                )
                            )
    return cells


def _coverage(
    frame: pd.DataFrame, expected: set[tuple], columns: list[str]
) -> dict[str, object]:
    successful = _success(frame)
    actual = _keyset(successful, columns)
    duplicated = successful.duplicated(columns, keep=False) if len(successful) else []
    return {
        "expected": len(expected),
        "rows": len(frame),
        "successful_unique": len(actual),
        "failures": int(frame.failure.fillna("").ne("").sum()) if len(frame) else 0,
        "duplicate_success_rows": int(np.sum(duplicated)),
        "missing": len(expected - actual),
        "unexpected": len(actual - expected),
        "complete": actual == expected,
        "missing_examples": [list(value) for value in sorted(expected - actual, key=str)[:10]],
    }


def _freeze_audit() -> dict[str, object]:
    path = RESULTS / "broad_freeze.json"
    if not path.exists():
        return {"exists": False, "matches": False, "changed": []}
    payload = json.loads(path.read_text())
    changed = []
    authorized_analysis_fixes = []
    missing = []
    addendum_path = RESULTS / "analysis_fix_addendum.json"
    addendum = json.loads(addendum_path.read_text()) if addendum_path.exists() else {"files": {}}
    for raw_path, expected in payload["sha256"].items():
        file = Path(raw_path)
        if not file.exists():
            missing.append(raw_path)
            continue
        digest = hashlib.sha256(file.read_bytes()).hexdigest()
        if digest != expected:
            fix = addendum.get("files", {}).get(raw_path)
            if (
                fix
                and fix.get("original_sha256") == expected
                and fix.get("corrected_sha256") == digest
            ):
                authorized_analysis_fixes.append(raw_path)
            else:
                changed.append(raw_path)
    return {
        "exists": True,
        "protected_files": len(payload["sha256"]),
        "aggregate_sha256": payload["aggregate_sha256"],
        "changed": changed,
        "authorized_analysis_fixes": authorized_analysis_fixes,
        "missing": missing,
        "matches": not changed and not missing,
    }


def main() -> None:
    phase = _load("phase1_shard[0-9]*.csv")
    confirmation = _load("confirmation_shard[0-9]*.csv")
    robustness = _load("robustness_shard[0-9]*.csv")
    phase_columns = [
        "dataset",
        "representation",
        "target_kappa",
        "model",
        "remedy",
        "seed",
    ]
    confirmation_columns = ["dataset", "target_kappa", "model", "remedy", "seed"]
    robustness_columns = [
        "dataset",
        "duplicate_fraction",
        "target_kappa",
        "remedy",
        "ridge",
        "seed",
    ]
    phase_expected = expected_phase1()
    confirmation_expected = expected_confirmation()
    robustness_expected = expected_robustness()
    payload = {
        "freeze": _freeze_audit(),
        "confirmation_selection_exists": (RESULTS / "confirmation_selection.json").exists(),
        "phase1": _coverage(phase, phase_expected, phase_columns),
        "confirmation": _coverage(
            confirmation, confirmation_expected, confirmation_columns
        ),
        "robustness": _coverage(robustness, robustness_expected, robustness_columns),
    }
    valid = _success(phase)
    controlled = valid[valid.representation.eq("controlled")]
    realized_basis_condition = (
        controlled.representation_metadata.map(
            lambda value: float(json.loads(value)["basis_condition"])
        )
        if len(controlled)
        else pd.Series(dtype=float)
    )
    payload["scientific_invariants"] = {
        "natural_equivalence_max_error": (
            float(valid.natural_equivalence_max_error.max()) if len(valid) else None
        ),
        "controlled_kappa_max_relative_error": (
            float(
                np.max(
                    np.abs(
                        realized_basis_condition.to_numpy()
                        / controlled.target_kappa.to_numpy()
                        - 1
                    )
                )
            )
            if len(controlled)
            else None
        ),
    }
    payload["complete"] = bool(
        payload["freeze"]["matches"]
        and payload["phase1"]["complete"]
        and payload["confirmation_selection_exists"]
        and payload["confirmation"]["complete"]
        and payload["robustness"]["complete"]
        and payload["scientific_invariants"]["natural_equivalence_max_error"] is not None
        and payload["scientific_invariants"]["natural_equivalence_max_error"] <= 1e-8
        and payload["scientific_invariants"]["controlled_kappa_max_relative_error"] is not None
        and payload["scientific_invariants"]["controlled_kappa_max_relative_error"] <= 1e-6
    )
    output = RESULTS / "completion_audit.json"
    output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    raise SystemExit(0 if payload["complete"] else 1)


if __name__ == "__main__":
    main()
