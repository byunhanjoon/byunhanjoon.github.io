"""Fail-closed completion and freeze audit for the five-dataset extension."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .broad_extension_data import extension_config


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/day3/broad_benchmark"


def main() -> None:
    cfg = extension_config()
    freeze_path = RESULTS / "broad_extension_freeze.json"
    freeze = json.loads(freeze_path.read_text()) if freeze_path.exists() else None
    changed = []
    missing_protected = []
    if freeze is not None:
        for raw_path, expected in freeze["sha256"].items():
            path = Path(raw_path)
            if not path.exists():
                missing_protected.append(raw_path)
            elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                changed.append(raw_path)

    paths = sorted(
        path
        for path in RESULTS.glob("extension_shard*.csv")
        if not path.stem.endswith("_curves")
    )
    frame = (
        pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
        if paths
        else pd.DataFrame()
    )
    columns = ["dataset", "target_kappa", "model", "remedy", "seed"]
    expected = {
        (dataset, float(kappa), model, cfg["remedy"], int(seed))
        for dataset in cfg["datasets"]
        for kappa in cfg["kappas"]
        for model in cfg["models"]
        for seed in cfg["seeds"]
    }
    if len(frame):
        successful = frame[frame.failure.fillna("").eq("")]
        actual = set(successful[columns].itertuples(index=False, name=None))
        duplicate_success_rows = int(successful.duplicated(columns, keep=False).sum())
        failures = int(frame.failure.fillna("").ne("").sum())
        equivalence = float(successful.natural_equivalence_max_error.max())
        realized = successful.representation_metadata.map(
            lambda value: float(json.loads(value)["basis_condition"])
        )
        kappa_error = float(
            np.max(np.abs(realized.to_numpy() / successful.target_kappa.to_numpy() - 1))
        )
    else:
        actual = set()
        duplicate_success_rows = failures = 0
        equivalence = kappa_error = None
    complete = bool(
        freeze is not None
        and not changed
        and not missing_protected
        and actual == expected
        and duplicate_success_rows == 0
        and failures == 0
        and equivalence is not None
        and equivalence <= 1e-8
        and kappa_error is not None
        and kappa_error <= 1e-6
    )
    payload = {
        "freeze": {
            "exists": freeze is not None,
            "matches": freeze is not None and not changed and not missing_protected,
            "aggregate_sha256": freeze.get("aggregate_sha256") if freeze else None,
            "changed": changed,
            "missing": missing_protected,
        },
        "coverage": {
            "expected": len(expected),
            "rows": len(frame),
            "successful_unique": len(actual),
            "failures": failures,
            "duplicate_success_rows": duplicate_success_rows,
            "missing": len(expected - actual),
            "unexpected": len(actual - expected),
            "missing_examples": [list(value) for value in sorted(expected - actual, key=str)[:10]],
        },
        "scientific_invariants": {
            "natural_equivalence_max_error": equivalence,
            "controlled_kappa_max_relative_error": kappa_error,
        },
        "complete": complete,
    }
    output = RESULTS / "extension_completion_audit.json"
    output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    raise SystemExit(0 if complete else 1)


if __name__ == "__main__":
    main()
