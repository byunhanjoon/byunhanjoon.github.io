"""Fail-closed completion audit for the Day 3 trajectory extension."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from .trajectory_decomposition import CONFIG_PATH, RESULTS, ROOT


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text())
    freeze = json.loads((RESULTS / "preregistration_freeze.json").read_text())
    run_paths = [RESULTS / "runs_gpu0.csv", RESULTS / "runs_gpu1.csv"]
    trajectory_paths = [
        RESULTS / "runs_gpu0_trajectories.csv",
        RESULTS / "runs_gpu1_trajectories.csv",
    ]
    runs = pd.concat([pd.read_csv(path) for path in run_paths], ignore_index=True)
    trajectories = pd.concat([pd.read_csv(path) for path in trajectory_paths], ignore_index=True)
    identifiers = ["dataset", "pair_label", "model", "arm", "seed"]
    expected = (
        len(config["datasets"])
        * len(config["models"])
        * len(config["seeds"])
        * len(config["arms"])
        * (len(config["representation_pairs"]["controlled"]["kappas"]) + 1)
    )
    failure = runs["failure"].fillna("").astype(str).str.strip() != ""
    counts = trajectories.groupby(identifiers).size()
    observed_steps = sorted(trajectories["step"].unique().astype(int).tolist())
    matched = trajectories[
        trajectories["arm"].isin(("matched_adamw", "matched_input_natural"))
        & (trajectories["step"] == 0)
    ]
    checks = {
        "config_digest_matches_freeze": _sha256(CONFIG_PATH) == freeze["config_sha256"],
        "protocol_digest_matches_freeze": _sha256(ROOT / freeze["protocol"]) == freeze["protocol_sha256"],
        "expected_cells": len(runs) == expected,
        "unique_cells": not runs.duplicated(identifiers).any(),
        "no_failures": not failure.any(),
        "twelve_trajectory_rows_per_cell": bool((counts == 12).all()) and len(counts) == expected,
        "all_preregistered_steps": observed_steps == config["training"]["trajectory_steps"],
        "matched_step0_gate": float(matched["prediction_drift"].max())
        <= float(config["analysis_gates"]["matched_step0_max_drift"]),
        "basis_relation_gate": float(runs["basis_relation_max_error"].max()) < 1e-9,
        "analysis_present": (RESULTS / "analysis_summary.json").exists(),
    }
    output = {
        "status": "complete" if all(checks.values()) else "failed",
        "checks": checks,
        "expected_cells": expected,
        "observed_cells": int(len(runs)),
        "failures": int(failure.sum()),
        "trajectory_rows": int(len(trajectories)),
        "matched_step0_max_drift": float(matched["prediction_drift"].max()),
        "basis_relation_max_error": float(runs["basis_relation_max_error"].max()),
        "artifacts": {
            str(path.relative_to(ROOT)): _sha256(path)
            for path in [*run_paths, *trajectory_paths, RESULTS / "analysis_summary.json"]
        },
    }
    (RESULTS / "completion_audit.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))
    if output["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
