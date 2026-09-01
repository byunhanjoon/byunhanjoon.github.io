#!/usr/bin/env python3
"""Aggregate Stage-1 cells and apply the frozen survival rules."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tournament.common import load_protocol, protocol_hashes, write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    protocol = load_protocol()
    directory = ROOT / "results" / "processed" / "stage1_cells"
    files = sorted(path for path in directory.glob("*.csv") if "coordinate_audit" not in path.name)
    expected = (
        len(protocol["stage1_datasets"])
        * len(protocol["stage1_seeds"])
        * 2
    )
    if len(files) != expected and not args.allow_partial:
        raise RuntimeError(f"Stage 1 incomplete: found {len(files)} cell files, expected {expected}")
    if not files:
        raise RuntimeError("no Stage-1 cell files")
    all_rows = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    test = all_rows[all_rows["split"] == "test"].copy()
    unit_records = []
    keys = ["dataset", "problem_type", "model", "seed", "method", "track"]
    for key, frame in test.groupby(keys, sort=True):
        reference = frame[frame["is_reference"]]
        orbit = frame[~frame["is_reference"]]
        if len(reference) != 1 or len(orbit) != int(protocol["orbit_members"]):
            raise RuntimeError(f"incomplete method orbit {key}: ref={len(reference)} orbit={len(orbit)}")
        unit_records.append(
            {
                **dict(zip(keys, key)),
                "disagreement": float(orbit["disagreement"].mean()),
                "max_disagreement": float(orbit["disagreement"].max()),
                "reference_task_error": float(reference["task_error"].iloc[0]),
                "orbit_mean_task_error": float(orbit["task_error"].mean()),
                "worst_orbit_task_error": float(orbit["task_error"].max()),
                "runtime_seconds": float(frame["fit_seconds"].sum()),
                "gpu_peak_memory_mb": float(frame["gpu_peak_memory_mb"].max()),
                "max_input_reconstruction_error": float(frame["max_input_reconstruction_error"].max()),
            }
        )
    units = pd.DataFrame(unit_records)
    raw = units[units["method"] == "AdamW"][
        ["dataset", "model", "seed", "disagreement", "reference_task_error"]
    ].rename(columns={"disagreement": "raw_disagreement", "reference_task_error": "raw_task_error"})
    units = units.merge(raw, on=["dataset", "model", "seed"], how="left", validate="many_to_one")
    units["disagreement_reduction"] = 1.0 - units["disagreement"] / units["raw_disagreement"].clip(lower=1e-12)
    units["relative_task_change"] = (
        units["reference_task_error"] - units["raw_task_error"]
    ) / units["raw_task_error"].abs().clip(lower=1e-12)
    units.to_csv(ROOT / "results" / "processed" / "stage1_units.csv", index=False)

    summary_records = []
    thresholds = protocol["stage1_survival"]
    for (method, track), frame in units.groupby(["method", "track"], sort=True):
        median_reduction = float(frame["disagreement_reduction"].median())
        median_task_change = float(frame["relative_task_change"].median())
        invariance_route = (
            median_reduction >= float(thresholds["invariance_route"]["median_reduction"])
            and median_task_change <= float(thresholds["invariance_route"]["max_task_cost"])
        )
        performance_route = (
            median_task_change < 0.0
            and median_reduction >= -float(thresholds["performance_route"]["max_disagreement_increase"])
        )
        verdict = "CONTROL" if method == "AdamW" else "SURVIVE" if invariance_route or performance_route else "KILL"
        summary_records.append(
            {
                "method": method,
                "track": track,
                "median_disagreement": float(frame["disagreement"].median()),
                "median_disagreement_reduction": median_reduction,
                "median_relative_task_change": median_task_change,
                "median_worst_orbit_task_error": float(frame["worst_orbit_task_error"].median()),
                "runtime_seconds": float(frame["runtime_seconds"].sum()),
                "max_gpu_memory_mb": float(frame["gpu_peak_memory_mb"].max()),
                "units": int(len(frame)),
                "invariance_route": bool(invariance_route),
                "performance_route": bool(performance_route),
                "verdict": verdict,
            }
        )
    summary = pd.DataFrame(summary_records).sort_values(
        ["verdict", "median_disagreement_reduction"], ascending=[False, False]
    )
    summary.to_csv(ROOT / "results" / "processed" / "stage1_summary.csv", index=False)

    coordinate_files = sorted(directory.glob("*coordinate_audit.csv"))
    coordinate = pd.concat([pd.read_csv(path) for path in coordinate_files], ignore_index=True)
    coordinate.to_csv(ROOT / "results" / "processed" / "stage1_coordinate_audit.csv", index=False)
    coordinate_summary = (
        coordinate.groupby("method", as_index=False)[
            ["train_relative_error", "validation_relative_error", "test_relative_error"]
        ]
        .max()
        .rename(columns=lambda value: f"max_{value}" if value != "method" else value)
    )
    coordinate_summary.to_csv(
        ROOT / "results" / "processed" / "stage1_coordinate_audit_summary.csv", index=False
    )
    survivors = summary[summary["verdict"] == "SURVIVE"]["method"].tolist()
    payload = {
        "status": "FROZEN_FROM_STAGE1_DEVELOPMENT_ONLY",
        "complete": len(files) == expected,
        "input_cell_files": [str(path.relative_to(ROOT)) for path in files],
        "input_cell_file_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files
        },
        "protocol_hashes": protocol_hashes(),
        "survivors": survivors,
        "killed": summary[summary["verdict"] == "KILL"]["method"].tolist(),
        "rules": thresholds,
        "summary": summary.to_dict(orient="records"),
    }
    output = ROOT / "configs" / "STAGE1_SURVIVORS.json"
    write_json(output, payload)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    (ROOT / "configs" / "STAGE1_SURVIVORS.sha256").write_text(f"{digest}  STAGE1_SURVIVORS.json\n")
    print(summary.to_string(index=False))
    print(f"survivors={survivors}")


if __name__ == "__main__":
    main()
