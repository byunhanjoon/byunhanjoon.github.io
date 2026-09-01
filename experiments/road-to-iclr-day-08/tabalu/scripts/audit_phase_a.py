#!/usr/bin/env python3
"""Recompute clustered summaries and validate Phase-A artifact completeness."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from tabalu.scripts.run_phase_a import gate_decision, plot_curves, summarize, write_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    output = args.results.resolve()
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    with (output / "records.csv").open(encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    with (output / "program_recovery.csv").open(encoding="utf-8") as handle:
        recovery = list(csv.DictReader(handle))
    expected_models = [*config["baselines"], "TabALU-soft", "TabALU-hard", "TabALU-compiled"]
    expected_cells = (
        int(config["n_tasks"])
        * len(config["seeds"])
        * len(expected_models)
        * len(config["multipliers"])
    )
    cell_keys = {
        (row["dataset"], row["seed"], row["model"], row["extrapolation_multiplier"])
        for row in records
    }
    finite = all(
        math.isfinite(float(row[key]))
        for row in records
        for key in ("mae", "rmse", "r2", "nrmse", "relative_error")
    )
    checks = {
        "record_count": len(records) == expected_cells,
        "unique_cell_count": len(cell_keys) == expected_cells,
        "all_status_ok": Counter(row["status"] for row in records) == {"ok": expected_cells},
        "all_metrics_finite": finite,
        "task_json_count": len(list((output / "tasks").glob("*.json"))) == int(config["n_tasks"]),
        "program_json_count": len(list((output / "programs").glob("*.json")))
        == int(config["n_tasks"]) * len(config["seeds"]),
        "history_json_count": len(list((output / "histories").glob("*.json")))
        == int(config["n_tasks"]) * len(config["seeds"]),
        "recovery_row_count": len(recovery) == int(config["n_tasks"]) * len(config["seeds"]),
        "failure_log_empty": json.loads((output / "failures.json").read_text(encoding="utf-8")) == [],
    }
    summary = summarize(records)
    write_csv(output / "summary.csv", summary)
    plot_curves(summary, output / "extrapolation_curve.png")
    gate = gate_decision(summary, [float(value) for value in config["multipliers"]])
    exact = sum(row["exact_program"] == "True" for row in recovery)
    structural = {
        "exact_program_fraction": exact / max(len(recovery), 1),
        "feature_f1_mean": sum(float(row["feature_f1"]) for row in recovery) / max(len(recovery), 1),
        "operator_accuracy_mean": sum(float(row["operator_accuracy"]) for row in recovery)
        / max(len(recovery), 1),
    }
    audit = json.loads((output / "audit.json").read_text(encoding="utf-8"))
    audit["completeness"] = {
        "expected_cells": expected_cells,
        "observed_cells": len(records),
        "checks": checks,
        "passed": all(checks.values()),
    }
    audit["structural_recovery"] = structural
    audit["gate"] = gate
    audit["confidence_interval_unit"] = "independent task (seed metrics averaged within task)"
    audit["audit_passed"] = all(checks.values()) and bool(gate["passed"])
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if not audit["audit_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
