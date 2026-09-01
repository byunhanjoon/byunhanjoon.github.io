#!/usr/bin/env python3
"""Post-hoc diagnosis of unbounded temporal primitives on UCI Bike Sharing."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from tabalu.data import load_hourly_bike_sharing, temporal_bike_split
from tabalu.evaluation import regression_metrics
from tabalu.models.bike_typed import SeasonRoutedBikeProgram, SparseBikeProgram
from tabalu.scripts.run_real_bike_temporal import plot, summarize, write_csv


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def run(config_path: Path) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = (PACKAGE_ROOT / config["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    source = (PACKAGE_ROOT / config["source_results_dir"]).resolve()
    source_records = read_csv(source / "records.csv")
    frame = load_hourly_bike_sharing((PACKAGE_ROOT / config["cache_dir"]).resolve())
    repository = PACKAGE_ROOT.parents[1]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repository, text=True).strip())
    records: list[dict[str, Any]] = []
    for seed in config["seeds"]:
        split = temporal_bike_split(frame, int(seed))
        train_y = split["train"]["cnt"].to_numpy(dtype=float)
        validation_y = split["validation"]["cnt"].to_numpy(dtype=float)
        models = {
            "TabALU-GlobalBoundedTime": SparseBikeProgram(bounded_time_only=True),
            "TabALU-SeasonRouterBoundedTime": SeasonRoutedBikeProgram(bounded_time_only=True),
        }
        for name, model in models.items():
            started = time.perf_counter()
            model.fit(split["train"], train_y, (split["validation"], validation_y))
            elapsed = time.perf_counter() - started
            for split_name in ("iid", "future"):
                targets = split[split_name]["cnt"].to_numpy(dtype=float)
                records.append(
                    {
                        "git_commit": commit,
                        "git_dirty": dirty,
                        "dataset": "UCI Bike Sharing (hourly)",
                        "seed": seed,
                        "model": name,
                        "split": split_name,
                        "training_seconds": elapsed,
                        "operation_count": model.operation_count,
                        **regression_metrics(targets, model.predict(split[split_name])),
                    }
                )
        print(f"completed seed {int(seed) + 1}/{len(config['seeds'])}", flush=True)
    write_csv(output / "records.csv", records)
    summary = summarize(records)
    write_csv(output / "summary.csv", summary)
    combined_records = source_records + [{key: str(value) for key, value in row.items()} for row in records]
    combined_summary = summarize(combined_records)
    keep = {
        "CatBoost",
        "XGBoost",
        "TabALU-Global",
        "TabALU-SeasonRouter",
        "TabALU-GlobalBoundedTime",
        "TabALU-SeasonRouterBoundedTime",
    }
    plot([row for row in combined_summary if row["model"] in keep], output / "bounded_time_diagnostic.png")
    lookup = {(row["model"], row["split"]): float(row["nrmse_mean"]) for row in combined_summary}
    observations = {
        "bounded_router_vs_unrestricted_router_future_ratio": lookup[("TabALU-SeasonRouterBoundedTime", "future")]
        / lookup[("TabALU-SeasonRouter", "future")],
        "bounded_router_vs_bounded_global_future_ratio": lookup[("TabALU-SeasonRouterBoundedTime", "future")]
        / lookup[("TabALU-GlobalBoundedTime", "future")],
        "bounded_router_vs_best_tree_future_ratio": lookup[("TabALU-SeasonRouterBoundedTime", "future")]
        / min(lookup[("CatBoost", "future")], lookup[("XGBoost", "future")]),
    }
    expected = len(config["seeds"]) * 2 * 2
    audit = {
        "git_commit": commit,
        "git_dirty": dirty,
        "confirmatory": False,
        "reason_posthoc": "bounded temporal basis chosen after inspecting unrestricted UCI failure",
        "expected_records": expected,
        "observed_records": len(records),
        "all_finite": all(math.isfinite(float(row["nrmse"])) for row in records),
        "observations_not_gates": observations,
    }
    audit["audit_passed"] = expected == len(records) and audit["all_finite"]
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    run(args.config.resolve())


if __name__ == "__main__":
    main()
