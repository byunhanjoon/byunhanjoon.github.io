"""Evaluate the frozen untouched Stage-1 gate without changing its thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--selections",
        type=Path,
        default=HERE / "results" / "tabarena_nested_screen_frozen_v3" / "selections.csv",
    )
    parser.add_argument(
        "--protocol", type=Path, default=HERE / "configs" / "frozen_protocol.json"
    )
    parser.add_argument(
        "--output", type=Path, default=HERE / "results" / "stage1_gate.json"
    )
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    thresholds = protocol["untouched_stage1"]["screen_gate"]
    raw = pd.concat([pd.read_csv(path) for path in args.inputs], ignore_index=True)
    key = ["dataset", "fold", "model", "seed"]
    baseline = raw[raw.representation.eq("baseline_ple")][key + ["test_error"]].rename(
        columns={"test_error": "baseline_error"}
    )
    paired = raw.merge(baseline, on=key)
    paired["gain_pct"] = np.where(
        paired.task.eq("binclass"),
        100.0 * (paired.baseline_error - paired.test_error),
        100.0 * (paired.baseline_error - paired.test_error) / paired.baseline_error,
    )
    adaptive = paired[paired.representation.eq("adaptive_atomic")].copy()
    random = paired[paired.representation.str.startswith("matched_random")]
    random_mean = random.groupby(key, as_index=False).gain_pct.mean().rename(
        columns={"gain_pct": "random_gain_pct"}
    )
    adaptive = adaptive.merge(random_mean, on=key, how="left")
    adaptive["beats_random"] = adaptive.gain_pct > adaptive.random_gain_pct
    selections = pd.read_csv(args.selections)
    planned_datasets = set(selections.dataset.unique())
    evaluated_datasets = set(adaptive.dataset.unique())
    full_downstream_screen = planned_datasets <= evaluated_datasets
    selected_datasets = selections.loc[selections.nonempty.eq(1), "dataset"].unique()
    selected_runs = adaptive[adaptive.dataset.isin(selected_datasets)]
    abstentions = adaptive[~adaptive.dataset.isin(selected_datasets)]

    observed = {
        "datasets_selecting_nonempty_structure": int(len(selected_datasets)),
        "selected_datasets": sorted(selected_datasets.tolist()),
        "downstream_datasets_evaluated": sorted(evaluated_datasets),
        "full_downstream_screen_completed": full_downstream_screen,
        "mean_gain_pct_on_evaluated_datasets": float(adaptive.gain_pct.mean()),
        "fraction_selected_runs_beating_matched_random": (
            float(selected_runs.beats_random.mean()) if len(selected_runs) else 0.0
        ),
        "maximum_absolute_mean_abstention_gain_pct": (
            float(
                abstentions.groupby("dataset").gain_pct.mean().abs().max()
            )
            if len(abstentions)
            else 0.0
        ),
    }
    checks = {
        "enough_nonempty_datasets": observed["datasets_selecting_nonempty_structure"]
        >= thresholds["minimum_datasets_selecting_nonempty_structure"],
        "full_downstream_screen": full_downstream_screen,
        "positive_mean_gain": full_downstream_screen
        and observed["mean_gain_pct_on_evaluated_datasets"]
        > thresholds["minimum_mean_gain_pct"],
        "beats_matched_random": full_downstream_screen and observed[
            "fraction_selected_runs_beating_matched_random"
        ]
        >= thresholds["minimum_fraction_selected_runs_beating_matched_random"],
        "safe_abstention": full_downstream_screen
        and observed["maximum_absolute_mean_abstention_gain_pct"]
        <= thresholds["maximum_mean_abstention_regression_pct"],
    }
    report = {
        "protocol_frozen_at": protocol["frozen_at"],
        "thresholds": thresholds,
        "observed": observed,
        "checks": checks,
        "gate_passed": bool(all(checks.values())),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
