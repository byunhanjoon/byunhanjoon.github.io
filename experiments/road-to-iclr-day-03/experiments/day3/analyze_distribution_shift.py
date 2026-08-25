"""Analyze chronological versus row-random controlled-basis sensitivity."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .analyze_broad_phase1 import bootstrap, controlled_pairs, load_phase1
from .distribution_shift_data import shift_config


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/day3/broad_benchmark"


def load_random() -> pd.DataFrame:
    paths = sorted(
        path
        for path in RESULTS.glob("distribution_shift_shard*.csv")
        if not path.stem.endswith("_curves")
    )
    if not paths:
        raise FileNotFoundError("No distribution-shift result shards")
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def split_summary(frame: pd.DataFrame) -> dict[str, object]:
    grouped = frame.groupby("dataset").sensitivity_normalized.mean()
    return {
        "datasets": int(frame.dataset.nunique()),
        "pairs": len(frame),
        "mean_sensitivity_normalized": float(frame.sensitivity_normalized.mean()),
        "median_sensitivity_normalized": float(frame.sensitivity_normalized.median()),
        "harmful_fraction": float((frame.sensitivity_normalized < 0).mean()),
        "dataset_bootstrap_ci": bootstrap(grouped.to_numpy()),
    }


def main() -> None:
    cfg = shift_config()
    mapping = {
        name: spec["source_name"] for name, spec in cfg["datasets"].items()
    }
    random_pairs = controlled_pairs(load_random(), "test_primary")
    random_pairs["source_dataset"] = random_pairs.dataset.map(mapping)
    random_pairs["split_kind"] = "row_random"

    phase_pairs = controlled_pairs(load_phase1(), "test_primary")
    chronological = phase_pairs[
        phase_pairs.dataset.isin(mapping.values())
        & phase_pairs.model.eq("mlp")
        & phase_pairs.remedy.eq("adamw")
    ].copy()
    chronological["source_dataset"] = chronological.dataset
    chronological["split_kind"] = "chronological_purged"

    combined = pd.concat([chronological, random_pairs], ignore_index=True)
    combined.to_csv(RESULTS / "distribution_shift_pairs.csv", index=False)
    wide = combined.pivot_table(
        index=["source_dataset", "task", "model", "remedy", "seed"],
        columns="split_kind",
        values="sensitivity_normalized",
    ).reset_index()
    wide["random_minus_chronological_sensitivity"] = (
        wide.row_random - wide.chronological_purged
    )
    wide.to_csv(RESULTS / "distribution_shift_paired_differences.csv", index=False)
    payload = {
        "chronological_purged": split_summary(chronological),
        "row_random": split_summary(random_pairs),
        "paired_split_difference": {
            "pairs": len(wide),
            "mean_random_minus_chronological_sensitivity": float(
                wide.random_minus_chronological_sensitivity.mean()
            ),
            "median_random_minus_chronological_sensitivity": float(
                wide.random_minus_chronological_sensitivity.median()
            ),
            "random_is_more_harmful_fraction": float(
                (wide.random_minus_chronological_sensitivity < 0).mean()
            ),
        },
        "interpretation_limit": cfg["interpretation_limit"],
    }
    output = RESULTS / "distribution_shift_summary.json"
    output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
