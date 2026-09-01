#!/usr/bin/env python3
"""Aggregate the explicitly exploratory condition<=3 experiment."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_stage2 import aggregate, summarize  # noqa: E402
from tournament.common import load_json, write_json  # noqa: E402


def main() -> None:
    processed = ROOT / "results" / "processed"
    files = sorted((processed / "condition_exploratory").glob("*.csv"))
    if not files:
        raise RuntimeError("no condition<=3 cells")
    config = load_json(ROOT / "configs" / "FINALIST_CONFIGS.json")
    track_map = {item["method_id"]: item["type"] for item in config["finalists"]}
    frames = []
    for path in files:
        frame = pd.read_csv(path)
        frame["track"] = frame["method"].map(track_map).fillna("exploratory_interface")
        frame.loc[frame["method"] == "Raw", "track"] = "baseline"
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined_path = processed / "condition_exploratory_rows.csv"
    combined.to_csv(combined_path, index=False)
    units = aggregate([combined_path], "Raw")
    units["predictive_rank"] = units.groupby(
        ["dataset", "model", "seed", "split"], sort=False
    )["task_error"].rank(method="average", ascending=True)
    summary = summarize(units)
    units.to_csv(processed / "condition_exploratory_units.csv", index=False)
    summary.to_csv(processed / "condition_exploratory_summary.csv", index=False)
    write_json(
        processed / "condition_exploratory_analysis.json",
        {
            "scope": "exploratory; three development datasets; seed 0; condition number <=3",
            "files": len(files),
            "summary": summary.to_dict(orient="records"),
        },
    )
    print(summary.sort_values("paper_method_score", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
