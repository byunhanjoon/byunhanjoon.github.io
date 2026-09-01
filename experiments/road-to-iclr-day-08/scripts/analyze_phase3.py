#!/usr/bin/env python3
"""Run prespecified cross-dataset descriptor analyses after Phase II completes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.io import code_digest  # noqa: E402
from src.analysis.phase3 import aggregate_descriptors, cross_dataset_meta_models  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-sha256")
    parser.add_argument("--phase2-dir", type=Path)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--transforms", nargs="+")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow descriptor screening from a PARTIAL.json Phase II snapshot.",
    )
    args = parser.parse_args()
    selected_code = args.code_sha256 or code_digest(ROOT)
    phase2 = args.phase2_dir.resolve() if args.phase2_dir else ROOT / "results" / "analysis" / "phase2" / selected_code[:16]
    complete = phase2 / "DONE.json"
    partial = phase2 / "PARTIAL.json"
    if not complete.exists() and not (args.allow_partial and partial.exists()):
        raise RuntimeError(f"Phase II analysis is incomplete: {phase2}")
    features = pd.read_csv(phase2 / "feature_descriptors.csv")
    effects = pd.read_csv(phase2 / "dataset_transform_effects.csv")
    if args.models:
        effects = effects[effects["model"].isin(args.models)]
    if args.transforms:
        effects = effects[effects["transform"].isin(args.transforms)]
    descriptors = aggregate_descriptors(features)
    # The unit is dataset/split. Average transform/model cells only after retaining
    # their labels, so analyses can be stratified from the saved joined table.
    joined = effects.merge(descriptors, on=["dataset", "split_seed"], validate="many_to_one")
    status = "exploratory_partial" if partial.exists() else "complete_confirmatory"
    output = phase2 / "phase3_descriptors"
    output.mkdir(exist_ok=True)
    joined.to_csv(output / "descriptor_effects.csv", index=False)
    metrics, details = [], []
    for target in ("matched_normalized_loss_gap", "matched_excess_disagreement"):
        for (model, transform), group in joined.groupby(["model", "transform"], sort=True):
            if group.dataset.nunique() < 5:
                continue
            score, detail = cross_dataset_meta_models(group, target)
            score.insert(0, "model", model); score.insert(1, "transform", transform)
            detail.insert(0, "model", model); detail.insert(1, "transform", transform); detail.insert(2, "target_name", target)
            metrics.append(score); details.append(detail)
    if not metrics:
        raise RuntimeError("No model/transform cell has at least five observed datasets")
    pd.concat(metrics, ignore_index=True).to_csv(output / "cross_dataset_metrics.csv", index=False)
    pd.concat(details, ignore_index=True).to_csv(output / "cross_dataset_predictions_and_importances.csv", index=False)
    (output / ("PARTIAL.json" if status == "exploratory_partial" else "DONE.json")).write_text(
        json.dumps(
            {
                "analysis_status": status,
                "phase2_input": str(phase2),
                "models": args.models or "all",
                "transforms": args.transforms or "all",
                "gate_g2_eligible": status == "complete_confirmatory",
                "warning": (
                    "Descriptor results are screening evidence only because Phase II coverage is incomplete."
                    if status == "exploratory_partial"
                    else None
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(output)


if __name__ == "__main__":
    main()
