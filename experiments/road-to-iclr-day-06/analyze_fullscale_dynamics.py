"""Descriptive H3 threshold-crossing and delayed-instability analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_semantic_arithmetic as h1_analysis

HERE = Path(__file__).resolve().parent


def analyze(input_dir: Path, output_dir: Path, threshold: float = 1e-5) -> dict:
    rows = []
    artifacts = sorted(input_dir.glob("*.npz"))
    if not artifacts:
        raise FileNotFoundError(f"no H3 artifacts under {input_dir}")
    for artifact in artifacts:
        trajectories, _ = h1_analysis.rows_from_artifact(artifact)
        frame = pd.DataFrame(trajectories)
        for keys, current in frame.groupby(["dataset", "model", "seed", "precision", "action"]):
            current = current.sort_values("checkpoint")
            values = current.validation_prediction_mse.to_numpy()
            checkpoints = current.checkpoint.to_numpy(dtype=int)
            hits = checkpoints[values > threshold]
            at_20 = float(current.loc[current.checkpoint == 20, "validation_prediction_mse"].iloc[0])
            at_200 = float(current.loc[current.checkpoint == 200, "validation_prediction_mse"].iloc[0])
            rows.append({
                "dataset": keys[0], "model": keys[1], "seed": int(keys[2]),
                "precision": keys[3], "action": int(keys[4]),
                "mse_epoch_20": at_20, "mse_epoch_200": at_200,
                "first_material_checkpoint": int(hits[0]) if len(hits) else 201,
                "material_epoch_20": bool(at_20 > threshold),
                "material_epoch_200": bool(at_200 > threshold),
                "delayed_after_epoch_20": bool(at_20 <= threshold < at_200),
                "exact_all_checkpoints": bool(np.all(values == 0)),
            })
    paths = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths.to_csv(output_dir / "h3_dynamics_paths.csv", index=False)
    bundles = paths.groupby(
        ["dataset", "model", "seed", "precision"], as_index=False
    ).agg(
        mean_mse_epoch_20=("mse_epoch_20", "mean"),
        mean_mse_epoch_200=("mse_epoch_200", "mean"),
        median_first_material_checkpoint=("first_material_checkpoint", "median"),
        material_paths_epoch_20=("material_epoch_20", "sum"),
        material_paths_epoch_200=("material_epoch_200", "sum"),
        delayed_paths=("delayed_after_epoch_20", "sum"),
        exact_paths=("exact_all_checkpoints", "sum"),
        paths=("action", "size"),
    )
    bundles.to_csv(output_dir / "h3_dynamics_bundles.csv", index=False)
    fp32 = bundles[bundles.precision == "fp32"]
    iea = bundles[bundles.precision == "iea64"]
    summary = {
        "status": "descriptive_partial" if len(artifacts) < 36 else "descriptive_complete",
        "bundles": len(artifacts), "threshold": threshold,
        "fp32_material_bundles_epoch_20": int((fp32.material_paths_epoch_20 > 0).sum()),
        "fp32_material_bundles_epoch_200": int((fp32.material_paths_epoch_200 > 0).sum()),
        "fp32_delayed_bundles": int((fp32.delayed_paths > 0).sum()),
        "iea64_exact_bundles": int((iea.exact_paths == iea.paths).sum()),
    }
    (output_dir / "h3_dynamics_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=HERE / "results" / "h3")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument("--threshold", type=float, default=1e-5)
    args = parser.parse_args()
    analyze(args.input_dir, args.output_dir, args.threshold)


if __name__ == "__main__":
    main()
