"""Compare deterministic canonicalization with Tier-1 raw and quotient fits."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
MODELS = ("ordinal_forest", "native_histgb", "catboost_native", "onehot_adam_mlp")


def proper_loss(y: np.ndarray, prediction: np.ndarray) -> float:
    if prediction.shape[-1] == 1:
        return float(np.mean((prediction[..., 0] - y) ** 2))
    targets = np.eye(prediction.shape[-1])[y.astype(int)]
    return float(np.mean(np.sum((prediction - targets) ** 2, axis=-1)))


def analyze_cell(canonical_path: Path, raw_path: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    canonical = np.load(canonical_path)
    raw = np.load(raw_path)
    rows = []
    for split in ("validation", "test"):
        cp = canonical[f"{split}_predictions"].astype(np.float64)
        rp = raw[f"{split}_predictions"].astype(np.float64)
        y = canonical[f"{split}_y"]
        raw_flat = rp.reshape((-1,) + rp.shape[-2:])
        raw_centroid = raw_flat.mean(axis=0)
        raw_joint_risk = float(np.mean(np.sum((raw_flat - raw_centroid) ** 2, axis=-1)))
        raw_mean_member = float(np.mean([proper_loss(y, item) for item in raw_flat]))
        raw_identity_members = rp[0, 0, 0]
        canonical_member = float(np.mean([proper_loss(y, item) for item in cp]))
        canonical_seed_ensemble = proper_loss(y, cp.mean(axis=0))
        raw_identity_member = float(np.mean([proper_loss(y, item) for item in raw_identity_members]))
        raw_identity_seed_ensemble = proper_loss(y, raw_identity_members.mean(axis=0))
        raw_quotient = proper_loss(y, raw_centroid)
        rows.append({
            "dataset": manifest["dataset"],
            "model": manifest["model"],
            "task": manifest["task"],
            "split": split,
            "checked_orbit_views": manifest["checked_orbit_views"],
            "unique_canonical_input_digests": manifest["unique_canonical_input_digests"],
            "canonical_mean_member_loss": canonical_member,
            "canonical_seed_ensemble_loss": canonical_seed_ensemble,
            "raw_identity_mean_member_loss": raw_identity_member,
            "raw_identity_seed_ensemble_loss": raw_identity_seed_ensemble,
            "raw_full_quotient_loss": raw_quotient,
            "raw_full_mean_member_loss": raw_mean_member,
            "raw_joint_schema_seed_risk": raw_joint_risk,
            "raw_ambiguity_identity_error": abs(raw_mean_member - raw_quotient - raw_joint_risk),
            "canonical_vs_raw_identity_member_relative_loss": (
                canonical_member - raw_identity_member
            ) / raw_identity_member,
            "canonical_vs_raw_identity_seed_ensemble_relative_loss": (
                canonical_seed_ensemble - raw_identity_seed_ensemble
            ) / raw_identity_seed_ensemble,
            "canonical_vs_raw_full_quotient_relative_loss": (
                canonical_seed_ensemble - raw_quotient
            ) / raw_quotient,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=HERE / "tier1_config.json")
    parser.add_argument("--canonical-dir", type=Path, default=HERE / "results" / "canonical_orbit")
    parser.add_argument("--raw-dir", type=Path, default=HERE / "results" / "tier1")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    rows = []
    missing = []
    for dataset, model in itertools.product(config["datasets"], MODELS):
        stem = f"{dataset}__{model}"
        canonical_path = args.canonical_dir / f"{stem}.npz"
        raw_path = args.raw_dir / f"{stem}.npz"
        manifest_path = args.canonical_dir / f"{stem}.json"
        if not canonical_path.exists() or not raw_path.exists() or not manifest_path.exists():
            missing.append(stem)
            continue
        rows.extend(analyze_cell(canonical_path, raw_path, json.loads(manifest_path.read_text())))
    if missing:
        raise RuntimeError(f"missing cells: {missing}")
    frame = pd.DataFrame(rows)
    test = frame[frame.split == "test"]
    equal_compute_wins = int((test.canonical_seed_ensemble_loss < test.raw_identity_seed_ensemble_loss).sum())
    rope = float(config["proper_loss_rope_relative"])
    summary = {
        "status": "complete",
        "cells": len(test),
        "all_orbits_canonicalized_to_one_digest": bool((test.unique_canonical_input_digests == 1).all()),
        "total_checked_views": int(test.checked_orbit_views.sum()),
        "equal_compute_cells_canonical_better_than_identity_seed_ensemble": equal_compute_wins,
        "equal_compute_cells_within_relative_loss_rope": int((test.canonical_vs_raw_identity_seed_ensemble_relative_loss <= rope).sum()),
        "mean_equal_compute_relative_loss_change": float(test.canonical_vs_raw_identity_seed_ensemble_relative_loss.mean()),
        "median_equal_compute_relative_loss_change": float(test.canonical_vs_raw_identity_seed_ensemble_relative_loss.median()),
        "canonical_cells_beating_full_32_schema_x_4_seed_quotient": int((test.canonical_seed_ensemble_loss < test.raw_full_quotient_loss).sum()),
        "mean_relative_loss_vs_full_quotient": float(test.canonical_vs_raw_full_quotient_relative_loss.mean()),
        "maximum_raw_ambiguity_identity_error": float(frame.raw_ambiguity_identity_error.max()),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "canonical_orbit_comparison.csv", index=False)
    (args.output_dir / "canonical_orbit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

