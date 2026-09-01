"""Exact strength-1 orthogonal-cover analysis on saved Tier-1 tensors."""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
IDEA = HERE.parent / "road-to-iclr-idea-search"
sys.path.insert(0, str(IDEA))
from orbit_anova import decompose  # noqa: E402


def proper_loss(y: np.ndarray, prediction: np.ndarray) -> float:
    if prediction.shape[-1] == 1:
        return float(np.mean((prediction[..., 0] - y) ** 2))
    targets = np.eye(prediction.shape[-1])[y.astype(int)]
    return float(np.mean(np.sum((prediction - targets) ** 2, axis=-1)))


def orthogonal_designs(class_levels: int, category_levels: int = 4) -> np.ndarray:
    permutations = list(itertools.permutations(range(4)))
    if category_levels == 4:
        category_assignments = permutations
    elif category_levels == 2:
        category_assignments = []
        for positive in itertools.combinations(range(4), 2):
            assignment = np.zeros(4, dtype=int)
            assignment[list(positive)] = 1
            category_assignments.append(assignment)
    elif category_levels == 1:
        category_assignments = [tuple(0 for _ in range(4))]
    else:
        raise ValueError("category factor must have four, two, or one levels")
    if class_levels == 4:
        class_assignments = permutations
    elif class_levels == 2:
        class_assignments = []
        for positive in itertools.combinations(range(4), 2):
            assignment = np.zeros(4, dtype=int)
            assignment[list(positive)] = 1
            class_assignments.append(assignment)
    elif class_levels == 1:
        class_assignments = [np.zeros(4, dtype=int)]
    else:
        raise ValueError("only four-level, binary, or singleton target factors are supported")
    designs = []
    for feature in permutations:
        for category in category_assignments:
            for classes in class_assignments:
                designs.append(np.stack((feature, category, classes), axis=-1))
    return np.asarray(designs, dtype=int)


def exact_cover_residual(predictions: np.ndarray, designs: np.ndarray, batch_size: int = 64) -> float:
    # predictions: [feature=4, category=4, class, seed=4, row, output]
    centroid = predictions.mean(axis=(0, 1, 2, 3), dtype=np.float64)
    seed = np.arange(4)[None, :]
    total = 0.0
    count = 0
    for start in range(0, len(designs), batch_size):
        current = designs[start : start + batch_size]
        members = predictions[
            current[:, :, 0], current[:, :, 1], current[:, :, 2], seed
        ].astype(np.float64)
        estimates = members.mean(axis=1)
        squared = np.sum((estimates - centroid[None, ...]) ** 2, axis=-1)
        total += float(squared.sum())
        count += squared.size
    return total / count


def analyze_cell(
    raw_path: Path,
    raw_manifest: dict[str, Any],
    canonical_path: Path | None,
) -> list[dict[str, Any]]:
    raw = np.load(raw_path)
    canonical = np.load(canonical_path) if canonical_path is not None else None
    rows = []
    for split in ("validation", "test"):
        predictions = raw[f"{split}_predictions"].astype(np.float64)
        y = raw[f"{split}_y"]
        designs = orthogonal_designs(predictions.shape[2], predictions.shape[1])
        quotient = predictions.mean(axis=(0, 1, 2, 3))
        flat = predictions.reshape((-1,) + predictions.shape[-2:])
        joint_risk = float(np.mean(np.sum((flat - quotient) ** 2, axis=-1)))
        iid_residual = joint_risk / 4
        seed_averaged = predictions.mean(axis=3)
        seed_flat = seed_averaged.reshape((-1,) + seed_averaged.shape[-2:])
        persistent_residual = float(np.mean(np.sum((seed_flat - quotient) ** 2, axis=-1)))
        cover_residual = exact_cover_residual(predictions, designs)
        quotient_loss = proper_loss(y, quotient)
        mean_member_loss = float(np.mean([proper_loss(y, item) for item in flat]))
        identity_seed_loss = proper_loss(y, predictions[0, 0, 0].mean(axis=0))
        canonical_seed_loss = (
            proper_loss(y, canonical[f"{split}_predictions"].astype(np.float64).mean(axis=0))
            if canonical is not None else np.nan
        )
        anova = decompose(predictions, ("feature", "category", "class", "seed"))
        main_risk = sum(float(anova[name]) for name in ("feature", "category", "class", "seed"))
        rows.append({
            "dataset": raw_manifest["dataset"],
            "model": raw_manifest["model"],
            "task": raw_manifest["task"],
            "split": split,
            "design_count": len(designs),
            "joint_schema_seed_risk": joint_risk,
            "joint_main_effect_risk": main_risk,
            "joint_main_effect_fraction": main_risk / joint_risk if joint_risk else 0.0,
            "mean_member_loss": mean_member_loss,
            "full_quotient_loss": quotient_loss,
            "iid_joint_expected_residual": iid_residual,
            "seed_only_expected_residual": persistent_residual,
            "orthogonal_cover_expected_residual": cover_residual,
            "orthogonal_cover_expected_loss": quotient_loss + cover_residual,
            "iid_joint_expected_loss": quotient_loss + iid_residual,
            "seed_only_expected_loss": quotient_loss + persistent_residual,
            "identity_four_seed_loss": identity_seed_loss,
            "canonical_four_seed_loss": canonical_seed_loss,
            "cover_vs_iid_residual_reduction": 1 - cover_residual / iid_residual if iid_residual else 0.0,
            "cover_vs_seed_residual_reduction": 1 - cover_residual / persistent_residual if persistent_residual else 0.0,
            "joint_risk_fraction_of_member_loss": joint_risk / mean_member_loss if mean_member_loss else 0.0,
            "ambiguity_identity_error": abs(mean_member_loss - quotient_loss - joint_risk),
            "anova_component_sum_error": float(anova["component_sum_error"]),
        })
    return rows


def exact_two_sided_sign_p(wins: int, total: int) -> float:
    tail = min(wins, total - wins)
    return float(min(1.0, 2 * sum(math.comb(total, k) for k in range(tail + 1)) / 2**total))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=HERE / "tier1_config.json")
    parser.add_argument("--raw-dir", type=Path, default=HERE / "results" / "tier1")
    parser.add_argument("--canonical-dir", type=Path, default=HERE / "results" / "canonical_orbit")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument("--output-prefix", default="joint_orthogonal_cover")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    rows = []
    missing = []
    for dataset, model in itertools.product(config["datasets"], config["models"]):
        stem = f"{dataset}__{model}"
        raw_path = args.raw_dir / f"{stem}.npz"
        manifest_path = args.raw_dir / f"{stem}.json"
        canonical_path = args.canonical_dir / f"{stem}.npz"
        if not raw_path.exists() or not manifest_path.exists():
            missing.append(stem)
            continue
        rows.extend(analyze_cell(
            raw_path, json.loads(manifest_path.read_text()),
            canonical_path if canonical_path.exists() else None,
        ))
    if missing:
        raise RuntimeError(f"missing raw cells: {missing}")
    frame = pd.DataFrame(rows)
    test = frame[frame.split == "test"]
    material = test[test.joint_risk_fraction_of_member_loss >= 0.005]
    wins_iid = int((material.orthogonal_cover_expected_residual < material.iid_joint_expected_residual).sum())
    wins_seed = int((material.orthogonal_cover_expected_residual < material.seed_only_expected_residual).sum())
    wins_both = int((
        (material.orthogonal_cover_expected_residual < material.iid_joint_expected_residual)
        & (material.orthogonal_cover_expected_residual < material.seed_only_expected_residual)
    ).sum())
    mean_cover = float(material.orthogonal_cover_expected_residual.mean())
    mean_iid = float(material.iid_joint_expected_residual.mean())
    mean_seed = float(material.seed_only_expected_residual.mean())
    gate = bool(
        len(material) and wins_both > len(material) / 2
        and mean_cover < mean_iid and mean_cover < mean_seed
    )
    source_groups = config.get("source_groups")
    group_summary = None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if source_groups:
        grouped_rows = []
        for group_name in sorted(set(source_groups.values())):
            datasets = [name for name, group in source_groups.items() if group == group_name]
            group_cells = material[material.dataset.isin(datasets)]
            if not len(group_cells):
                grouped_rows.append({"source_group": group_name, "material_cells": 0, "beats_both": False})
                continue
            cover_mean = float(group_cells.orthogonal_cover_expected_residual.mean())
            iid_mean = float(group_cells.iid_joint_expected_residual.mean())
            seed_mean = float(group_cells.seed_only_expected_residual.mean())
            grouped_rows.append({
                "source_group": group_name, "material_cells": len(group_cells),
                "mean_cover_residual": cover_mean, "mean_iid_residual": iid_mean,
                "mean_seed_residual": seed_mean,
                "beats_both": bool(cover_mean < iid_mean and cover_mean < seed_mean),
            })
        groups = pd.DataFrame(grouped_rows)
        group_wins = int(groups.beats_both.sum())
        group_summary = {
            "source_groups": len(groups),
            "source_groups_with_material_cells": int((groups.material_cells > 0).sum()),
            "source_groups_cover_beats_both": group_wins,
            "source_group_gate_passed": bool(group_wins >= 4 and mean_cover < mean_iid and mean_cover < mean_seed),
        }
        groups.to_csv(args.output_dir / f"{args.output_prefix}_source_groups.csv", index=False)
    summary = {
        "status": "complete",
        "cells": len(test),
        "material_cells": len(material),
        "orthogonal_cover_designs_classification": 3456,
        "material_cells_cover_beats_iid": wins_iid,
        "material_cells_cover_beats_seed_only": wins_seed,
        "material_cells_cover_beats_both": wins_both,
        "exact_two_sided_sign_p_cover_beats_both": exact_two_sided_sign_p(wins_both, len(material)) if len(material) else np.nan,
        "mean_cover_vs_iid_residual_reduction_material": 1 - mean_cover / mean_iid if mean_iid else np.nan,
        "mean_cover_vs_seed_residual_reduction_material": 1 - mean_cover / mean_seed if mean_seed else np.nan,
        "adaptive_orthogonal_cover_gate_passed": gate,
        "mean_joint_main_effect_fraction_material": float(material.joint_main_effect_fraction.mean()) if len(material) else np.nan,
        "maximum_ambiguity_identity_error": float(frame.ambiguity_identity_error.max()),
        "maximum_anova_component_sum_error": float(frame.anova_component_sum_error.max()),
        "source_group_confirmation": group_summary,
    }
    frame.to_csv(args.output_dir / f"{args.output_prefix}_cells.csv", index=False)
    (args.output_dir / f"{args.output_prefix}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
