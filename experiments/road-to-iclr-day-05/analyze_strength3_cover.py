"""Exact budget-64 strength-3 cover analysis from saved full tensors."""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S2 = load_module("strength2_for_strength3", "analyze_strength2_cover.py")
ORBIT_SPEC = importlib.util.spec_from_file_location(
    "orbit_anova_strength3", HERE.parent / "road-to-iclr-idea-search" / "orbit_anova.py"
)
assert ORBIT_SPEC is not None and ORBIT_SPEC.loader is not None
ORBIT = importlib.util.module_from_spec(ORBIT_SPEC)
ORBIT_SPEC.loader.exec_module(ORBIT)


def strength3_base(category_levels: int, class_levels: int, seed_levels: int = 4) -> np.ndarray:
    if seed_levels != 4:
        raise ValueError("strength-3 construction currently requires four seeds")
    rows = []
    for u, v, w in itertools.product(range(4), repeat=3):
        if category_levels == 4:
            category = v
        elif category_levels == 2:
            category = S2.gf4_trace(v)
        elif category_levels == 1:
            category = 0
        else:
            raise ValueError("category factor must have four, two, or one levels")
        linear = u ^ int(S2.MUL4[2, v]) ^ int(S2.MUL4[3, w])
        if class_levels == 4:
            label = linear
        elif class_levels == 2:
            label = S2.gf4_trace(linear)
        elif class_levels == 1:
            label = 0
        else:
            raise ValueError("class factor must have four, two, or one levels")
        rows.append((u, category, label, w))
    output = np.asarray(rows, dtype=int)
    S2.assert_strength(output, (4, category_levels, class_levels, seed_levels), 3)
    return output


def strength3_family(category_levels: int, class_levels: int, seed_levels: int = 4) -> np.ndarray:
    base = strength3_base(category_levels, class_levels, seed_levels)
    category_perms = (
        S2.PERMS4 if category_levels == 4
        else ((0, 1), (1, 0)) if category_levels == 2
        else ((0,),)
    )
    class_perms = S2.PERMS4 if class_levels == 4 else ((0, 1), (1, 0)) if class_levels == 2 else ((0,),)
    designs = []
    for feature_perm in S2.PERMS4:
        for category_perm in category_perms:
            for class_perm in class_perms:
                for seed_perm in S2.PERMS4:
                    current = base.copy()
                    current[:, 0] = np.asarray(feature_perm)[current[:, 0]]
                    current[:, 1] = np.asarray(category_perm)[current[:, 1]]
                    current[:, 2] = np.asarray(class_perm)[current[:, 2]]
                    current[:, 3] = np.asarray(seed_perm)[current[:, 3]]
                    designs.append(current)
    return np.asarray(designs, dtype=np.int8)


def exact_sign_p(wins: int, total: int) -> float:
    tail = min(wins, total - wins)
    return float(min(1, 2 * sum(math.comb(total, k) for k in range(tail + 1)) / 2**total))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    cache = {}
    rows = []
    for dataset, model in itertools.product(config["datasets"], config["models"]):
        stem = f"{dataset}__{model}"
        archive = np.load(args.input_dir / f"{stem}.npz")
        manifest = json.loads((args.input_dir / f"{stem}.json").read_text())
        for split in ("validation", "test"):
            predictions = archive[f"{split}_predictions"].astype(np.float64)
            y = archive[f"{split}_y"]
            cardinalities = predictions.shape[:4]
            key = cardinalities[1:]
            if key not in cache:
                family1 = S2.strength1_family(*key)
                family2 = S2.strength2_family(*key)
                family3 = strength3_family(*key)
                cov1 = S2.incidence_covariance(family1, cardinalities)
                cov2 = S2.incidence_covariance(family2, cardinalities)
                cov3 = S2.incidence_covariance(family3, cardinalities)
                cache[key] = (
                    cov1, cov2, cov3, len(family3),
                    S2.component_coefficients(cov3, cardinalities),
                )
            cov1, cov2, cov3, design_count, coefficients3 = cache[key]
            risk1 = S2.expected_residual(predictions, cov1)
            risk2 = S2.expected_residual(predictions, cov2)
            risk3 = S2.expected_residual(predictions, cov3)
            flat = predictions.reshape((-1,) + predictions.shape[-2:])
            quotient = flat.mean(axis=0)
            joint = float(np.mean(np.sum((flat - quotient) ** 2, axis=-1)))
            seed_average = predictions.mean(axis=3)
            persistent = float(np.mean(np.sum((seed_average - quotient) ** 2, axis=-1)))
            member_loss = float(np.mean([S2.proper_loss(y, member) for member in flat]))
            quotient_loss = S2.proper_loss(y, quotient)
            anova = ORBIT.decompose(predictions, ("feature", "category", "class", "seed"))
            component_names = [name for name in coefficients3]
            through_triple = sum(
                float(anova[name]) for name in component_names if name.count(":") <= 2
            )
            reconstructed = sum(coefficients3[name] * float(anova[name]) for name in coefficients3)
            comparators = {
                "iid64_residual": joint / 64,
                "four_strength2_residual": risk2 / 4,
                "sixteen_strength1_residual": risk1 / 16,
                "sixteen_seed_blocks_residual": persistent / 16,
            }
            rows.append({
                "dataset": dataset, "model": model, "task": manifest["task"], "split": split,
                "strength3_design_count": design_count, "joint_risk": joint,
                "through_triple_fraction": through_triple / joint if joint else 0,
                "mean_member_loss": member_loss, "quotient_loss": quotient_loss,
                **comparators, "strength3_residual": risk3,
                "strength3_expected_loss": quotient_loss + risk3,
                "component_reconstruction_error": abs(risk3 - reconstructed),
                "ambiguity_error": abs(member_loss - quotient_loss - joint),
            })
    frame = pd.DataFrame(rows)
    validation = frame[frame.split == "validation"].set_index(["dataset", "model"])
    test = frame[frame.split == "test"].set_index(["dataset", "model"])
    common = validation.index.intersection(test.index)
    material_index = common[(
        validation.loc[common, "joint_risk"] / validation.loc[common, "mean_member_loss"] >= 0.005
    ).to_numpy()]
    material = test.loc[material_index].reset_index()
    comparators = [
        "iid64_residual", "four_strength2_residual",
        "sixteen_strength1_residual", "sixteen_seed_blocks_residual",
    ]
    beats_all = np.ones(len(material), dtype=bool)
    lower_or_numerically_tied = np.ones(len(material), dtype=bool)
    tolerance = np.maximum(1e-18, 1e-12 * material.joint_risk.to_numpy())
    for comparator in comparators:
        beats_all &= material.strength3_residual.to_numpy() < material[comparator].to_numpy()
        lower_or_numerically_tied &= (
            material.strength3_residual.to_numpy()
            <= material[comparator].to_numpy() + tolerance
        )
    group_rows = []
    if config.get("source_groups"):
        material["source_group"] = material.dataset.map(config["source_groups"])
        for group, current in material.groupby("source_group"):
            record = {"source_group": group, "material_cells": len(current),
                      "strength3_residual": float(current.strength3_residual.mean())}
            for comparator in comparators:
                record[comparator] = float(current[comparator].mean())
            record["beats_all"] = all(record["strength3_residual"] < record[c] for c in comparators)
            group_rows.append(record)
    groups = pd.DataFrame(group_rows)
    means = {name: float(material[name].mean()) for name in ["strength3_residual", *comparators]}
    summary = {
        "status": "complete", "cells": int(len(test)),
        "validation_material_cells": len(material),
        "test_cells_strength3_beats_all": int(beats_all.sum()),
        "test_cells_strength3_lower_or_numerically_tied_all": int(lower_or_numerically_tied.sum()),
        "exact_two_sided_cell_sign_p": exact_sign_p(int(beats_all.sum()), len(material)) if len(material) else None,
        "mean_residual_reductions": {
            comparator: float(1 - means["strength3_residual"] / means[comparator])
            for comparator in comparators
        },
        "mean_through_triple_fraction": float(material.through_triple_fraction.mean()) if len(material) else None,
        "source_groups": len(groups),
        "source_groups_strength3_beats_all": int(groups.beats_all.sum()) if len(groups) else None,
        "exact_two_sided_source_sign_p": exact_sign_p(int(groups.beats_all.sum()), len(groups)) if len(groups) else None,
        "maximum_component_reconstruction_error": float(frame.component_reconstruction_error.max()),
        "maximum_ambiguity_error": float(frame.ambiguity_error.max()),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / f"{args.output_prefix}_cells.csv", index=False)
    material.assign(
        beats_all=beats_all, lower_or_numerically_tied_all=lower_or_numerically_tied
    ).to_csv(
        args.output_dir / f"{args.output_prefix}_validation_screened_test_cells.csv", index=False
    )
    if len(groups):
        groups.to_csv(args.output_dir / f"{args.output_prefix}_source_groups.csv", index=False)
    (args.output_dir / f"{args.output_prefix}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
