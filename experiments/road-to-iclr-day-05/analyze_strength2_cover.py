"""Exact budget-16 strength-2 nuisance-cover analysis."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "road-to-iclr-idea-search"))
from orbit_anova import decompose  # noqa: E402
from analyze_joint_orthogonal_cover import orthogonal_designs  # noqa: E402


PERMS4 = tuple(itertools.permutations(range(4)))
MUL4 = np.asarray([
    [0, 0, 0, 0],
    [0, 1, 2, 3],
    [0, 2, 3, 1],
    [0, 3, 1, 2],
], dtype=int)


def gf4_trace(value: int) -> int:
    result = int(value) ^ int(MUL4[value, value])
    if result not in (0, 1):
        raise AssertionError("GF(4) trace did not land in GF(2)")
    return result


def strength2_base(category_levels: int, class_levels: int, seed_levels: int = 4) -> np.ndarray:
    rows = []
    for u in range(4):
        for v in range(4):
            feature = u
            if category_levels == 4:
                category = v
            elif category_levels == 2:
                category = gf4_trace(v)
            elif category_levels == 1:
                category = 0
            else:
                raise ValueError("category factor must have four, two, or one levels")
            if seed_levels == 4:
                seed = u ^ v
            elif seed_levels == 2:
                seed = gf4_trace(u ^ int(MUL4[3, v]))
            elif seed_levels == 1:
                seed = 0
            else:
                raise ValueError("seed factor must have four, two, or one levels")
            if class_levels == 4:
                label = u ^ int(MUL4[2, v])
            elif class_levels == 2:
                label = gf4_trace(u ^ int(MUL4[2, v]))
            elif class_levels == 1:
                label = 0
            else:
                raise ValueError("class factor must have four, two, or one levels")
            rows.append((feature, category, label, seed))
    return np.asarray(rows, dtype=int)


def assert_strength(base: np.ndarray, cardinalities: tuple[int, ...], strength: int) -> None:
    for order in range(1, strength + 1):
        for factors in itertools.combinations(range(len(cardinalities)), order):
            if any(cardinalities[factor] == 1 for factor in factors):
                continue
            counts = np.zeros(tuple(cardinalities[factor] for factor in factors), dtype=int)
            for row in base:
                counts[tuple(row[factor] for factor in factors)] += 1
            if np.unique(counts).size != 1:
                raise AssertionError(f"unbalanced margin {factors}")


def incidence_covariance(
    designs: np.ndarray,
    cardinalities: tuple[int, int, int, int],
) -> np.ndarray:
    cell_count = math.prod(cardinalities)
    incidence = np.zeros((len(designs), cell_count), dtype=np.float32)
    for index, design in enumerate(designs):
        ids = np.ravel_multi_index(design.T, cardinalities)
        incidence[index, ids] += 1.0 / len(design)
    mean = np.full(cell_count, 1.0 / cell_count, dtype=np.float64)
    return incidence.T.astype(np.float64) @ incidence.astype(np.float64) / len(designs) - np.outer(mean, mean)


def strength1_family(category_levels: int, class_levels: int, seed_levels: int = 4) -> np.ndarray:
    raw = orthogonal_designs(class_levels, category_levels)
    if seed_levels == 4:
        seed_assignments = [np.arange(4)]
    elif seed_levels == 2:
        seed_assignments = []
        for positive in itertools.combinations(range(4), 2):
            assignment = np.zeros(4, dtype=int)
            assignment[list(positive)] = 1
            seed_assignments.append(assignment)
    elif seed_levels == 1:
        seed_assignments = [np.zeros(4, dtype=int)]
    else:
        raise ValueError("seed factor must have four, two, or one levels")
    return np.concatenate([
        np.concatenate((raw, np.broadcast_to(seed, raw.shape[:2])[..., None]), axis=-1)
        for seed in seed_assignments
    ], axis=0)


def strength2_family(category_levels: int, class_levels: int, seed_levels: int = 4) -> np.ndarray:
    base = strength2_base(category_levels, class_levels, seed_levels)
    cardinalities = (4, category_levels, class_levels, seed_levels)
    assert_strength(base, cardinalities, 2)
    category_perms = (
        PERMS4 if category_levels == 4
        else ((0, 1), (1, 0)) if category_levels == 2
        else ((0,),)
    )
    class_perms = PERMS4 if class_levels == 4 else ((0, 1), (1, 0)) if class_levels == 2 else ((0,),)
    seed_perms = (
        PERMS4 if seed_levels == 4
        else ((0, 1), (1, 0)) if seed_levels == 2
        else ((0,),)
    )
    designs = []
    for feature_perm in PERMS4:
        for category_perm in category_perms:
            for class_perm in class_perms:
                for seed_perm in seed_perms:
                    current = base.copy()
                    current[:, 0] = np.asarray(feature_perm)[current[:, 0]]
                    current[:, 1] = np.asarray(category_perm)[current[:, 1]]
                    current[:, 2] = np.asarray(class_perm)[current[:, 2]]
                    current[:, 3] = np.asarray(seed_perm)[current[:, 3]]
                    designs.append(current)
    return np.asarray(designs, dtype=np.int8)


def expected_residual(predictions: np.ndarray, covariance: np.ndarray) -> float:
    flat = predictions.reshape((-1,) + predictions.shape[-2:]).astype(np.float64)
    centered = flat - flat.mean(axis=0, keepdims=True)
    gram = np.einsum("irk,jrk->ij", centered, centered, optimize=True) / predictions.shape[-2]
    return float(np.sum(covariance * gram))


def component_coefficients(
    covariance: np.ndarray,
    cardinalities: tuple[int, ...],
    names: tuple[str, ...] = ("feature", "category", "class", "seed"),
) -> dict[str, float]:
    """Eigenvalue multiplier on each product-ANOVA subspace.

    Exhaustive independent level permutations make the design covariance
    commute with every factor's symmetric group. It is therefore scalar on
    each tensor-product contrast subspace. The trace/rank quotient recovers
    that scalar, with a cell-count correction because ANOVA risk uses the
    uniform cell measure.
    """
    cell_count = math.prod(cardinalities)
    output = {}
    for order in range(1, len(cardinalities) + 1):
        for subset in itertools.combinations(range(len(cardinalities)), order):
            rank = math.prod(cardinalities[index] - 1 for index in subset)
            if rank == 0:
                output[":".join(names[index] for index in subset)] = 0.0
                continue
            projection = np.asarray([[1.0]])
            for index, cardinality in enumerate(cardinalities):
                mean = np.ones((cardinality, cardinality)) / cardinality
                factor = np.eye(cardinality) - mean if index in subset else mean
                projection = np.kron(projection, factor)
            coefficient = cell_count * float(np.trace(covariance @ projection)) / rank
            output[":".join(names[index] for index in subset)] = coefficient
    return output


def proper_loss(y: np.ndarray, prediction: np.ndarray) -> float:
    if prediction.shape[-1] == 1:
        return float(np.mean((prediction[:, 0] - y) ** 2))
    targets = np.eye(prediction.shape[-1])[y.astype(int)]
    return float(np.mean(np.sum((prediction - targets) ** 2, axis=-1)))


def exact_sign_p(wins: int, total: int) -> float:
    tail = min(wins, total - wins)
    return float(min(1.0, 2 * sum(math.comb(total, k) for k in range(tail + 1)) / 2**total))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    covariance_cache = {}
    rows = []
    for dataset, model in itertools.product(config["datasets"], config["models"]):
        stem = f"{dataset}__{model}"
        path = args.input_dir / f"{stem}.npz"
        manifest_path = args.input_dir / f"{stem}.json"
        if not path.exists() or not manifest_path.exists():
            raise RuntimeError(f"missing {stem}")
        archive = np.load(path)
        manifest = json.loads(manifest_path.read_text())
        for split in ("validation", "test"):
            predictions = archive[f"{split}_predictions"].astype(np.float64)
            y = archive[f"{split}_y"]
            cardinalities = predictions.shape[:4]
            key = (predictions.shape[1], predictions.shape[2], predictions.shape[3])
            if key not in covariance_cache:
                family1 = strength1_family(*key)
                family2 = strength2_family(*key)
                covariance1 = incidence_covariance(family1, cardinalities)
                covariance2 = incidence_covariance(family2, cardinalities)
                covariance_cache[key] = (
                    covariance1, covariance2, len(family1), len(family2),
                    component_coefficients(covariance1, cardinalities),
                    component_coefficients(covariance2, cardinalities),
                )
            covariance1, covariance2, count1, count2, coefficients1, coefficients2 = covariance_cache[key]
            risk1 = expected_residual(predictions, covariance1)
            risk2 = expected_residual(predictions, covariance2)
            flat = predictions.reshape((-1,) + predictions.shape[-2:])
            quotient = flat.mean(axis=0)
            joint = float(np.mean(np.sum((flat - quotient) ** 2, axis=-1)))
            seed_average = predictions.mean(axis=3)
            persistent = float(np.mean(np.sum((seed_average - quotient) ** 2, axis=-1)))
            member_loss = float(np.mean([proper_loss(y, member) for member in flat]))
            quotient_loss = proper_loss(y, quotient)
            anova = decompose(predictions, ("feature", "category", "class", "seed"))
            pairwise = sum(float(value) for name, value in anova.items() if name.count(":") == 1)
            main = sum(float(anova[name]) for name in ("feature", "category", "class", "seed"))
            reconstructed1 = sum(coefficients1[name] * float(anova[name]) for name in coefficients1)
            reconstructed2 = sum(coefficients2[name] * float(anova[name]) for name in coefficients2)
            comparators = (joint / 16, risk1 / 4, persistent / 4)
            rows.append({
                "dataset": dataset, "model": model, "task": manifest["task"], "split": split,
                "strength1_design_count": count1, "strength2_design_count": count2,
                "joint_risk": joint, "main_fraction": main / joint if joint else 0.0,
                "main_plus_pair_fraction": (main + pairwise) / joint if joint else 0.0,
                "mean_member_loss": member_loss, "quotient_loss": quotient_loss,
                "iid16_residual": comparators[0],
                "four_strength1_residual": comparators[1],
                "four_seed_blocks_residual": comparators[2],
                "strength2_residual": risk2,
                "strength1_component_reconstruction_error": abs(risk1 - reconstructed1),
                "strength2_component_reconstruction_error": abs(risk2 - reconstructed2),
                "strength2_expected_loss": quotient_loss + risk2,
                "strength2_vs_iid16_reduction": 1 - risk2 / comparators[0] if comparators[0] else 0.0,
                "strength2_vs_four_strength1_reduction": 1 - risk2 / comparators[1] if comparators[1] else 0.0,
                "strength2_vs_seed_blocks_reduction": 1 - risk2 / comparators[2] if comparators[2] else 0.0,
                "material": bool(joint / member_loss >= 0.005) if member_loss else False,
                "ambiguity_error": abs(member_loss - quotient_loss - joint),
            })
    frame = pd.DataFrame(rows)
    material = frame[(frame.split == "test") & frame.material]
    beats_all = (
        (material.strength2_residual < material.iid16_residual)
        & (material.strength2_residual < material.four_strength1_residual)
        & (material.strength2_residual < material.four_seed_blocks_residual)
    )
    means = {name: float(material[name].mean()) for name in (
        "strength2_residual", "iid16_residual", "four_strength1_residual", "four_seed_blocks_residual"
    )}
    wins = int(beats_all.sum())
    source_group_summary = None
    if config.get("source_groups"):
        group_rows = []
        for group_name in sorted(set(config["source_groups"].values())):
            datasets = [dataset for dataset, group in config["source_groups"].items() if group == group_name]
            current = material[material.dataset.isin(datasets)]
            if not len(current):
                group_rows.append({"source_group": group_name, "material_cells": 0, "beats_all": False})
                continue
            group_means = {
                name: float(current[name].mean()) for name in (
                    "strength2_residual", "iid16_residual", "four_strength1_residual", "four_seed_blocks_residual"
                )
            }
            group_rows.append({
                "source_group": group_name, "material_cells": len(current), **group_means,
                "beats_all": bool(
                    group_means["strength2_residual"] < group_means["iid16_residual"]
                    and group_means["strength2_residual"] < group_means["four_strength1_residual"]
                    and group_means["strength2_residual"] < group_means["four_seed_blocks_residual"]
                ),
            })
        groups = pd.DataFrame(group_rows)
        group_wins = int(groups.beats_all.sum())
        source_group_summary = {
            "source_groups": len(groups),
            "source_groups_with_material_cells": int((groups.material_cells > 0).sum()),
            "source_groups_strength2_beats_all": group_wins,
            "exact_two_sided_source_group_sign_p": exact_sign_p(group_wins, len(groups)),
            "source_group_gate_passed": bool(group_wins >= 4),
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        groups.to_csv(args.output_dir / f"{args.output_prefix}_source_groups.csv", index=False)
    summary = {
        "status": "complete", "cells": int((frame.split == "test").sum()),
        "material_cells": len(material), "material_cells_strength2_beats_all": wins,
        "exact_two_sided_sign_p": exact_sign_p(wins, len(material)) if len(material) else np.nan,
        "mean_strength2_vs_iid16_reduction": 1 - means["strength2_residual"] / means["iid16_residual"],
        "mean_strength2_vs_four_strength1_reduction": 1 - means["strength2_residual"] / means["four_strength1_residual"],
        "mean_strength2_vs_seed_blocks_reduction": 1 - means["strength2_residual"] / means["four_seed_blocks_residual"],
        "adaptive_gate_passed": bool(
            wins > len(material) / 2
            and means["strength2_residual"] < means["iid16_residual"]
            and means["strength2_residual"] < means["four_strength1_residual"]
            and means["strength2_residual"] < means["four_seed_blocks_residual"]
        ),
        "mean_main_plus_pair_fraction": float(material.main_plus_pair_fraction.mean()) if len(material) else np.nan,
        "maximum_ambiguity_error": float(frame.ambiguity_error.max()),
        "maximum_component_risk_reconstruction_error": float(max(
            frame.strength1_component_reconstruction_error.max(),
            frame.strength2_component_reconstruction_error.max(),
        )),
        "source_group_confirmation": source_group_summary,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / f"{args.output_prefix}_cells.csv", index=False)
    (args.output_dir / f"{args.output_prefix}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
