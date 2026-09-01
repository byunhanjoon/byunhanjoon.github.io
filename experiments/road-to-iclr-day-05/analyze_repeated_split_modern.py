"""Aggregate the frozen three-split HistGB/CatBoost transport experiment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SPLITS = (2026082901, 2026082911, 2026082921)
ORIGINAL_SPLIT = 2026082831
TOLERANCE = 1e-15
COMPARISONS = {
    "pair32": ("pair32", "disjoint_pair_mean32", "independent_pair_mean32"),
    "pack64": ("pack64", "mutually_disjoint_pack64", "two_disjoint_pairs64"),
    "unbiased_pair_cross64": (
        "pair_cross64", "disjoint_pair_cross64", "independent_block_u64"
    ),
}


def transfer_rows(split_seed: int, prefix: str) -> list[dict[str, object]]:
    payload = json.loads((RESULTS / f"{prefix}_extension_summary.json").read_text())
    return [
        {"split_seed": split_seed, **row}
        for row in payload["exact_validation_to_test_transfer"]["sources"]
    ]


def main() -> None:
    manifest_issues, manifests = [], []
    for split_seed in SPLITS:
        directory = RESULTS / f"openml_modern_model_split_{split_seed}"
        for path in sorted(directory.glob("*.json")):
            row = json.loads(path.read_text())
            manifests.append(row)
            if row.get("status") != "complete" or row.get("fits") != 128:
                manifest_issues.append(f"incomplete:{path}")
            if row.get("effective_split_seed") != split_seed:
                manifest_issues.append(f"seed:{path}")

    strength_frames, per_split = [], {}
    for split_seed in SPLITS:
        frame = pd.read_csv(RESULTS / f"modern_split_{split_seed}_strength2_cells.csv")
        frame = frame[frame.split == "test"].copy()
        frame["split_seed"] = split_seed
        frame["beats_all"] = (
            (frame.strength2_residual < frame.iid16_residual)
            & (frame.strength2_residual < frame.four_strength1_residual)
            & (frame.strength2_residual < frame.four_seed_blocks_residual)
        )
        strength_frames.append(frame)
        material = frame[frame.material]
        source = frame.groupby("dataset")[[
            "strength2_residual", "iid16_residual", "four_strength1_residual",
            "four_seed_blocks_residual",
        ]].mean()
        source_win = (
            (source.strength2_residual < source.iid16_residual)
            & (source.strength2_residual < source.four_strength1_residual)
            & (source.strength2_residual < source.four_seed_blocks_residual)
        )
        per_split[str(split_seed)] = {
            "material_wins": int(material.beats_all.sum()),
            "material_cells": int(len(material)),
            "source_mean_wins": int(source_win.sum()),
            "sources": int(len(source)),
        }
    strength = pd.concat(strength_frames, ignore_index=True)
    material = strength[strength.material]
    source_split = strength.groupby(["split_seed", "dataset"])[[
        "strength2_residual", "iid16_residual", "four_strength1_residual",
        "four_seed_blocks_residual",
    ]].mean()
    source_split_wins = (
        (source_split.strength2_residual < source_split.iid16_residual)
        & (source_split.strength2_residual < source_split.four_strength1_residual)
        & (source_split.strength2_residual < source_split.four_seed_blocks_residual)
    )
    material_wins = int(material.beats_all.sum())
    strength_summary = {
        "material_wins": material_wins,
        "material_cells": int(len(material)),
        "material_win_fraction": float(material_wins / len(material)),
        "required_material_win_fraction": 0.8,
        "dataset_split_mean_wins": int(source_split_wins.sum()),
        "dataset_split_means": int(len(source_split_wins)),
        "required_dataset_split_mean_wins": 22,
        "frozen_transport_gate_passed": bool(
            material_wins / len(material) >= .8 and source_split_wins.sum() >= 22
        ),
        "by_split": per_split,
    }

    packing = {}
    for name, (family, action, control) in COMPARISONS.items():
        frames = []
        for split_seed in SPLITS:
            current = pd.read_csv(
                RESULTS / f"modern_split_{split_seed}_packing_calibration.csv"
            )
            current = current[
                (current.family == family) & current.method.isin((action, control))
            ].copy()
            current["split_seed"] = split_seed
            frames.append(current)
        frame = pd.concat(frames, ignore_index=True)
        pivot = frame.pivot(
            index=["split_seed", "dataset", "model"],
            columns="method", values="score_rmse",
        )
        difference = pivot[control] - pivot[action]
        nondegenerate = pivot[control] > 1e-12
        source = frame.groupby(
            ["split_seed", "dataset", "method"]
        ).score_rmse.mean().unstack()
        source_wins = int((source[action] < source[control] - TOLERANCE).sum())
        relative = 1 - pivot.loc[nondegenerate, action] / pivot.loc[nondegenerate, control]
        losses = int((difference[nondegenerate] < -TOLERANCE).sum())
        packing[name] = {
            "nondegenerate_candidate_wins": int(
                (difference[nondegenerate] > TOLERANCE).sum()
            ),
            "nondegenerate_candidate_losses": losses,
            "nondegenerate_candidates": int(nondegenerate.sum()),
            "mean_nondegenerate_relative_rmse_reduction": float(relative.mean()),
            "dataset_split_mean_wins": source_wins,
            "dataset_split_means": int(len(source)),
            "required_dataset_split_mean_wins": 22,
            "frozen_transport_gate_passed": bool(losses == 0 and source_wins >= 22),
        }

    transfer = []
    for split_seed in SPLITS:
        transfer.extend(transfer_rows(split_seed, f"modern_split_{split_seed}"))
    transfer_frame = pd.DataFrame(transfer)
    transfer_frame.to_csv(RESULTS / "repeated_split_modern_transfer.csv", index=False)
    original = transfer_rows(ORIGINAL_SPLIT, "modern_model")
    all_four = pd.DataFrame(original + transfer)
    transfer_summary = {
        "winner_agreements": int(transfer_frame.winner_agreement.sum()),
        "dataset_split_pairs": int(len(transfer_frame)),
        "mean_test_regret_of_validation_winner": float(
            transfer_frame.test_regret_of_validation_winner.mean()
        ),
        "failures_by_dataset": {
            str(name): int((~group.winner_agreement).sum())
            for name, group in transfer_frame.groupby("dataset")
            if (~group.winner_agreement).any()
        },
        "four_split_sensitivity": {
            "winner_agreements": int(all_four.winner_agreement.sum()),
            "dataset_split_pairs": int(len(all_four)),
            "mean_test_regret_of_validation_winner": float(
                all_four.test_regret_of_validation_winner.mean()
            ),
        },
        "formal_gate_imposed": False,
    }
    summary = {
        "status": "complete",
        "evidence_status": "frozen_splits_after_original_split_outcomes",
        "split_seeds": list(SPLITS),
        "complete_tensors": len(manifests),
        "represented_complete_product_fits": int(sum(row["fits"] for row in manifests)),
        "manifest_issues": manifest_issues,
        "strength2": strength_summary,
        "packing": packing,
        "all_frozen_nuisance_transport_gates_passed": bool(
            strength_summary["frozen_transport_gate_passed"]
            and all(row["frozen_transport_gate_passed"] for row in packing.values())
        ),
        "exact_validation_to_test_transfer": transfer_summary,
        "interpretation": (
            "nuisance-efficiency transports; exact selection remains partition-dependent"
        ),
    }
    (RESULTS / "repeated_split_modern_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
