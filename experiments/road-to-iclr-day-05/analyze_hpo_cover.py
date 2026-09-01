"""Apply orthogonal nuisance covers after ordinary per-schema HPO."""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_hpo_quotient import gather_per_schema, validation_losses
from analyze_strength2_cover import (
    expected_residual,
    exact_sign_p,
    incidence_covariance,
    proper_loss,
    strength1_family,
    strength2_family,
)


HERE = Path(__file__).resolve().parent


def main() -> None:
    config = json.loads((HERE / "hpo_quotient_config.json").read_text())
    input_dir = HERE / "results" / "hpo_quotient"
    cache = {}
    rows = []
    for dataset, family in itertools.product(config["datasets"], config["families"]):
        archive = np.load(input_dir / f"{dataset}__{family}.npz")
        validation = archive["validation_predictions"].astype(np.float64)
        test = archive["test_predictions"].astype(np.float64)
        validation_y = archive["validation_y"]
        test_y = archive["test_y"]
        selected = np.empty(test.shape[1:], dtype=np.float64)
        entropies = []
        for seed in range(test.shape[4]):
            losses = validation_losses(validation[..., seed, :, :], validation_y)
            choices = np.argmin(losses, axis=0)
            selected[..., seed, :, :] = gather_per_schema(test[..., seed, :, :], choices)
            counts = np.bincount(choices.flat, minlength=test.shape[0])
            probabilities = counts[counts > 0] / choices.size
            entropies.append(float(-np.sum(probabilities * np.log2(probabilities))))
        cardinalities = selected.shape[:4]
        key = (selected.shape[1], selected.shape[2], selected.shape[3])
        if key not in cache:
            family1 = strength1_family(*key)
            family2 = strength2_family(*key)
            cache[key] = (
                incidence_covariance(family1, cardinalities),
                incidence_covariance(family2, cardinalities),
            )
        covariance1, covariance2 = cache[key]
        risk1 = expected_residual(selected, covariance1)
        risk2 = expected_residual(selected, covariance2)
        flat = selected.reshape((-1,) + selected.shape[-2:])
        quotient = flat.mean(axis=0)
        joint = float(np.mean(np.sum((flat - quotient) ** 2, axis=-1)))
        seed_average = selected.mean(axis=3)
        persistent = float(np.mean(np.sum((seed_average - quotient) ** 2, axis=-1)))
        member_loss = float(np.mean([proper_loss(test_y, member) for member in flat]))
        quotient_loss = proper_loss(test_y, quotient)
        rows.append({
            "dataset": dataset, "family": family,
            "mean_selection_entropy_bits": float(np.mean(entropies)),
            "joint_risk": joint, "joint_risk_fraction_of_member_loss": joint / member_loss,
            "strength1_residual": risk1, "iid4_residual": joint / 4,
            "seed_only_residual": persistent / (4 / selected.shape[3]),
            "strength2_residual": risk2, "iid16_residual": joint / 16,
            "four_strength1_residual": risk1 / 4,
            "four_seed_blocks_residual": persistent / (16 / selected.shape[3]),
            "quotient_loss": quotient_loss, "mean_member_loss": member_loss,
            "strength1_expected_loss": quotient_loss + risk1,
            "strength2_expected_loss": quotient_loss + risk2,
            "ambiguity_error": abs(member_loss - quotient_loss - joint),
        })
    frame = pd.DataFrame(rows)
    material = frame[frame.joint_risk_fraction_of_member_loss >= 0.005]
    win1 = (
        (material.strength1_residual < material.iid4_residual)
        & (material.strength1_residual < material.seed_only_residual)
    )
    win2 = (
        (material.strength2_residual < material.iid16_residual)
        & (material.strength2_residual < material.four_strength1_residual)
        & (material.strength2_residual < material.four_seed_blocks_residual)
    )
    summary = {
        "status": "complete", "cells": len(frame), "material_cells": len(material),
        "strength1_cells_beating_both": int(win1.sum()),
        "strength1_exact_sign_p": exact_sign_p(int(win1.sum()), len(material)) if len(material) else np.nan,
        "strength2_cells_beating_all": int(win2.sum()),
        "strength2_exact_sign_p": exact_sign_p(int(win2.sum()), len(material)) if len(material) else np.nan,
        "mean_strength1_vs_iid_reduction": 1 - material.strength1_residual.mean() / material.iid4_residual.mean(),
        "mean_strength1_vs_seed_reduction": 1 - material.strength1_residual.mean() / material.seed_only_residual.mean(),
        "mean_strength2_vs_iid_reduction": 1 - material.strength2_residual.mean() / material.iid16_residual.mean(),
        "mean_strength2_vs_four_strength1_reduction": 1 - material.strength2_residual.mean() / material.four_strength1_residual.mean(),
        "mean_strength2_vs_seed_blocks_reduction": 1 - material.strength2_residual.mean() / material.four_seed_blocks_residual.mean(),
        "maximum_ambiguity_error": float(frame.ambiguity_error.max()),
    }
    output = HERE / "results"
    frame.to_csv(output / "hpo_cover_cells.csv", index=False)
    (output / "hpo_cover_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
