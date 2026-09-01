"""Operator calibration for the independent four-pack cross-score."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_cross_quotient_selection as CQS
from analyze_disjoint_pack64 import sample_pack_and_pairs
from analyze_mixed_resolvable_packing import SHAPE


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OPERATOR_BLOCKS = 8
PACKS_PER_BLOCK = 8_192


def operator_covariances() -> list[np.ndarray]:
    covariances = []
    for block in range(OPERATOR_BLOCKS):
        second = np.zeros((128, 128), dtype=np.float64)
        first = np.zeros(128, dtype=np.float64)
        for chunk in range(PACKS_PER_BLOCK // 1_024):
            pack, _, _ = sample_pack_and_pairs(
                SHAPE, "pack-cross128-variance", f"{block}-{chunk}"
            )
            ids = pack.reshape(1_024, 64)
            weights = np.zeros((1_024, 128), dtype=np.float64)
            weights[np.arange(1_024)[:, None], ids] = 1 / 64
            second += weights.T @ weights
            first += weights.sum(axis=0)
        mean = first / PACKS_PER_BLOCK
        covariances.append(
            (second - PACKS_PER_BLOCK * np.outer(mean, mean)) / (PACKS_PER_BLOCK - 1)
        )
    return covariances


def predicted_variance(
    flat: np.ndarray, y: np.ndarray, covariance: np.ndarray
) -> tuple[float, float, float]:
    matrix = flat.reshape(128, -1).astype(np.float64)
    quotient = flat.mean(axis=0)
    residual = CQS.residuals(y, quotient[None])[0].reshape(-1)
    examples = flat.shape[-2]
    projected_residual = matrix @ residual
    linear = float(projected_residual @ covariance @ projected_residual / examples ** 2)
    gram = matrix @ matrix.T
    quadratic = float(np.trace(covariance @ gram @ covariance @ gram) / examples ** 2)
    return 2 * linear + quadratic, 2 * linear, quadratic


def main() -> None:
    covariances = operator_covariances()
    calibration = pd.read_csv(RESULTS / "disjoint_pack_cross128_calibration.csv")
    calibration = calibration[
        (calibration.method == "disjoint_pack_cross128")
        & (calibration.product_cells == 128)
    ]
    rows = []
    for record in calibration.itertuples(index=False):
        directory = dict((panel, directory) for panel, _, directory in CQS.PANELS)[record.panel]
        archive = np.load(RESULTS / directory / f"{record.dataset}__{record.model}.npz")
        flat = archive["validation_predictions"].reshape(
            (-1,) + archive["validation_predictions"].shape[-2:]
        ).astype(np.float64)
        y = archive["validation_y"]
        estimates = np.asarray([predicted_variance(flat, y, covariance) for covariance in covariances])
        predicted, linear, quadratic = estimates.mean(axis=0)
        operator_se = float(estimates[:, 0].std(ddof=1) / np.sqrt(OPERATOR_BLOCKS))
        observed = max(float(record.score_rmse ** 2 - record.score_bias ** 2), 0.0)
        observed_se = float(np.sqrt(2 / (512 - 1)) * observed)
        combined = float(np.hypot(operator_se, observed_se))
        z = abs(predicted - observed) / combined if combined else 0.0
        rows.append({
            "panel": record.panel, "dataset": record.dataset, "model": record.model,
            "predicted_variance": predicted, "observed_variance": observed,
            "predicted_to_observed_ratio": predicted / observed if observed else 1.0,
            "operator_standard_error": operator_se,
            "observed_variance_reference_se": observed_se,
            "combined_standardized_difference": z,
            "linear_variance_term": linear, "quadratic_variance_term": quadratic,
            "quadratic_fraction": quadratic / predicted if predicted else 0.0,
            "within_2_58_combined_se": bool(z <= 2.58),
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "pack_cross128_variance_calibration.csv", index=False)
    panel_rows = []
    for panel, current in frame.groupby("panel"):
        ratio = float(np.exp(np.mean(np.log(current.predicted_to_observed_ratio))))
        panel_rows.append({
            "panel": panel, "candidates": int(len(current)),
            "geometric_predicted_to_observed_ratio": ratio,
            "within_frozen_ratio_range": bool(.85 <= ratio <= 1.15),
            "mean_quadratic_fraction": float(current.quadratic_fraction.mean()),
        })
    within = int(frame.within_2_58_combined_se.sum())
    summary = {
        "status": "complete", "operator_packs": OPERATOR_BLOCKS * PACKS_PER_BLOCK,
        "operator_blocks": OPERATOR_BLOCKS, "candidates": int(len(frame)),
        "candidates_within_2_58_combined_se": within,
        "max_combined_standardized_difference": float(frame.combined_standardized_difference.max()),
        "predicted_to_observed_ratio_range": [
            float(frame.predicted_to_observed_ratio.min()),
            float(frame.predicted_to_observed_ratio.max()),
        ],
        "panels": panel_rows,
        "frozen_gate_passed": bool(
            within >= 20 and all(row["within_frozen_ratio_range"] for row in panel_rows)
        ),
    }
    (RESULTS / "pack_cross128_variance_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
