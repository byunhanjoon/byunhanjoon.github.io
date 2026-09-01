#!/usr/bin/env python3
"""Conditional target-free descriptor-gate ablation with leave-one-dataset-out CV."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from safe_basis.common import (  # noqa: E402
    bd,
    development_specs,
    disagreement,
    load_blocks,
    load_frozen_development_predictions,
    load_protocol,
    mix_predictions,
    normalized_excess_risk,
    orthogonal_orbit,
    write_json,
)


FEATURES = [
    "log_n_rows",
    "log_n_raw_features",
    "median_empirical_rank",
    "mean_gram_spectrum_entropy",
    "median_block_condition_log10",
    "class_imbalance",
    "categorical_fraction",
    "raw_gram_validation_disagreement",
]
ALPHAS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])


def training_descriptors(blocks: Any, raw_validation: np.ndarray, gram_validation: np.ndarray) -> dict[str, float]:
    ranks = []
    entropies = []
    conditions = []
    for indices in blocks.feature_blocks.values():
        values = blocks.X_train[:, indices]
        singular = np.linalg.svd(values, compute_uv=False)
        ranks.append(int(np.sum(singular / max(singular[0], 1e-12) > 1e-4)))
        energy = singular**2
        probability = energy / max(energy.sum(), 1e-12)
        positive = probability[probability > 0]
        entropies.append(float(-(positive * np.log(positive)).sum() / max(np.log(len(probability)), 1e-12)))
        positive_singular = singular[singular > max(singular[0] * 1e-12, 1e-12)]
        conditions.append(float(positive_singular[0] / positive_singular[-1]))
    if blocks.dataset.problem_type == "classification":
        counts = np.bincount(blocks.dataset.y_train.astype(int))
        imbalance = float(counts.max() / counts.sum())
    else:
        imbalance = 0.0
    return {
        "log_n_rows": float(np.log1p(len(blocks.dataset.X_train_raw))),
        "log_n_raw_features": float(np.log1p(blocks.dataset.X_train_raw.shape[1])),
        "median_empirical_rank": float(np.median(ranks)),
        "mean_gram_spectrum_entropy": float(np.mean(entropies)),
        "median_block_condition_log10": float(np.median(np.log10(np.maximum(conditions, 1.0)))),
        "class_imbalance": imbalance,
        "categorical_fraction": float(len(blocks.dataset.nominal_columns) / max(blocks.dataset.X_train_raw.shape[1], 1)),
        "raw_gram_validation_disagreement": disagreement(
            blocks.dataset.problem_type,
            blocks.dataset.y_validation,
            raw_validation,
            gram_validation,
        ),
    }


def quantize(value: float) -> float:
    return float(ALPHAS[np.argmin(np.abs(ALPHAS - np.clip(value, 0.0, 1.0)))])


def main() -> None:
    protocol = load_protocol()
    gate_summary = pd.read_csv(ROOT / "results" / "processed" / "development_gate_summary.csv")
    safe = gate_summary[gate_summary["method"] == "SafeGram-t01"].iloc[0]
    if not (safe["raw_fallback_rate"] >= 0.4 or safe["median_disagreement_reduction"] < 0.4):
        write_json(ROOT / "results" / "processed" / "descriptor_gate_manifest.json", {"status": "NOT_TRIGGERED", "reason": "SafeGram was not too conservative on development."})
        return
    evidence = pd.read_csv(ROOT / "results" / "processed" / "development_gate_alpha_evidence.csv")
    specs = {spec["key"]: spec for spec in development_specs(protocol)}
    records: list[dict[str, Any]] = []
    cache: dict[tuple[str, str, int], tuple[Any, Any, Any, Any]] = {}
    for dataset in protocol["development_datasets"]:
        blocks = load_blocks(specs[dataset], protocol)
        orbit = orthogonal_orbit(blocks, protocol)
        reference_id = orbit[0].representation_id
        for model in protocol["development_models"]:
            for seed in protocol["model_seeds"]:
                raw, gram, _ = load_frozen_development_predictions(model, dataset, int(seed))
                subset = evidence[(evidence["dataset"] == dataset) & (evidence["model"] == model) & (evidence["seed"] == seed)]
                passing = subset[subset["normalized_excess_risk"] <= 0.01]
                target_alpha = float(passing["alpha"].max()) if len(passing) else 0.0
                descriptor = training_descriptors(blocks, raw[reference_id]["validation"], gram["validation"])
                records.append({"dataset": dataset, "problem_type": blocks.dataset.problem_type, "model": model, "seed": int(seed), "target_safe_alpha": target_alpha, **descriptor})
                cache[(dataset, model, int(seed))] = (blocks, orbit, raw, gram)
    frame = pd.DataFrame(records)
    predictions = []
    for held_out in protocol["development_datasets"]:
        train = frame[frame["dataset"] != held_out]
        test = frame[frame["dataset"] == held_out]
        scaler = StandardScaler().fit(train[FEATURES])
        model = Ridge(alpha=1.0).fit(scaler.transform(train[FEATURES]), train["target_safe_alpha"])
        for row, value in zip(test.itertuples(index=False), model.predict(scaler.transform(test[FEATURES]))):
            predictions.append({"dataset": row.dataset, "model": row.model, "seed": row.seed, "target_safe_alpha": row.target_safe_alpha, "predicted_alpha_continuous": float(value), "descriptor_alpha": quantize(float(value))})
    prediction_frame = pd.DataFrame(predictions)
    evaluation = []
    for row in prediction_frame.itertuples(index=False):
        blocks, orbit, raw, gram = cache[(row.dataset, row.model, int(row.seed))]
        reference_id = orbit[0].representation_id
        method_prediction = mix_predictions(raw[reference_id]["test"], gram["test"], row.descriptor_alpha)
        safety = normalized_excess_risk(blocks.dataset.problem_type, blocks.dataset.y_test, raw[reference_id]["test"], method_prediction, blocks.dataset.y_train)
        evaluation.append({"dataset": row.dataset, "problem_type": blocks.dataset.problem_type, "model": row.model, "seed": row.seed, "method": "DescriptorGate-Ridge-LODO", "alpha": row.descriptor_alpha, "disagreement_reduction": row.descriptor_alpha, **safety})
    evaluation_frame = pd.DataFrame(evaluation)
    units = evaluation_frame.groupby(["dataset", "problem_type", "model", "method"], as_index=False).median(numeric_only=True)
    costs = units["normalized_excess_risk"].to_numpy(float)
    descriptor_summary = {
        "median_disagreement_reduction": float(units["disagreement_reduction"].median()),
        "median_C": float(np.median(costs)),
        "p95_C": float(np.quantile(costs, 0.95)),
        "max_C": float(np.max(costs)),
        "raw_fallback_rate": float((evaluation_frame["alpha"] == 0).mean()),
    }
    safe_summary = {key: float(safe[key]) for key in ("median_disagreement_reduction", "median_C", "p95_C", "max_C", "raw_fallback_rate")}
    clearly_outperforms = bool(
        descriptor_summary["median_disagreement_reduction"] >= safe_summary["median_disagreement_reduction"] + 0.10
        and descriptor_summary["p95_C"] <= 0.05
        and descriptor_summary["max_C"] <= 0.20
    )
    scaler = StandardScaler().fit(frame[FEATURES])
    final_model = Ridge(alpha=1.0).fit(scaler.transform(frame[FEATURES]), frame["target_safe_alpha"])
    config = {
        "status": "KEEP_FOR_PROSPECTIVE" if clearly_outperforms else "DISCARDED_AFTER_DEVELOPMENT_CV",
        "reason": "LODO descriptor gate clearly improves control while retaining tail safety." if clearly_outperforms else "LODO descriptor gate did not improve control by >=10 points while satisfying p95<=0.05 and max<=0.20.",
        "model": "Ridge(alpha=1.0)",
        "features": FEATURES,
        "feature_mean": scaler.mean_.tolist(),
        "feature_scale": scaler.scale_.tolist(),
        "coefficients": final_model.coef_.tolist(),
        "intercept": float(final_model.intercept_),
        "quantization": ALPHAS.tolist(),
        "target": "largest alpha with development validation point-estimate C<=0.01",
        "validation": "leave-one-dataset-out",
        "descriptor_summary": descriptor_summary,
        "SafeGram_t01_summary": safe_summary,
        "clearly_outperforms": clearly_outperforms,
        "prospective_outcomes_accessed": False,
    }
    processed = ROOT / "results" / "processed"
    frame.to_csv(processed / "descriptor_gate_descriptors.csv", index=False)
    prediction_frame.to_csv(processed / "descriptor_gate_lodo_predictions.csv", index=False)
    evaluation_frame.to_csv(processed / "descriptor_gate_cells.csv", index=False)
    units.to_csv(processed / "descriptor_gate_units.csv", index=False)
    write_json(processed / "descriptor_gate_config.json", config)
    write_json(processed / "descriptor_gate_manifest.json", {"status": "COMPLETE", **config})
    print(config)


if __name__ == "__main__":
    main()
