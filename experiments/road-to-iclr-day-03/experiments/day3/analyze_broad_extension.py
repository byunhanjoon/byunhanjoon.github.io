"""Analyze the prospective extension and the combined 30-dataset AdamW screen."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .analyze_broad_phase1 import bootstrap, controlled_pairs, load_phase1
from .broad_data import config
from .broad_extension_data import extension_config


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/day3/broad_benchmark"


def load_extension() -> pd.DataFrame:
    paths = sorted(
        path
        for path in RESULTS.glob("extension_shard*.csv")
        if not path.stem.endswith("_curves")
    )
    if not paths:
        raise FileNotFoundError("No extension shards")
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    key = [
        "dataset",
        "representation",
        "target_kappa",
        "model",
        "remedy",
        "seed",
        "learning_rate_requested",
        "ridge_requested",
        "precondition_frequency_requested",
    ]
    return frame.drop_duplicates(key, keep="last")


def summarize(pairs: pd.DataFrame) -> dict[str, object]:
    grouped_dataset_model = pairs.groupby(["dataset", "model"]).sensitivity_normalized.mean()
    nonzero = grouped_dataset_model[grouped_dataset_model.abs() > 1e-12]
    ci = bootstrap(grouped_dataset_model.to_numpy())
    by_model = {}
    for model, part in pairs.groupby("model"):
        grouped = part.groupby("dataset").sensitivity_normalized.mean()
        model_ci = bootstrap(grouped.to_numpy())
        by_model[model] = {
            "datasets": int(part.dataset.nunique()),
            "pairs": len(part),
            "mean_sensitivity_normalized": float(part.sensitivity_normalized.mean()),
            "median_sensitivity_normalized": float(part.sensitivity_normalized.median()),
            "harmful_fraction": float((part.sensitivity_normalized < 0).mean()),
            "dataset_bootstrap_ci": model_ci,
        }
    return {
        "datasets": int(pairs.dataset.nunique()),
        "models": int(pairs.model.nunique()),
        "pairs": len(pairs),
        "mean_sensitivity_normalized": float(pairs.sensitivity_normalized.mean()),
        "median_sensitivity_normalized": float(pairs.sensitivity_normalized.median()),
        "harmful_fraction": float((pairs.sensitivity_normalized < 0).mean()),
        "dataset_model_bootstrap_ci": ci,
        "dataset_model_wilcoxon_p": (
            float(wilcoxon(nonzero).pvalue) if len(nonzero) >= 3 else None
        ),
        "by_model": by_model,
    }


def main() -> None:
    extension = load_extension()
    extension.to_csv(RESULTS / "extension_all.csv", index=False)
    extension_pairs = controlled_pairs(extension, "test_primary")
    extension_pairs.to_csv(RESULTS / "extension_pairs.csv", index=False)

    phase_pairs = controlled_pairs(load_phase1(), "test_primary")
    original_adam = phase_pairs[phase_pairs.remedy.eq("adamw")].copy()
    combined = pd.concat([original_adam, extension_pairs], ignore_index=True)
    combined.to_csv(RESULTS / "combined_30_adamw_pairs.csv", index=False)

    original_summary = summarize(original_adam)
    extension_summary = summarize(extension_pairs)
    combined_summary = summarize(combined)
    claim_cfg = config()["claim_gates"]
    combined_summary["basis_sensitivity_gate_passes"] = bool(
        combined_summary["median_sensitivity_normalized"] < 0
        and combined_summary["dataset_model_bootstrap_ci"][1] < 0
        and combined_summary["harmful_fraction"]
        >= float(claim_cfg["minimum_harmful_fraction"])
    )
    payload = {
        "design": {
            "original_datasets": len(config()["datasets"]),
            "prospective_extension_datasets": len(extension_config()["datasets"]),
            "combined_distinct_datasets": int(combined.dataset.nunique()),
            "kappas": extension_config()["kappas"],
            "seeds": extension_config()["seeds"],
            "models": extension_config()["models"],
        },
        "original_25": original_summary,
        "prospective_extension_5": extension_summary,
        "combined_30": combined_summary,
    }
    output = RESULTS / "combined_30_summary.json"
    output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
