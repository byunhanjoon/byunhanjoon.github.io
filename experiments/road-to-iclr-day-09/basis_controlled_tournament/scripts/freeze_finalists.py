#!/usr/bin/env python3
"""Select and immutably freeze at most three methods from development data."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys
import time

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_stage2_representations import METHODS as INTERFACES  # noqa: E402
from tournament.common import (  # noqa: E402
    load_json,
    load_protocol,
    protocol_hashes,
    sha256_file,
    write_json,
)


OUTPUT = ROOT / "configs" / "FINALIST_CONFIGS.json"
OUTPUT_SHA = ROOT / "configs" / "FINALIST_CONFIGS.sha256"
ALL_MODELS = ["controlled_mlp", "tabm_d", "tabicl_v2", "tabpfn_2_6", "catboost"]
TRAINABLE_MODELS = ["controlled_mlp", "tabm_d"]


OPTIMIZER_DEFINITIONS = {
    "BlockAdam+DataInit": ("block_adam", "data_equivariant", {}),
    "MatrixAdam+DataInit": ("matrix_adam", "data_equivariant", {}),
    "SoftBlockAdam-a0.1+DataInit": ("soft_block_adam", "data_equivariant", {"alpha": 0.1}),
}


def evidence(row: pd.Series) -> dict:
    keys = [
        "median_disagreement_reduction",
        "median_relative_task_change",
        "median_worst_orbit_gain",
        "failure_fraction",
        "paper_method_score",
        "wins",
        "ties",
        "losses",
        "units",
        "model_families",
    ]
    return {key: row[key].item() if hasattr(row[key], "item") else row[key] for key in keys}


def choose_representation(summary: pd.DataFrame) -> pd.Series:
    pool = summary[
        (summary["track"] == "representation")
        & (summary["method"] != "Raw")
        & (summary["median_relative_task_change"] <= 0.01)
    ]
    if pool.empty:
        pool = summary[(summary["track"] == "representation") & (summary["method"] != "Raw")]
    return pool.sort_values(
        ["paper_method_score", "median_worst_orbit_gain", "median_relative_task_change"],
        ascending=[False, False, True],
    ).iloc[0]


def choose_hybrid(units: pd.DataFrame, representation_method: str) -> pd.Series:
    validation = units[
        (units["split"] == "validation")
        & (units["track"] == "hybrid_prediction_mixture")
        & units["method"].str.startswith(f"Raw+{representation_method}@")
        & ~units["method"].str.endswith("@1")
    ].copy()
    records = []
    for method, frame in validation.groupby("method"):
        reduction = float(frame["disagreement_reduction"].median())
        cost = float(frame["relative_task_change"].median())
        score = reduction - 5 * max(cost, 0.0) - 0.25 * float(frame["failure"].mean())
        records.append(
            {
                "method": method,
                "median_disagreement_reduction": reduction,
                "median_relative_task_change": cost,
                "median_worst_orbit_gain": float(frame["worst_orbit_gain"].median()),
                "paper_method_score": score,
            }
        )
    candidates = pd.DataFrame(records)
    eligible = candidates[candidates["median_relative_task_change"] <= 0.01]
    if len(eligible):
        candidates = eligible
    return candidates.sort_values(
        ["paper_method_score", "median_worst_orbit_gain", "median_relative_task_change"],
        ascending=[False, False, True],
    ).iloc[0]


def choose_optimizer(summary: pd.DataFrame) -> pd.Series:
    pool = summary[
        summary["method"].isin([f"{name}[equal-HPO]" for name in OPTIMIZER_DEFINITIONS])
    ]
    eligible = pool[pool["median_relative_task_change"] <= 0.03]
    if len(eligible):
        pool = eligible
    return pool.sort_values(
        ["paper_method_score", "median_disagreement_reduction", "median_relative_task_change"],
        ascending=[False, False, True],
    ).iloc[0]


def main() -> None:
    if OUTPUT.exists() or OUTPUT_SHA.exists():
        raise RuntimeError("refusing to overwrite already frozen FINALIST_CONFIGS")
    processed = ROOT / "results" / "processed"
    required = [
        processed / "development_all_method_summary.csv",
        processed / "development_all_units.csv",
        processed / "equal_hpo_lr_selection.csv",
        processed / "mechanism_equivariance_verdict.csv",
        processed / "anchor_ablation_summary.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"development evidence incomplete: {missing}")
    if len(list((processed / "stage2_optimizer_cells").glob("*.csv"))) != 36:
        raise RuntimeError("optimizer development panel incomplete")
    representation_cells = [
        path for path in (processed / "stage2_representation_cells").glob("*.csv")
        if "coordinate_audit" not in path.name
    ]
    if len(representation_cells) != 90:
        raise RuntimeError("representation development panel incomplete")
    for path in representation_cells:
        if "GramAnchor-m8" not in set(pd.read_csv(path, usecols=["method"])["method"]):
            raise RuntimeError(f"m=8 full-panel rescue missing from {path}")
    if len(list((processed / "equal_hpo").glob("*__trials.csv"))) != 36:
        raise RuntimeError("equal-HPO development panel incomplete")
    if len(list((processed / "mechanism").glob("*.csv"))) != 6:
        raise RuntimeError("matched-function mechanism panel incomplete")

    summary = pd.read_csv(processed / "development_all_method_summary.csv")
    units = pd.read_csv(processed / "development_all_units.csv")
    mechanism = pd.read_csv(processed / "mechanism_equivariance_verdict.csv")
    for method in ("BlockAdam", "MatrixAdam"):
        audited = mechanism[mechanism["method"] == method]
        if audited.empty or not audited["preserves_matched_equivalence"].astype(bool).all():
            raise RuntimeError(f"cannot freeze finalists: {method} failed matched-function audit")
    representation = choose_representation(summary)
    representation_method = str(representation["method"])
    if representation_method not in INTERFACES:
        raise RuntimeError(f"no frozen interface definition for {representation_method}")
    hybrid = choose_hybrid(units, representation_method)
    optimizer = choose_optimizer(summary)
    optimizer_development_method = str(optimizer["method"])
    optimizer_name = optimizer_development_method.removesuffix("[equal-HPO]")
    optimizer_kind, initialization, fixed_overrides = OPTIMIZER_DEFINITIONS[optimizer_name]

    lr_selection = pd.read_csv(processed / "equal_hpo_lr_selection.csv")
    selected_lr = lr_selection[lr_selection["selected"].astype(bool)]
    protocol = load_protocol()

    baselines = {}
    for model in ALL_MODELS:
        if model in TRAINABLE_MODELS:
            row = selected_lr[
                (selected_lr["model"] == model) & (selected_lr["method"] == "AdamW")
            ].iloc[0]
            multiplier = float(row["multiplier"])
            baselines[model] = {
                "optimizer": "adamw",
                "initialization": "default",
                "learning_rate_multiplier": multiplier,
                "optimizer_overrides": {
                    "learning_rate": float(protocol["models"][model]["learning_rate"]) * multiplier
                },
            }
        else:
            baselines[model] = {"optimizer": "native_frozen_default", "initialization": "default"}

    per_model = {}
    for model in TRAINABLE_MODELS:
        row = selected_lr[
            (selected_lr["model"] == model) & (selected_lr["method"] == optimizer_name)
        ].iloc[0]
        multiplier = float(row["multiplier"])
        per_model[model] = {
            "optimizer": optimizer_kind,
            "initialization": initialization,
            "learning_rate_multiplier": multiplier,
            "optimizer_overrides": {
                **fixed_overrides,
                "learning_rate": float(protocol["models"][model]["learning_rate"]) * multiplier,
            },
        }

    interface_definition = INTERFACES[representation_method]
    hybrid_method = str(hybrid["method"])
    alpha = float(hybrid_method.rsplit("@", 1)[1])
    now = datetime.now(timezone.utc)
    config = {
        "status": "FROZEN_BEFORE_PROSPECTIVE_DATA_ACCESS",
        "frozen_at_utc": now.isoformat(),
        "frozen_at_utc_epoch": time.time(),
        "repository_commit": protocol["repository_commit"],
        "prospective_panel_sha256": protocol_hashes()["new_prospective_panel_sha256"],
        "selection_scope": "six development datasets, three seeds; no prospective data loaded",
        "selection_rule": {
            "representation": "highest paper-method score among <=1% cost pure interfaces",
            "hybrid": "same selected interface; nontrivial alpha chosen on validation only",
            "optimizer": "highest paper-method score after equal-HPO; <=3% pool preferred",
        },
        "baselines": baselines,
        "finalists": [
            {
                "method_id": optimizer_development_method,
                "development_method": optimizer_development_method,
                "type": "optimizer",
                "applicable_models": TRAINABLE_MODELS,
                "per_model": per_model,
                "development_evidence": evidence(optimizer),
            },
            {
                "method_id": representation_method,
                "development_method": representation_method,
                "type": "interface",
                "applicable_models": ALL_MODELS,
                "interface": interface_definition["interface"],
                "interface_parameters": interface_definition["parameters"],
                "development_evidence": evidence(representation),
            },
            {
                "method_id": hybrid_method,
                "development_method": hybrid_method,
                "type": "hybrid_prediction_mixture",
                "applicable_models": ALL_MODELS,
                "interface": interface_definition["interface"],
                "interface_parameters": interface_definition["parameters"],
                "alpha": alpha,
                "selection_split": "development_validation_only",
                "development_validation_evidence": hybrid.to_dict(),
            },
        ],
        "development_artifact_hashes": {
            path.name: sha256_file(path)
            for path in required
        },
    }
    if len(config["finalists"]) > 3:
        raise RuntimeError("finalist cap exceeded")
    write_json(OUTPUT, config)
    digest = sha256_file(OUTPUT)
    OUTPUT_SHA.write_text(f"{digest}  {OUTPUT.name}\n")
    print(f"froze {len(config['finalists'])} finalists: {digest}")
    for finalist in config["finalists"]:
        print(finalist["method_id"])


if __name__ == "__main__":
    main()
