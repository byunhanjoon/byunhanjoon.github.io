"""Analyze the secondary CatBoost/XGBoost independent-seed scope."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import closure_core as core
from analysis_utils import squared_residual, write_summary
from closure_designs import sample_schema_design
from analyze_experiment_a import choose_pool_predictions


OUT = core.HERE / "summaries"


def main() -> None:
    manifests = sorted((core.RAW / "experiment_a_classical").glob("*/manifest.json"))
    expected = len(core.CONFIG["all_datasets"]) * len(core.CONFIG["secondary_a_models"])
    if len(manifests) != expected:
        raise AssertionError(f"secondary A cells missing {len(manifests)}/{expected}")
    rows = []
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text()); path = manifest_path.parent
        canonical = np.load(path / "canonical_test.npy", mmap_mode="r")
        joint = np.load(path / "joint_test.npy", mmap_mode="r")
        cards = tuple(int(value) for value in manifest["schema_cards"])
        qcanonical = np.asarray(canonical, dtype=np.float64).mean(axis=0)
        qjoint = np.asarray(joint, dtype=np.float64).mean(axis=(0, 1))
        rng = np.random.default_rng(core.stable_seed("A-secondary-analysis", path.name))
        method_residuals = {method: [] for method in (
            "CANONICAL-INDEPENDENT", "IID-JOINT", "SRS-JOINT", "OC1-INDEPENDENT", "OC2-INDEPENDENT"
        )}
        for _ in range(512):
            estimate = canonical[rng.choice(len(canonical), 16, replace=False)].mean(axis=0)
            method_residuals["CANONICAL-INDEPENDENT"].append(squared_residual(estimate, qcanonical))
            for method in ("IID-JOINT", "SRS-JOINT", "OC1-INDEPENDENT", "OC2-INDEPENDENT"):
                for _attempt in range(100):
                    design = sample_schema_design(method, cards, 16, rng)
                    _, counts = np.unique(design, axis=0, return_counts=True)
                    if counts.max() <= joint.shape[1]:
                        break
                estimate = choose_pool_predictions(joint, design, cards, rng).mean(axis=0)
                method_residuals[method].append(squared_residual(estimate, qjoint))
        for method, values in method_residuals.items():
            rows.append({
                "dataset": manifest["dataset"], "model": manifest["model"],
                "task": manifest["task"], "method": method, "budget": 16,
                "residual_mean": float(np.mean(values)),
                "residual_median": float(np.median(values)),
                "canonical_joint_distance": squared_residual(qcanonical, qjoint),
            })
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "experiment_a_classical_cells.csv", index=False)
    pivot = frame.pivot_table(index=["dataset", "model"], columns="method", values="residual_mean").reset_index()
    model_summary = {}
    for model, group in pivot.groupby("model"):
        canonical = group["CANONICAL-INDEPENDENT"].mean()
        oc2 = group["OC2-INDEPENDENT"].mean()
        model_summary[model] = {
            "canonical_residual": float(canonical), "oc2_independent_residual": float(oc2),
            "oc2_relative_reduction": float(1 - oc2 / canonical) if canonical > 0 else None,
            "oc2_cell_wins": int((group["OC2-INDEPENDENT"] < group["CANONICAL-INDEPENDENT"]).sum()),
            "cells": len(group),
        }
    summary = {"status": "complete", "cells": expected, "model_summary": model_summary}
    write_summary(OUT / "experiment_a_classical_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
