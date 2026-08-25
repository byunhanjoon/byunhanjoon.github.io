"""Fail-closed completion audit for the untouched external experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .external_measure_orbit import RESULTS, ROOT, config, prediction_path


EXPECTED_HASHES = {
    "experiments/day3/configs/external_measure_orbit_preregistered.json": "c1634756feed51a8ea52f1ee20b2e0d39d37e9e6bc80a4f78fcf191575433969",
    "experiments/day3/external_measure_orbit.py": "9348543246688a0d74870a451da35a072890c7962bfa0b72edda9a8108d143ea",
    "experiments/day3/analyze_external_measure_orbit.py": "7716b060d16d9f759810c7c06307f1ac3bc252d145ddd8111f6df9e6b4c4b92c",
    "EXTERNAL_MEASURE_ORBIT_PROTOCOL.md": "97285709741cdecad6443582a84a7acf24bf6ae93592d38668b79e1a583f6b22",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    cfg = config()
    run_paths = sorted(RESULTS.glob("runs*.csv"))
    runs = (
        pd.concat([pd.read_csv(path) for path in run_paths], ignore_index=True)
        if run_paths
        else pd.DataFrame()
    )
    expected_runs = len(cfg["datasets"]) * len(cfg["seeds"]) * len(cfg["arms"])
    key = ["dataset", "seed", "arm"]
    duplicates = int(runs.duplicated(key).sum()) if len(runs) else 0
    failures = int(runs.failure.fillna("").ne("").sum()) if len(runs) else 0
    expected_cells = {
        (dataset, int(seed), arm)
        for dataset in cfg["datasets"]
        for seed in cfg["seeds"]
        for arm in cfg["arms"]
    }
    observed_cells = (
        {
            (str(row.dataset), int(row.seed), str(row.arm))
            for row in runs.itertuples()
        }
        if len(runs)
        else set()
    )
    prediction_count = sum(
        prediction_path(dataset, seed, arm).exists()
        for dataset, seed, arm in expected_cells
    )
    parameter_matched = bool(
        len(runs)
        and (runs.groupby(["dataset", "seed"]).parameters.nunique() == 1).all()
    )
    update_matched = True
    if len(runs):
        for (_, _), group in runs.groupby(["dataset", "seed"]):
            by_arm = group.set_index("arm")
            update_matched &= int(
                by_arm.loc["measure_orbit", "gradient_updates"]
            ) == int(
                by_arm.loc[
                    "baseline_seedmate_update_matched", "gradient_updates"
                ]
            )
    else:
        update_matched = False
    numeric_finite = bool(
        len(runs)
        and np.isfinite(
            runs[
                [
                    "val_proper_loss",
                    "test_proper_loss",
                    "test_metric",
                    "gradient_updates",
                    "train_seconds",
                ]
            ].to_numpy()
        ).all()
    )

    hashes = {relative: sha256(ROOT / relative) for relative in EXPECTED_HASHES}
    hash_match = hashes == EXPECTED_HASHES
    summary_path = RESULTS / "analysis_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    report_path = ROOT / "EXTERNAL_MEASURE_ORBIT_REPORT.md"
    report_text = report_path.read_text() if report_path.exists() else ""

    checks = {
        "frozen_artifact_hashes": {
            "passes": hash_match,
            "observed": hashes,
        },
        "matrix_coverage": {
            "passes": len(runs) == expected_runs
            and observed_cells == expected_cells
            and duplicates == 0,
            "expected_runs": expected_runs,
            "completed_runs": len(runs),
            "missing_cells": len(expected_cells - observed_cells),
            "unexpected_cells": len(observed_cells - expected_cells),
            "duplicates": duplicates,
        },
        "training_integrity": {
            "passes": failures == 0
            and parameter_matched
            and update_matched
            and numeric_finite,
            "failures": failures,
            "parameter_matched": parameter_matched,
            "exact_gradient_update_match": update_matched,
            "numeric_finite": numeric_finite,
        },
        "prediction_coverage": {
            "passes": prediction_count == expected_runs,
            "expected": expected_runs,
            "found": prediction_count,
        },
        "frozen_gate_evaluated": {
            "passes": summary.get("external_claim_validated") is False
            and summary.get("primary_vs_two_seed_prediction_ensemble", {}).get(
                "passed"
            )
            is False
            and summary.get("preservation_vs_single_baseline", {}).get("passed")
            is False,
            "external_claim_validated": summary.get("external_claim_validated"),
            "primary": summary.get("primary_vs_two_seed_prediction_ensemble"),
            "preservation": summary.get("preservation_vs_single_baseline"),
        },
        "negative_result_reported": {
            "passes": "external performance claim is rejected" in report_text.lower()
            and "do not claim selective measure-orbit" in report_text.lower(),
            "report": "EXTERNAL_MEASURE_ORBIT_REPORT.md",
        },
    }
    complete = all(item["passes"] for item in checks.values())
    payload = {
        "interpretation": (
            "Completion means the frozen test was executed and audited; it does "
            "not mean the external scientific claim passed."
        ),
        "checks": checks,
        "experiment_complete": complete,
        "external_claim_validated": summary.get("external_claim_validated"),
    }
    output = RESULTS / "completion_audit.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    raise SystemExit(0 if complete else 1)


if __name__ == "__main__":
    main()
