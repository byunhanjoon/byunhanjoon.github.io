#!/usr/bin/env python3
"""Integrity audit separated from scientific go/no-go decisions."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MAIN_RUNS = (
    "phase_a_pilot",
    "phase_b_recovery",
    "phase_c_operand",
    "phase_d_regime",
    "phase_e_temporal",
    "exact_execution_ablation",
    "phase_f_typed",
    "phase_g_residual",
    "real_bike_temporal",
    "real_bike_bounded_diagnostic",
    "general_pilot",
    "depth_scaling",
    "regime_scaling",
)


def main() -> None:
    rows = []
    for name in MAIN_RUNS:
        path = RESULTS / name / "audit.json"
        if not path.exists():
            rows.append({"run": name, "integrity": False, "scientific_gate": None, "reason": "missing audit"})
            continue
        audit = json.loads(path.read_text(encoding="utf-8"))
        expected = audit.get("expected_records", audit.get("completeness", {}).get("expected_cells"))
        observed = audit.get("observed_records", audit.get("completeness", {}).get("observed_cells"))
        finite = audit.get(
            "all_finite",
            audit.get("all_prediction_metrics_finite", audit.get("completeness", {}).get("checks", {}).get("all_metrics_finite", True)),
        )
        integrity = bool(expected == observed and finite)
        rows.append(
            {
                "run": name,
                "expected_records": expected,
                "observed_records": observed,
                "finite": bool(finite),
                "integrity": integrity,
                "scientific_gate": audit.get("gate", {}).get("passed"),
                "confirmatory": audit.get("confirmatory", True),
            }
        )
    result = {
        "all_integrity_checks_passed": all(row["integrity"] for row in rows),
        "run_count": len(rows),
        "runs": rows,
    }
    output = RESULTS / "study_integrity_audit.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["all_integrity_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
