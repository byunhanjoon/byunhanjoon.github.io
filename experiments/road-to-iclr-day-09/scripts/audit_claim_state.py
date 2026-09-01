#!/usr/bin/env python3
"""Assert the final mix of positive and negative Day-09 claim gates."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "results" / "processed"


def load(name: str) -> dict:
    return json.loads((PROCESSED / name).read_text())


def main() -> None:
    synthetic = load("fallback_loss_router_audit_v1.json")
    breadth = load("openml_breadth_competence_audit_v1.json")
    regression = load("regression_confirmation_audit_v1.json")
    classification = load("classification_shrinkage_confirmation_audit_v1.json")
    rescaled = load("context_rescaled_confirmation_audit_v1.json")

    observed = {
        "synthetic_opportunity_pass": synthetic["performance_opportunity_gate_pass"],
        "openml_strong_transfer_pass": breadth["strong_transfer_pass"],
        "openml_scoped_transfer_pass": breadth["scoped_transfer_pass"],
        "regression_confirmation_pass": regression["confirmation_pass"],
        "classification_shrinkage_pass": classification["confirmation_pass"],
        "context_classification_pass": rescaled["tasks"]["classification"]["gate_pass"],
        "context_regression_pass": rescaled["tasks"]["regression"]["gate_pass"],
        "context_joint_pass": rescaled["joint_robustness_pass"],
    }
    expected = {
        "synthetic_opportunity_pass": True,
        "openml_strong_transfer_pass": False,
        "openml_scoped_transfer_pass": True,
        "regression_confirmation_pass": True,
        "classification_shrinkage_pass": True,
        "context_classification_pass": True,
        "context_regression_pass": False,
        "context_joint_pass": False,
    }
    if observed != expected:
        raise AssertionError(f"claim-state drift: observed={observed}, expected={expected}")
    print(json.dumps(observed, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
