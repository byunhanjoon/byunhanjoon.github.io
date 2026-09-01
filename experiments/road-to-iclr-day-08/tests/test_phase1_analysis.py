from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.analysis.phase1 import (
    NUMERICAL_INVERSE_RTOL,
    NUMERICAL_ORDER_TIE_RTOL,
    bootstrap_mean,
    expected_jobs,
    validate_transform_audit,
)
from src.analysis.runner import load_config


def test_frozen_phase1_grid_has_expected_size_and_unique_jobs():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "audit" / "pilot.yaml")
    jobs = expected_jobs(config)
    assert len(jobs) == 12 * 7 * 13 * 3
    assert len(set(jobs)) == len(jobs)


def test_dataset_bootstrap_is_deterministic_and_contains_mean():
    values = np.asarray([-0.1, 0.2, 0.3, 0.7])
    first = bootstrap_mean(values, seed=19, draws=2_000)
    second = bootstrap_mean(values, seed=19, draws=2_000)
    assert first == second
    assert first["ci_low"] <= first["mean"] <= first["ci_high"]


def _exact_audit(error: float) -> dict:
    return {
        "missing_mask_preserved": True,
        "all_finite_inputs_have_finite_outputs": True,
        "strict_order_violations": 0,
        "max_rel_reconstruction_error": error,
        "metadata": {"exactness_class": "exact analytic bijection"},
    }


def test_numerical_inverse_tolerance_accepts_float64_roundoff():
    validate_transform_audit(_exact_audit(3.51e-7), "within-tolerance")
    validate_transform_audit(_exact_audit(NUMERICAL_INVERSE_RTOL), "boundary")


def test_numerical_inverse_tolerance_rejects_material_error():
    with pytest.raises(ValueError, match="numerical inverse tolerance failed"):
        validate_transform_audit(_exact_audit(NUMERICAL_INVERSE_RTOL * 1.01), "too-large")


def test_order_ties_are_accepted_only_at_float64_dust_scale():
    negligible = _exact_audit(NUMERICAL_ORDER_TIE_RTOL)
    negligible["strict_order_violations"] = 6
    validate_transform_audit(negligible, "one-ulp-ties")

    material = _exact_audit(NUMERICAL_ORDER_TIE_RTOL * 1.01)
    material["strict_order_violations"] = 1
    with pytest.raises(ValueError, match="material order collision"):
        validate_transform_audit(material, "material-tie")


def test_missingness_change_is_always_rejected():
    audit = _exact_audit(0.0)
    audit["missing_mask_preserved"] = False
    with pytest.raises(ValueError, match="transform audit failed"):
        validate_transform_audit(audit, "missingness")


def test_declared_lossy_rank_transform_can_have_order_ties():
    audit = _exact_audit(0.5)
    audit["metadata"]["exactness_class"] = "order-preserving but lossy because of ties/finite precision"
    audit["strict_order_violations"] = 12
    validate_transform_audit(audit, "declared-lossy")


def test_order_reversal_is_rejected_even_with_exact_roundtrip():
    audit = _exact_audit(0.0)
    audit["strict_order_violations"] = 1
    audit["strict_order_reversals"] = 1
    audit["max_order_tie_relative_input_gap"] = 0.0
    with pytest.raises(ValueError, match="order reversal"):
        validate_transform_audit(audit, "reversal")
