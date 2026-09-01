from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from safe_basis.common import PANEL_PATH, mix_predictions, normalized_excess_risk, sha256_file
from safe_basis.gating import alpha_evidence, select_gates, verify_alpha_zero
from safe_basis.rankgram import anchor_count, empirical_rank, rank_adaptive_block


def test_prospective_panel_is_locked_and_balanced() -> None:
    panel = json.loads(PANEL_PATH.read_text())
    assert panel["status"] == "LOCKED_BEFORE_DEVELOPMENT_OUTCOME_ACCESS"
    assert 8 <= len(panel["datasets"]) <= 12
    assert {row["problem_type"] for row in panel["datasets"]} == {"classification", "regression"}
    assert all(1_000 <= row["rows"] <= 50_000 and row["raw_columns"] <= 100 for row in panel["datasets"])
    assert panel["selection_evidence"]["outcomes_accessed_before_lock"] is False


def test_panel_sha_sidecar_matches() -> None:
    expected = PANEL_PATH.with_suffix(".sha256").read_text().split()[0]
    assert expected == sha256_file(PANEL_PATH)


def test_prediction_mixture_endpoints() -> None:
    raw = np.array([1.0, 2.0])
    gram = np.array([3.0, 4.0])
    np.testing.assert_allclose(mix_predictions(raw, gram, 0.0), raw)
    np.testing.assert_allclose(mix_predictions(raw, gram, 1.0), gram)
    np.testing.assert_allclose(mix_predictions(raw, gram, 0.25), np.array([1.5, 2.5]))


def test_normalized_excess_exact_classification() -> None:
    y_train = np.array([0, 0, 1, 1])
    y = np.array([0, 1])
    raw = np.array([[0.8, 0.2], [0.2, 0.8]])
    same = normalized_excess_risk("classification", y, raw, raw, y_train)
    assert same["normalized_excess_risk"] == 0.0
    assert same["trivial_loss"] > same["raw_loss"]


def test_safe_gate_fallback_and_monotonic_choice() -> None:
    y_train = np.array([0, 0, 1, 1, 0, 1])
    y = np.array([0, 1, 0, 1])
    raw = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])
    bad = raw[:, ::-1]
    records = alpha_evidence(
        "classification", y, y_train, raw, bad,
        alphas=[0.0, 0.25, 0.5, 0.75, 1.0], bootstrap_resamples=50, seed=7,
    )
    verify_alpha_zero(records)
    gates = select_gates(records, taus=[0.0, 0.01], constrained_lambda_multipliers=[0.001])
    assert gates["SafeGram-t0"] == 0.0
    assert gates["SafeGram-t01"] == 0.0


def test_rank_rules() -> None:
    assert anchor_count(4, "rank") == 4
    assert anchor_count(4, "rank_plus_one") == 5
    assert anchor_count(4, "double_rank_capped_16") == 8
    assert anchor_count(10, "double_rank_capped_16") == 16
    assert anchor_count(4, "fixed_16") == 16


def test_empirical_rank_relative_threshold() -> None:
    values = np.diag([10.0, 1.0, 1e-5])
    assert empirical_rank(values, 1e-4)[0] == 2
    assert empirical_rank(values, 1e-7)[0] == 3


def test_rankgram_rotation_invariance_and_reconstruction() -> None:
    rng = np.random.default_rng(10)
    train = rng.normal(size=(100, 4))
    validation = rng.normal(size=(20, 4))
    test = rng.normal(size=(30, 4))
    q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    config = dict(
        relative_threshold=1e-6,
        anchor_rule="rank",
        normalization="N1_anchor_norm",
        standardize=True,
    )
    left, metadata = rank_adaptive_block((train, validation, test), **config)
    right, rotated_metadata = rank_adaptive_block((train @ q, validation @ q, test @ q), **config)
    for first, second in zip(left, right):
        np.testing.assert_allclose(first, second, atol=1e-9, rtol=1e-9)
    assert metadata["empirical_rank"] == 4
    assert metadata["reconstruction_error"] < 1e-10
    assert rotated_metadata["reconstruction_error"] < 1e-10


def test_rankgram_normalizations_finite() -> None:
    rng = np.random.default_rng(1)
    splits = (rng.normal(size=(50, 5)), rng.normal(size=(10, 5)), rng.normal(size=(11, 5)))
    for normalization in ("N0_raw_inner_product", "N1_anchor_norm", "N2_cosine", "N3_block_rms"):
        outputs, metadata = rank_adaptive_block(
            splits,
            relative_threshold=1e-6,
            anchor_rule="rank_plus_one",
            normalization=normalization,
            standardize=False,
        )
        assert all(np.isfinite(values).all() for values in outputs)
        assert metadata["coordinate_dimension"] == 6
