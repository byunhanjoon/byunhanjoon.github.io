import numpy as np
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def test_trust_parabola_matches_direct_risk_difference():
    mu = np.array([0.8, -0.3, 0.2])
    g = np.array([0.5, -0.1, 0.4])
    lam = 0.37
    direct = np.mean(mu**2 - (mu - lam * g) ** 2)
    formula = 2 * lam * np.mean(mu * g) - lam**2 * np.mean(g**2)
    assert abs(direct - formula) < 1e-14


def test_clipped_oracle_dominates_zero_and_full():
    rng = np.random.default_rng(7)
    for _ in range(100):
        mu = rng.normal(size=20)
        g = rng.normal(size=20)
        cross = np.mean(mu * g)
        square = np.mean(g**2)
        lam = np.clip(cross / square, 0.0, 1.0)
        gain = lambda x: 2 * x * cross - x * x * square
        assert gain(lam) >= gain(0.0) - 1e-14
        assert gain(lam) >= gain(1.0) - 1e-14


def test_completed_replay_has_all_cells_and_finite_results():
    cells = np.genfromtxt(HERE / "results" / "cells.csv", delimiter=",", names=True, dtype=None, encoding="utf-8")
    assert len(cells) == 45 * 5
    assert np.isfinite(cells["outer_gain"]).all()
    summary = json.loads((HERE / "results" / "summary.json").read_text())
    assert summary["status"] == "complete"
    assert summary["recommend_confirmation"] is False


def test_neural_matrix_and_base_exclusion_audit():
    cells = np.genfromtxt(HERE / "results" / "neural" / "cells.csv", delimiter=",", names=True, dtype=None, encoding="utf-8")
    assert len(cells) == 4 * 2 * 9
    assert cells["oof_finite"].all()
    assert np.isfinite(cells["predicted_gain"]).all()
    assert np.isfinite(cells["actual_gain"]).all()
    for task in ("acs_occupation", "tlc_pickup_zone", "airline_origin_airport", "medical_charges"):
        manifest = json.loads((HERE.parent / "mpe_iclr" / "processed" / task / "manifest.json").read_text())
        ordinary = set(manifest["ordinary_covariates"])
        assert "field_state" not in ordinary
        assert "target" not in ordinary


def test_pfn_prior_phase_matrix_and_matched_bayes_check():
    cells = np.genfromtxt(HERE / "results" / "pfn_prior" / "cells.csv", delimiter=",", names=True, dtype=None, encoding="utf-8")
    assert len(cells) == 3 * 3 * 3
    matched = cells[np.isclose(cells["true_prior"], 0.5)]
    assert np.all(matched["mse_bayes_mixture"] <= matched["mse_zero"])
    assert np.all(matched["mse_bayes_mixture"] <= matched["mse_always_smooth"])
    hard_gap = matched["mse_bayes_mixture"] - matched["mse_hard_route"]
    combined_mc = 2 * (matched["se_bayes_mixture"] + matched["se_hard_route"])
    assert np.all(hard_gap <= combined_mc)
    assert np.isfinite(cells["posterior_auroc"]).all()
