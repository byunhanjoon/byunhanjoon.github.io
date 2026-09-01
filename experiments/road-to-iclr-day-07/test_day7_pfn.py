from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import learned_structured_pfn as pfn


HERE = Path(__file__).resolve().parent


def test_heat_covariance_is_correlation_matrix() -> None:
    for scale in pfn.SCALES:
        covariance = pfn.heat_covariance(scale)
        assert covariance.shape == (pfn.N_STATES, pfn.N_STATES)
        assert np.allclose(covariance, covariance.T)
        assert np.allclose(np.diag(covariance), 1.0)
        assert np.linalg.eigvalsh(covariance).min() > 0


def test_task_generator_and_analytic_router_are_finite() -> None:
    generator = pfn.TaskGenerator(torch.device("cpu"), 17)
    task = generator.sample(16, prior=0.5, scale_index=1, noise_index=1)
    assert torch.all(task["mask"].sum(dim=1) == pfn.N_CONTEXT)
    assert generator.features(task, structured=True).shape == (16, pfn.N_STATES, 13)
    assert generator.features(task, structured=False).shape == (16, pfn.N_STATES, 5)
    predictions = pfn.analytic_predictions(task)
    assert torch.all((predictions["posterior"] >= 0) & (predictions["posterior"] <= 1))
    for name in ("zero", "always_smooth", "hard_route", "bayes_mixture", "regime_oracle"):
        assert predictions[name].shape == task["effect"].shape
        assert torch.isfinite(predictions[name]).all()


def test_routing_regret_identity() -> None:
    rng = np.random.default_rng(29)
    first = rng.normal(size=11)
    second = rng.normal(size=11)
    posterior = 0.37
    trust = 0.81
    bayes = posterior * first + (1.0 - posterior) * second
    routed = trust * first + (1.0 - trust) * second
    left = np.sum((routed - bayes) ** 2)
    right = (trust - posterior) ** 2 * np.sum((first - second) ** 2)
    assert np.isclose(left, right)


def test_sealed_result_artifacts_match_decisions() -> None:
    learned = json.loads((HERE / "results" / "learned_pfn" / "summary.json").read_text())
    nested = json.loads((HERE / "results" / "nested_backbone" / "summary.json").read_text())
    assert learned["passes"] is True
    assert learned["matched_prior"]["structured_wins_over_set"] == 9
    assert learned["matched_prior"]["positive_seed_advantages"] == 3
    assert nested["passes"] is False
    assert nested["gates"]["at_least_two_source_splits_and_sources"] is False
    assert nested["rules"]["backbone_worst"]["harmful_cells"] == 0
