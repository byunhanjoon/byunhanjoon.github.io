from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("analyze_hpo_quotient", HERE / "analyze_hpo_quotient.py")
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_proper_loss_ambiguity_identity() -> None:
    rng = np.random.default_rng(7)
    logits = rng.normal(size=(4, 3, 2, 19, 2))
    exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
    predictions = exp / exp.sum(axis=-1, keepdims=True)
    result = MODULE.orbit_metrics(predictions, rng.integers(0, 2, 19))
    assert result["ambiguity_identity_error"] < 1e-14


def test_gather_per_schema() -> None:
    predictions = np.zeros((2, 2, 2, 2, 3, 2))
    predictions[0, ..., 0] = 1
    predictions[1, ..., 1] = 1
    choices = np.indices((2, 2, 2)).sum(axis=0) % 2
    selected = MODULE.gather_per_schema(predictions, choices)
    assert np.all(np.argmax(selected, axis=-1) == choices[..., None])


def test_switch_decomposition_closes() -> None:
    rng = np.random.default_rng(9)
    baseline = rng.normal(size=(4, 4, 2, 13, 2))
    selected = baseline + rng.normal(scale=0.1, size=baseline.shape)
    result = MODULE.switch_decomposition(baseline, selected)
    assert result["reconstruction_error"] < 1e-14

