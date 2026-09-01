from __future__ import annotations

import torch

from tabalu.models.operand import build_operand_estimator


def test_operand_variants_preserve_shape_and_start_at_identity() -> None:
    observed = torch.randn(16, 4)
    mean = torch.zeros(4)
    scale = torch.ones(4)
    for variant in ("raw", "bounded_correction", "unrestricted_encoder"):
        estimator = build_operand_estimator(variant, mean, scale)
        estimated = estimator(observed)
        assert estimated.shape == observed.shape
        torch.testing.assert_close(estimated, observed)


def test_bounded_correction_cannot_exceed_declared_bound() -> None:
    observed = torch.randn(32, 3)
    scale = torch.tensor([1.0, 2.0, 0.5])
    estimator = build_operand_estimator("bounded_correction", torch.zeros(3), scale)
    with torch.no_grad():
        estimator.network[-1].bias.fill_(100.0)
    normalized = (estimator(observed) - observed).abs() / scale
    assert float(normalized.max()) <= 0.300001


def test_confidence_gate_reports_usage() -> None:
    observed = torch.randn(8, 2)
    estimator = build_operand_estimator("confidence_gated", torch.zeros(2), torch.ones(2))
    estimated = estimator(observed)
    diagnostics = estimator.diagnostics(observed, estimated)
    assert 0 < float(diagnostics["observed_confidence"]) < 1
