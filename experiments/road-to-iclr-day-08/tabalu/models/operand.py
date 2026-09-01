"""Conservative and unrestricted operand estimators for Phase C."""

from __future__ import annotations

import torch
from torch import nn


class OperandEstimator(nn.Module):
    variant: str

    def diagnostics(self, observed: torch.Tensor, estimated: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"correction_rms": (estimated - observed).square().mean().sqrt()}


class RawOperand(OperandEstimator):
    variant = "raw"

    def forward(self, observed: torch.Tensor) -> torch.Tensor:
        return observed


class _NormalizedEstimator(OperandEstimator):
    def __init__(self, mean: torch.Tensor, scale: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("mean", mean.detach().clone())
        self.register_buffer("scale", scale.detach().clone().clamp_min(1.0e-6))

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mean) / self.scale


def _mlp(input_width: int, output_width: int) -> nn.Sequential:
    network = nn.Sequential(
        nn.Linear(input_width, 64),
        nn.SiLU(),
        nn.Linear(64, 64),
        nn.SiLU(),
        nn.Linear(64, output_width),
    )
    nn.init.zeros_(network[-1].weight)
    nn.init.zeros_(network[-1].bias)
    return network


class BoundedCorrection(_NormalizedEstimator):
    variant = "bounded_correction"

    def __init__(self, mean: torch.Tensor, scale: torch.Tensor, bound: float = 0.30) -> None:
        _NormalizedEstimator.__init__(self, mean, scale)
        self.bound = bound
        self.network = _mlp(len(mean), len(mean))

    def forward(self, observed: torch.Tensor) -> torch.Tensor:
        correction = self.bound * self.scale * torch.tanh(self.network(self.normalize(observed)))
        return observed + correction


class ConfidenceGatedReconstruction(_NormalizedEstimator):
    variant = "confidence_gated"

    def __init__(self, mean: torch.Tensor, scale: torch.Tensor) -> None:
        _NormalizedEstimator.__init__(self, mean, scale)
        self.network = _mlp(len(mean), 2 * len(mean))
        with torch.no_grad():
            self.network[-1].bias[len(mean) :].fill_(2.0)

    def forward(self, observed: torch.Tensor) -> torch.Tensor:
        reconstruction_raw, confidence_logits = self.network(self.normalize(observed)).chunk(2, dim=-1)
        reconstruction = self.mean + 3.0 * self.scale * torch.tanh(reconstruction_raw)
        confidence = torch.sigmoid(confidence_logits)
        return confidence * observed + (1.0 - confidence) * reconstruction

    def diagnostics(self, observed: torch.Tensor, estimated: torch.Tensor) -> dict[str, torch.Tensor]:
        output = super().diagnostics(observed, estimated)
        with torch.no_grad():
            _, confidence_logits = self.network(self.normalize(observed)).chunk(2, dim=-1)
            output["observed_confidence"] = torch.sigmoid(confidence_logits).mean()
        return output


class UnrestrictedEncoder(_NormalizedEstimator):
    variant = "unrestricted_encoder"

    def __init__(self, mean: torch.Tensor, scale: torch.Tensor) -> None:
        _NormalizedEstimator.__init__(self, mean, scale)
        self.network = _mlp(len(mean), len(mean))

    def forward(self, observed: torch.Tensor) -> torch.Tensor:
        return observed + self.scale * self.network(self.normalize(observed))


def build_operand_estimator(
    variant: str, mean: torch.Tensor, scale: torch.Tensor
) -> OperandEstimator:
    if variant == "raw":
        return RawOperand()
    if variant == "bounded_correction":
        return BoundedCorrection(mean, scale)
    if variant == "confidence_gated":
        return ConfidenceGatedReconstruction(mean, scale)
    if variant == "unrestricted_encoder":
        return UnrestrictedEncoder(mean, scale)
    raise KeyError(variant)
