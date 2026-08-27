"""Finite-sample fallback certificate for additive residual interventions.

Candidates must be fitted without the calibration labels supplied here. The
bound assumes bounded, i.i.d. calibration examples and does not certify
temporal distribution shift.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GainCertificate:
    name: str
    empirical_gain: float
    lower_bound: float


def certify_squared_loss_gains(
    residual: np.ndarray,
    interventions: dict[str, np.ndarray],
    *,
    residual_bound: float,
    intervention_bound: float,
    delta: float = 0.05,
) -> list[GainCertificate]:
    """Return simultaneous Hoeffding lower bounds on squared-loss reduction."""
    r = np.asarray(residual, dtype=np.float64).reshape(-1)
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must lie in (0, 1)")
    if residual_bound <= 0 or intervention_bound <= 0:
        raise ValueError("declared bounds must be positive")
    if np.any(np.abs(r) > residual_bound + 1e-12):
        raise ValueError("residual exceeds its declared bound")
    if not interventions:
        return []
    # For Z=2rh-h^2, [-2RH-H^2, 2RH] is a valid common interval.
    z_range = 4 * residual_bound * intervention_bound + intervention_bound**2
    penalty = z_range * np.sqrt(
        np.log(len(interventions) / delta) / (2 * len(r))
    )
    output = []
    for name, values in interventions.items():
        h = np.asarray(values, dtype=np.float64).reshape(-1)
        if h.shape != r.shape:
            raise ValueError(f"{name}: intervention shape does not match residual")
        if np.any(np.abs(h) > intervention_bound + 1e-12):
            raise ValueError(f"{name}: intervention exceeds its declared bound")
        gain = float(np.mean(2 * r * h - h * h))
        output.append(GainCertificate(name, gain, gain - float(penalty)))
    return output


def select_certified_intervention(
    certificates: list[GainCertificate],
) -> str | None:
    """Select the best positive-LCB candidate, or abstain with ``None``."""
    if not certificates:
        return None
    best = max(certificates, key=lambda item: item.lower_bound)
    return best.name if best.lower_bound > 0 else None
