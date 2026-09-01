"""Fast integrity tests that do not open any evaluation labels."""

from __future__ import annotations

import numpy as np

from common import CONFIG, gaussian_scores, make_coefficients
from train_head import ProjectiveHead


def test_coefficients_are_group_local_and_well_scaled() -> None:
    families, coefficients = make_coefficients(19)
    groups, q = int(CONFIG["query_groups"]), int(CONFIG["query_size"])
    assert coefficients.shape == (len(families), groups, groups * q)
    for family_index, family in enumerate(families):
        for group in range(groups):
            outside = np.r_[0 : group * q, (group + 1) * q : groups * q]
            assert np.max(np.abs(coefficients[family_index, group, outside]), initial=0.0) == 0.0
            norm = np.linalg.norm(coefficients[family_index, group])
            if family not in {"subset_mean", "scaled_dense"}:
                assert np.isclose(norm, 1.0)


def test_gaussian_crps_is_nonnegative() -> None:
    scores = gaussian_scores(np.array([0.0, 1.0]), np.array([0.0, 0.0]), np.ones(2))
    assert np.all(scores["crps"] >= 0.0)
    assert scores["crps"][0] < scores["crps"][1]


def test_head_kernel_has_unit_diagonal() -> None:
    import torch

    torch.manual_seed(3)
    model = ProjectiveHead(12, 5, 8)
    hidden = torch.randn(2, 7, 12)
    unit = model.features(hidden)
    kernel = torch.einsum("snr,smr->nm", unit, unit) / unit.shape[0]
    assert torch.allclose(torch.diag(kernel), torch.ones(7), atol=1e-6)
    assert torch.linalg.eigvalsh(kernel).min() > -1e-5
