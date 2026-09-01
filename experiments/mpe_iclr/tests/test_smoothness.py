from __future__ import annotations

import numpy as np

from smoothness_diagnostic import diagnostic_from_state_residuals, fold_assignment


def test_fold_assignment_is_deterministic_and_target_independent() -> None:
    row_ids = np.asarray([f"row-{index}" for index in range(100)])
    left = fold_assignment(row_ids, 2)
    right = fold_assignment(row_ids, 2)
    assert np.array_equal(left, right)
    assert set(left) == set(range(5))


def test_smooth_residuals_have_negative_distance_difference_correlation() -> None:
    coordinate = np.arange(12, dtype=float)
    distance = np.abs(coordinate[:, None] - coordinate[None, :])
    residual = np.sin(coordinate / 4.0)
    result = diagnostic_from_state_residuals(distance, np.arange(12), residual)
    assert np.isfinite(result["distance_residual_difference_spearman"])
    assert result["prespecified_smoothness"] == -result["distance_residual_difference_spearman"]
    assert result["state_pairs"] == 66
