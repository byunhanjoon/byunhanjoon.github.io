"""Construction and frozen-result checks for the controlled selection phase."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from analyze_synthetic_selection_phase import SHAPE, design_ids
from analyze_strength2_cover import strength2_family


HERE = Path(__file__).resolve().parent


def test_strength2_family_has_uniform_pair_margins() -> None:
    family = strength2_family(4, 2)
    assert family.shape == (27_648, 16, 4)
    for left in range(len(SHAPE)):
        for right in range(left + 1, len(SHAPE)):
            encoded = family[:, :, left] * SHAPE[right] + family[:, :, right]
            expected = 16 // (SHAPE[left] * SHAPE[right])
            for level_pair in range(SHAPE[left] * SHAPE[right]):
                assert np.all((encoded == level_pair).sum(axis=1) == expected)


def test_design_ids_are_valid_complete_product_indices() -> None:
    ids = design_ids(strength2_family(4, 2))
    assert ids.shape == (27_648, 16)
    assert np.all((ids >= 0) & (ids < np.prod(SHAPE)))
    assert np.all(np.apply_along_axis(lambda row: len(np.unique(row)), 1, ids) == 16)


def test_frozen_selection_phase_gate_passes() -> None:
    result = json.loads((HERE / "results/synthetic_selection_phase_summary.json").read_text())
    assert result["draws_per_cell"] == 65_536
    assert result["frozen_gate_passed"]
    assert all(result["clauses"].values())
    assert len(result["cells"]) == 32
