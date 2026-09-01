"""Checks for independent screening and precise deployment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from analyze_cheap_screen_precise_deploy import action_ids


HERE = Path(__file__).resolve().parent


def test_allocation_cost_identity() -> None:
    for candidates, expected_saving in ((5, .35), (4, .25), (3, 1 / 12)):
        proposed = 16 * candidates + 64 * 2
        control = 64 * candidates
        assert np.isclose(1 - proposed / control, expected_saving)


def test_pilot_and_deployment_use_distinct_reproducible_streams() -> None:
    shape = (4, 4, 2, 4)
    pilot_a, blocks_a = action_ids(shape, "test", "dataset")
    pilot_b, blocks_b = action_ids(shape, "test", "dataset")
    assert pilot_a.shape == (1_024, 16)
    assert blocks_a.shape == (1_024, 4, 16)
    assert np.array_equal(pilot_a, pilot_b)
    assert np.array_equal(blocks_a, blocks_b)
    assert not np.array_equal(pilot_a, blocks_a[:, 0])
    assert all(not np.array_equal(blocks_a[:, left], blocks_a[:, right])
               for left in range(4) for right in range(left + 1, 4))


def test_frozen_allocator_gate_passes_and_first_rule_fails() -> None:
    passed = json.loads((HERE / "results/cheap_screen_precise_deploy_summary.json").read_text())
    failed = json.loads((HERE / "results/screen_then_cross_summary.json").read_text())
    assert passed["frozen_gate_passed"]
    assert passed["panels_passing_by_clause"] == {
        "saving": 4, "inclusion": 5, "agreement": 4, "regret": 4,
    }
    assert not failed["frozen_gate_passed"]
