from __future__ import annotations

import analyze_successor as analysis
import pytest


def test_relative_improvement_uses_lower_is_better() -> None:
    assert analysis.relative_improvement(1.0, 0.9) == pytest.approx(0.1)


def test_paired_summary_counts_strict_wins() -> None:
    result = analysis.paired_summary([(1.0, 0.9), (1.0, 1.0), (2.0, 2.2)])
    assert result["cells"] == 3
    assert result["wins"] == 1
    assert result["win_fraction"] == 1 / 3


def test_e2_gate_requires_and_accepts_all_frozen_checks() -> None:
    sources = {
        "acs_occupation": "ACS",
        "tlc_dropoff_zone": "NYC_TLC",
        "citibike_start_station": "CITI_BIKE",
        "medical_charges": "STRING_BENCHMARK",
    }
    scores = {
        "raw_base": 1.0,
        "lookup_unknown": 1.0,
        "transport_zero": 0.95,
        "transport_first_order": 0.80,
        "transport_shuffled_metric": 1.05,
    }
    cells = []
    for task in analysis.E1B_TASKS:
        for split in (0, 1):
            for seed in analysis.NEURAL_SEEDS:
                for condition in analysis.E2_CONDITIONS:
                    cells.append(
                        {
                            "task": task,
                            "source_unit": sources[task],
                            "split": split,
                            "seed": seed,
                            "condition": condition,
                            "result": {analysis.SCORE_KEY: scores[condition]},
                        }
                    )
    result = analysis.analyze_e2(cells, {"promotion_gate": {"promote_to_e2": True}})
    assert result["complete"]
    assert result["success_gate"]["success"]
    assert all(result["success_gate"]["checks"].values())
