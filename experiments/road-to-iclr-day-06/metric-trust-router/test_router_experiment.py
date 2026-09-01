from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

import router_experiment as router


HERE = Path(__file__).resolve().parent


def test_protocol_hash_matches() -> None:
    expected = (HERE / "PROTOCOL_SHA256.txt").read_text().split()[0]
    observed = hashlib.sha256((HERE / "EXPLORATORY_PROTOCOL.md").read_bytes()).hexdigest()
    assert observed == expected


def test_five_folds_exactly_partition_outer_training_states() -> None:
    task = router.load_task("employee_salaries")
    outer = router.split_state_indices(task, 0)
    folds = router.state_folds(task, 0)
    assert len(folds) == 5
    assert set(np.concatenate(folds).tolist()) == set(outer["train"].tolist())
    for left in range(5):
        for right in range(left + 1, 5):
            assert set(folds[left]).isdisjoint(set(folds[right]))
    assert set(np.concatenate(folds)).isdisjoint(set(outer["validation"]))
    assert set(np.concatenate(folds)).isdisjoint(set(outer["test"]))
