from __future__ import annotations

import numpy as np
import pandas as pd

from control_suite import seen_row_partition, small_state_partition
from representations import TaskData


def synthetic_task() -> TaskData:
    states = [f"s{index:02d}" for index in range(3)]
    rows = pd.DataFrame(
        {
            "row_id": [f"r{index:03d}" for index in range(150)],
            "field_state": np.repeat(states, 50),
            "target": np.arange(150, dtype=float),
        }
    )
    return TaskData(
        name="synthetic", rows=rows, states=pd.DataFrame({"state_id": states}),
        distance=np.ones((3, 3)) - np.eye(3), splits={},
        manifest={"status": "RUN", "ordinary_covariates": [], "source_unit": "TEST"}, arrays={},
    )


def test_seen_row_partition_is_disjoint_deterministic_and_covers_states() -> None:
    task = synthetic_task()
    left = seen_row_partition(task, 17)
    right = seen_row_partition(task, 17)
    assert all(np.array_equal(left[name], right[name]) for name in left)
    assert not (set(left["train"]) & set(left["validation"]))
    assert not (set(left["train"]) & set(left["test"]))
    assert not (set(left["validation"]) & set(left["test"]))
    state = task.rows["field_state"].to_numpy()
    assert all(set(state[indices]) == set(task.state_ids) for indices in left.values())


def test_small_nominal_partition_retains_all_levels_without_overlap() -> None:
    states = [f"level-{index}" for index in range(6)]
    parts = small_state_partition(states, 41)
    assert set().union(*map(set, parts.values())) == set(states)
    assert sum(map(len, parts.values())) == len(states)
    assert all(parts[name] for name in parts)
