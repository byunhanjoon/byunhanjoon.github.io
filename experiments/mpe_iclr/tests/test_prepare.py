from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mpe import validate_distance_matrix
from prepare_data import deterministic_take, hierarchy_geometry, string_jaccard_distance


def test_deterministic_take_does_not_depend_on_target():
    frame = pd.DataFrame({"row_id": [f"r{i}" for i in range(100)], "target": np.arange(100.0)})
    first = deterministic_take(frame, 17, "test")
    changed = frame.copy()
    changed["target"] = changed["target"].sample(frac=1.0, random_state=7).to_numpy()
    second = deterministic_take(changed, 17, "test")
    assert first.row_id.tolist() == second.row_id.tolist()


def test_prefix_hierarchy_is_metric_and_has_expected_sibling_distance():
    states = ["111011", "111012", "131011", "291111"]
    distance, paths, parents = hierarchy_geometry(states, [2, 3, 5, 6], "ROOT")
    validate_distance_matrix(distance)
    assert distance[0, 1] < distance[0, 2]
    assert distance[0, 2] == distance[0, 3]
    assert paths[states[0]][0] == "ROOT"
    assert parents[states[0]].startswith("L5:")


def test_character_trigram_jaccard_is_metric():
    distance = string_jaccard_distance(["data scientist", "data science", "bus driver", "teacher"])
    validate_distance_matrix(distance)
    assert distance[0, 1] < distance[0, 2]


def test_duplicate_string_states_have_zero_distance():
    distance = string_jaccard_distance(["Nurse", " nurse "])
    assert np.array_equal(distance, np.zeros((2, 2)))
