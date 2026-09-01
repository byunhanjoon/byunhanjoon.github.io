from pathlib import Path
import sys

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from representations import categorical_unknown_table, piecewise_linear_table, spectral_coordinates


def test_unknown_table_collapses_all_unseen_states():
    table = categorical_unknown_table(7, np.asarray([0, 2, 4]))
    assert table.shape == (7, 4)
    assert np.array_equal(table[1], table[3])
    assert np.array_equal(table[3], table[6])
    assert not np.array_equal(table[0], table[1])


def test_piecewise_linear_is_continuous_and_monotone_coordinatewise():
    values = np.linspace(0, 3, 31)
    table = piecewise_linear_table(values, np.asarray([0, 1, 2, 3]), width=3)
    assert table.shape == (31, 3)
    assert np.all(np.diff(table, axis=0) >= -1e-7)
    assert np.max(np.abs(table[10] - np.asarray([1, 0, 0]))) < 1e-6


def test_spectral_coordinates_have_requested_width_and_are_finite():
    graph = nx.path_graph(9)
    adjacency = nx.to_numpy_array(graph)
    result = spectral_coordinates(adjacency, 5)
    assert result.shape == (9, 5)
    assert np.isfinite(result).all()
