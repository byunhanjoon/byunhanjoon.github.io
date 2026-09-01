import numpy as np

from completion_row_order import ROW_ORDERS, row_permutations


def test_row_permutations_are_nested_reproducible_and_valid():
    first = row_permutations(23, 991)
    second = row_permutations(23, 991)
    assert len(first) == ROW_ORDERS
    assert np.array_equal(first[0], np.arange(23))
    for left, right in zip(first, second):
        assert np.array_equal(left, right)
        assert np.array_equal(np.sort(left), np.arange(23))
    assert len({tuple(value) for value in first}) == ROW_ORDERS
