import numpy as np

from analyze_anytime_nested_cover import PREFIXES, nested_base, nested_family
from analyze_strength2_cover import assert_strength


def test_literal_nested_prefixes_all_supported_shapes():
    for category in (1, 2, 4):
        for classes in (1, 2, 4):
            base = nested_base(category, classes)
            assert base.shape == (64, 4)
            for budget, strength in PREFIXES:
                assert_strength(base[:budget], (4, category, classes, 4), strength)


def test_randomization_keeps_one_shared_ordered_schedule():
    family = nested_family(4, 2)
    assert family.shape == (24 * 24 * 2 * 24, 64, 4)
    assert np.array_equal(family[:, :4], family[:, :16][:, :4])
    assert np.array_equal(family[:, :16], family[:, :64][:, :16])


def test_multiclass_randomization_and_prefixes():
    family = nested_family(4, 4)
    assert family.shape == (24 ** 4, 64, 4)
    for budget, strength in PREFIXES:
        assert_strength(family[0, :budget], (4, 4, 4, 4), strength)
