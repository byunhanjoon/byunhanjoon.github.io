import numpy as np

from src.representations import TieAwareECDF
from src.transforms import SignedPowerTransform


def test_rank_invariance_and_ties_under_increasing_map():
    context = np.array([[0.0], [1.0], [1.0], [3.0], [9.0]])
    query = np.array([[-1.0], [1.0], [2.0], [10.0]])
    transform = SignedPowerTransform(power=1.7)
    original = TieAwareECDF().fit(context).transform(query)
    mapped = TieAwareECDF().fit(transform.transform(context[:, 0])[:, None]).transform(
        transform.transform(query[:, 0])[:, None]
    )
    np.testing.assert_array_equal(original, mapped)
    assert original[1, 0] == 0.4


def test_fit_is_context_only():
    context = np.arange(8.0)[:, None]
    fitted = TieAwareECDF().fit(context)
    state_before = fitted.columns_[0].copy()
    fitted.transform(np.array([[1e100], [-1e100]]))
    np.testing.assert_array_equal(fitted.columns_[0], state_before)

