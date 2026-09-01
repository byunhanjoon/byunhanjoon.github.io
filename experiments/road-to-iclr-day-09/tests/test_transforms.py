import numpy as np
import pytest

from src.transforms import audit_transform, make_warp


@pytest.mark.parametrize("name", ["identity", "affine", "signed_power", "asinh", "pwl", "sinh"])
def test_transform_contract(name):
    rng = np.random.default_rng(4)
    context = np.r_[np.linspace(-3, 3, 101), 0.0, 0.0]
    transform = make_warp(name, rng).fit(context)
    audit = audit_transform(transform, context)
    assert audit["strictly_increasing"]
    assert audit["inverse_max_scaled_error"] < 1e-10
    assert audit["ties_preserved"]
    assert audit["all_finite"]
    assert transform.state_dict()["name"] == name

