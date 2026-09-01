from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.transforms import (
    AsinhTransform,
    AtomicSpacingTransform,
    CategoricalBijectionTransform,
    ComposedTransform,
    EmpiricalCDFTransform,
    IdentityTransform,
    MonotoneSplineTransform,
    NegativeAffineTransform,
    PositiveAffineTransform,
    QuantileGaussianTransform,
    RandomMonotonePWLTransform,
    SignedPowerTransform,
    transform_from_state,
)


@pytest.fixture
def arrays():
    rng = np.random.default_rng(20260831)
    train = rng.normal(size=(257, 6))
    test = rng.normal(size=(113, 6))
    train[::31, 2] = np.nan
    test[::17, 2] = np.nan
    return train, test


@pytest.mark.parametrize(
    "transform",
    [
        IdentityTransform(),
        PositiveAffineTransform(1.0, 11),
        NegativeAffineTransform(1.0, 12),
        SignedPowerTransform(0.5),
        SignedPowerTransform(2.0),
        SignedPowerTransform(3.0),
        AsinhTransform(1.4),
        RandomMonotonePWLTransform(1.5, 13),
        MonotoneSplineTransform(1.5, 14),
        AtomicSpacingTransform(1.0, 15),
        ComposedTransform(
            [SignedPowerTransform(2.0), RandomMonotonePWLTransform(0.7, 16)]
        ),
    ],
)
def test_exact_transform_is_invertible_and_monotone(transform, arrays):
    train, test = arrays
    transform.fit(train)
    audit = transform.audit(train, test)
    assert audit["missing_mask_preserved"]
    assert audit["all_finite_inputs_have_finite_outputs"]
    assert audit["max_rel_reconstruction_error"] < 1e-10
    assert audit["strict_order_violations"] == 0


@pytest.mark.parametrize("transform", [EmpiricalCDFTransform(), QuantileGaussianTransform()])
def test_rank_baselines_are_labeled_lossy(transform, arrays):
    train, test = arrays
    transform.fit(train)
    assert transform.metadata.exactness_class == "order-preserving but lossy because of ties/finite precision"
    assert np.array_equal(np.isnan(test), np.isnan(transform.transform(test)))


def test_transform_state_round_trip_is_deterministic(arrays):
    train, test = arrays
    transform = MonotoneSplineTransform(1.25, 123).fit(train)
    state = json.loads(json.dumps(transform.state_dict()))
    restored = transform_from_state(state)
    np.testing.assert_array_equal(transform.transform(test), restored.transform(test))
    np.testing.assert_allclose(restored.inverse_transform(restored.transform(test)), test, rtol=1e-11, atol=1e-11)


def test_fit_does_not_depend_on_query_rows(arrays):
    train, test = arrays
    first = RandomMonotonePWLTransform(1.0, 99).fit(train)
    first.audit(train, test)
    second = RandomMonotonePWLTransform(1.0, 99).fit(train)
    second.audit(train, test * 1e6)
    assert first.state_dict() == second.state_dict()


def test_categorical_bijection_preserves_string_and_integer_membership():
    dtype_string = pd.CategoricalDtype(["red", "green", "blue", "query_only"])
    dtype_integer = pd.CategoricalDtype([10, 20, 30, 40], ordered=True)
    train = pd.DataFrame(
        {
            "string_category": pd.Series(["red", "green", None, "red"], dtype=dtype_string),
            "integer_category": pd.Series([10, 20, 20, None], dtype=dtype_integer),
            "numeric": [1.0, 2.0, 3.0, 4.0],
        }
    )
    query = pd.DataFrame(
        {
            "string_category": pd.Series(["query_only", "blue", None], dtype=dtype_string),
            "integer_category": pd.Series([40, 30, None], dtype=dtype_integer),
            "numeric": [5.0, 6.0, 7.0],
        }
    )
    transform = CategoricalBijectionTransform(seed=20260831).fit(
        train, ["string_category", "integer_category"]
    )
    audit = transform.audit(train, query)
    assert audit["equality_classes_preserved"]
    assert audit["missing_mask_preserved"]
    assert audit["exact_round_trip"]
    warped = transform.transform(query)
    assert warped["numeric"].equals(query["numeric"])
    assert isinstance(warped["string_category"].dtype, pd.CategoricalDtype)
    assert warped["integer_category"].cat.ordered


def test_categorical_bijection_state_round_trip():
    dtype = pd.CategoricalDtype(["a", "b", "c"])
    frame = pd.DataFrame({"category": pd.Series(["a", "c", None], dtype=dtype)})
    first = CategoricalBijectionTransform(seed=19).fit(frame, ["category"])
    state = json.loads(json.dumps(first.state_dict()))
    restored = CategoricalBijectionTransform.from_state(state)
    assert first.transform(frame).equals(restored.transform(frame))
