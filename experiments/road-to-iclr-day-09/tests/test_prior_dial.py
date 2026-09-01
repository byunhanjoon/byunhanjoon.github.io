from collections import Counter

import numpy as np
import pytest

from src.priors import PriorDial, balanced_coupling_schedule, population_coupling_mi


def test_schedule_has_fixed_marginals_and_deterministic_seed():
    a = balanced_coupling_schedule(120, 0.75, np.random.default_rng(8))
    b = balanced_coupling_schedule(120, 0.75, np.random.default_rng(8))
    assert a == b
    assert len(set(Counter(x[0] for x in a).values())) == 1
    assert len(set(Counter(x[1] for x in a).values())) == 1
    independent = balanced_coupling_schedule(120, 0.0, np.random.default_rng(9))
    joint_counts = Counter((x[0], x[1]) for x in independent)
    assert max(joint_counts.values()) - min(joint_counts.values()) <= 1


def test_population_coupling_information_endpoints_and_monotonicity():
    grid = np.linspace(0.0, 1.0, 101)
    information = np.asarray([population_coupling_mi(rho, 6) for rho in grid])
    assert population_coupling_mi(0.0, 6) == pytest.approx(0.0, abs=1e-14)
    assert population_coupling_mi(1.0, 6) == pytest.approx(np.log(6))
    assert np.all(np.diff(information) >= -1e-14)
    assert population_coupling_mi(0.5, 6) != pytest.approx(0.5 * np.log(6))


def test_population_coupling_information_validates_arguments():
    with pytest.raises(ValueError):
        population_coupling_mi(-0.1)
    with pytest.raises(ValueError):
        population_coupling_mi(0.5, 1)


def test_episode_determinism_alignment_and_no_query_label_feedback():
    kwargs = dict(seed=22, n_context=32, n_query=16, n_features=4, task_type="classification")
    a = PriorDial(**kwargs).generate("linear", "pwl", coupled=True)
    b = PriorDial(**kwargs).generate("linear", "pwl", coupled=True)
    np.testing.assert_array_equal(a.context_x, b.context_x)
    np.testing.assert_array_equal(a.context_y, b.context_y)
    np.testing.assert_array_equal(a.query_x, b.query_x)
    np.testing.assert_array_equal(a.query_y, b.query_y)
    assert a.context_x.shape[0] == a.context_y.shape[0]
    assert a.query_x.shape[0] == a.query_y.shape[0]
    assert "query_y" not in repr(a.metadata["transform_states"])
