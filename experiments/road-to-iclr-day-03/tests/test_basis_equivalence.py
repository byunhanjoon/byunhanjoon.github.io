import numpy as np

from experiments.day3.core import equivalence_diagnostics, exact_state_ple_and_identity, condition_transform, geometry


def test_condition_transform_is_invertible_scale_controlled():
    for kappa in (1, 10, 1000):
        transform = condition_transform(11, kappa, 7)
        singular = np.linalg.svd(transform, compute_uv=False)
        assert np.isclose(singular[0] / singular[-1], kappa, rtol=1e-10)
        assert np.isclose(np.exp(np.log(singular).mean()), 1.0, atol=1e-12)


def test_geometry_ignores_exact_redundancy():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 4))
    redundant = np.column_stack((x, x[:, 0] + x[:, 1]))
    assert geometry(redundant)["rank"] == 4
    assert geometry(redundant)["min_nonzero_variance"] > 0


def test_state_local_affine_extension_stays_equivalent_on_unseen_values():
    train = np.asarray([[0.0], [1.0], [3.0], [7.0]] * 20)
    parts = {
        "train": train,
        "val": np.asarray([[0.5], [2.0], [5.0]]),
        "test": np.asarray([[-1.0], [8.0], [4.0]]),
    }
    ple, identity, _ = exact_state_ple_and_identity(parts, 0)
    diagnostics = equivalence_diagnostics(ple, identity)
    assert diagnostics["a_to_b"]["val"] < 1e-12
    assert diagnostics["b_to_a"]["test"] < 1e-12
