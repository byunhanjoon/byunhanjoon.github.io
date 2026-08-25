import numpy as np

from experiments.day3.optimizer_remedies import anchor_canonicalize, fit_input_preconditioner


def test_anchor_canonicalization_is_invariant_to_general_linear_basis():
    rng = np.random.default_rng(123)
    train = rng.normal(size=(300, 12))
    transform = rng.normal(size=(12, 12))
    while np.linalg.cond(transform) > 100:
        transform = rng.normal(size=(12, 12))
    parts = {
        "train": train,
        "val": rng.normal(size=(60, 12)),
        "test": rng.normal(size=(70, 12)),
    }
    changed = {part: values @ transform for part, values in parts.items()}
    canonical, _ = anchor_canonicalize(parts)
    canonical_changed, _ = anchor_canonicalize(changed)
    for part in parts:
        assert np.allclose(canonical[part], canonical_changed[part], atol=1e-9)


def test_natural_first_layer_step_is_equivariant():
    rng = np.random.default_rng(321)
    x = rng.normal(size=(500, 9))
    transform = rng.normal(size=(9, 9))
    while np.linalg.cond(transform) > 30:
        transform = rng.normal(size=(9, 9))
    gradient = rng.normal(size=(7, 10))
    preconditioner, _ = fit_input_preconditioner(x, 1.0)
    # Augmented transform keeps the bias coordinate fixed.
    augmented_transform = np.eye(10)
    augmented_transform[:9, :9] = transform
    changed_x = x @ transform
    changed_gradient = gradient @ augmented_transform
    changed_preconditioner, _ = fit_input_preconditioner(changed_x, 1.0)
    reference_step = gradient @ preconditioner
    changed_step = changed_gradient @ changed_preconditioner
    expected = reference_step @ np.linalg.inv(augmented_transform).T
    assert np.allclose(changed_step, expected, rtol=1e-8, atol=1e-8)
