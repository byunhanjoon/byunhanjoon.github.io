import numpy as np
import torch

from experiments.day3.orbit_ensemble import (
    OrbitDenseStemTabM,
    natural_orbits,
    random_orthogonal_orbits,
)


def test_natural_orbits_apply_the_declared_exact_map():
    transform = np.array([[1.0, 0.5], [0.0, 2.0]])
    orbits = natural_orbits(transform, ["cumulative", "local"])
    x = np.array([[2.0, 3.0]])
    actual = np.einsum("bd,kde->bke", x, orbits)
    assert np.allclose(actual[:, 0], x)
    assert np.allclose(actual[:, 1], x @ transform)


def test_random_orbits_are_full_rank_and_energy_preserving():
    orbits = random_orthogonal_orbits(7, 8, 123)
    identity = np.eye(7)
    assert np.allclose(orbits[0], identity)
    for transform in orbits:
        assert np.linalg.matrix_rank(transform) == 7
        assert np.allclose(transform.T @ transform, identity, atol=1e-12)


def test_repeated_member_view_matches_ordinary_tabm_forward():
    torch.manual_seed(0)
    transforms = np.repeat(np.eye(5)[None], 4, axis=0)
    model = OrbitDenseStemTabM(
        5,
        2,
        transforms,
        member_views=True,
        latent_size=8,
        n_blocks=1,
        d_block=16,
        dropout=0.0,
    ).eval()
    x = torch.randn(11, 5)
    with torch.inference_mode():
        ordinary = model.forward_members_ordinary(x)
        repeated = model.forward_members(x)
    assert torch.allclose(ordinary, repeated, atol=1e-6, rtol=1e-6)


def test_member_views_do_not_change_trainable_parameter_count():
    transforms = np.repeat(np.eye(5)[None], 4, axis=0)
    ordinary = OrbitDenseStemTabM(5, 1, transforms, member_views=False)
    orbit = OrbitDenseStemTabM(5, 1, transforms, member_views=True)
    ordinary_count = sum(parameter.numel() for parameter in ordinary.parameters())
    orbit_count = sum(parameter.numel() for parameter in orbit.parameters())
    assert ordinary_count == orbit_count
