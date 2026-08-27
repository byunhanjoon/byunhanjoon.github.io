"""Algebraic tests for the Day 4 chart-covariance claims."""

from __future__ import annotations

import numpy as np

from residual_gain_certificate import (
    certify_squared_loss_gains,
    select_certified_intervention,
)
from residual_riesz_pilot import field_forms
from residual_spectral_profile import energy_curve
from spatial_product_riesz_pilot import (
    exact_isospectral_operator,
    generalized_median_scale,
    product_block,
    support_graph_stiffness,
)
from support_heat_pilot import count_spike_statistics


def random_invertible(rng: np.random.Generator, dimension: int) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(dimension, dimension)))
    scales = np.linspace(0.6, 1.8, dimension)
    return q @ np.diag(scales) @ q.T


def test_support_spike_allocator_uses_individual_fixed_threshold() -> None:
    base_counts = np.full(9, 10)

    one_count = base_counts.copy()
    one_count[4] = 11
    values = np.repeat(np.arange(9), one_count)
    _, _, excess, retained = count_spike_statistics(values)
    assert excess[4] == 1
    assert not retained.any()

    threshold_clearing = base_counts.copy()
    threshold_clearing[4] = 12
    values = np.repeat(np.arange(9), threshold_clearing)
    _, _, excess, retained = count_spike_statistics(values)
    assert excess[4] == 2
    assert retained.sum() == 1
    assert retained[4]


def test_first_layer_penalty_and_metric_step_are_chart_covariant() -> None:
    rng = np.random.default_rng(20260827)
    dimension, output = 7, 4
    phi = rng.normal(size=dimension)
    raw = rng.normal(size=(dimension, dimension))
    operator = raw @ raw.T + 0.3 * np.eye(dimension)
    weight = rng.normal(size=(output, dimension))
    gradient = rng.normal(size=(output, dimension))
    chart = random_invertible(rng, dimension)

    phi_prime = chart @ phi
    operator_prime = chart @ operator @ chart.T
    weight_prime = weight @ np.linalg.inv(chart)
    gradient_prime = gradient @ chart.T

    np.testing.assert_allclose(weight_prime @ phi_prime, weight @ phi)
    np.testing.assert_allclose(
        np.trace(weight_prime @ operator_prime @ weight_prime.T),
        np.trace(weight @ operator @ weight.T),
    )

    rate = 0.013
    update = weight - rate * gradient @ np.linalg.inv(operator)
    update_prime = weight_prime - rate * gradient_prime @ np.linalg.inv(operator_prime)
    np.testing.assert_allclose(
        update_prime, update @ np.linalg.inv(chart), rtol=1e-11, atol=1e-11
    )


def test_riesz_renderings_differ_only_by_an_orthogonal_map() -> None:
    rng = np.random.default_rng(20260911)
    dimension = 7
    raw = rng.normal(size=(dimension, dimension))
    operator = raw @ raw.T + 0.4 * np.eye(dimension)
    chart = random_invertible(rng, dimension)
    operator_prime = chart @ operator @ chart.T
    phi = rng.normal(size=dimension)

    def symmetric_power(matrix: np.ndarray, exponent: float) -> np.ndarray:
        values, vectors = np.linalg.eigh(matrix)
        return (vectors * values[None, :] ** exponent) @ vectors.T

    inverse_root = symmetric_power(operator, -0.5)
    root = symmetric_power(operator, 0.5)
    inverse_root_prime = symmetric_power(operator_prime, -0.5)
    rotation = inverse_root_prime @ chart @ root
    rendering = inverse_root @ phi
    rendering_prime = inverse_root_prime @ (chart @ phi)

    np.testing.assert_allclose(rotation.T @ rotation, np.eye(dimension), atol=1e-10)
    np.testing.assert_allclose(rendering_prime, rotation @ rendering, atol=1e-10)
    np.testing.assert_allclose(
        rendering_prime @ rendering_prime, rendering @ rendering, atol=1e-10
    )


def test_residual_riesz_representer_is_chart_invariant() -> None:
    rng = np.random.default_rng(20260828)
    dimension = 9
    phi = rng.normal(size=dimension)
    raw = rng.normal(size=(dimension, dimension))
    operator = raw @ raw.T + 0.5 * np.eye(dimension)
    covector = rng.normal(size=dimension)
    chart = random_invertible(rng, dimension)

    value = covector @ np.linalg.solve(operator, phi)
    phi_prime = chart @ phi
    covector_prime = chart @ covector
    operator_prime = chart @ operator @ chart.T
    value_prime = covector_prime @ np.linalg.solve(operator_prime, phi_prime)

    np.testing.assert_allclose(value_prime, value, rtol=1e-11, atol=1e-11)
    energy = covector @ np.linalg.solve(operator, covector)
    energy_prime = covector_prime @ np.linalg.solve(
        operator_prime, covector_prime
    )
    np.testing.assert_allclose(energy_prime, energy, rtol=1e-11, atol=1e-11)


def test_residual_gain_certificate_selects_only_positive_lcb() -> None:
    residual = np.tile(np.array([-1.0, 1.0]), 5000)
    interventions = {
        "aligned": 0.5 * residual,
        "opposed": -0.5 * residual,
    }
    certificates = certify_squared_loss_gains(
        residual,
        interventions,
        residual_bound=1.0,
        intervention_bound=0.5,
        delta=0.05,
    )
    assert select_certified_intervention(certificates) == "aligned"

    null = certify_squared_loss_gains(
        residual,
        {"zero": np.zeros_like(residual)},
        residual_bound=1.0,
        intervention_bound=0.5,
        delta=0.05,
    )
    assert select_certified_intervention(null) is None


def test_isospectral_control_preserves_generalized_spectrum() -> None:
    rng = np.random.default_rng(20260829)
    phi = rng.normal(size=(600, 11))
    phi -= phi.mean(axis=0)
    nodes = np.cumsum(rng.uniform(0.2, 1.4, size=11))
    mass, correct, _, isospectral = field_forms(
        phi, nodes, column=3, strength=0.7, wrong_permutation_seed=91
    )
    values, vectors = np.linalg.eigh(mass)
    keep = values > values[-1] * 1e-10
    whitener = vectors[:, keep] / np.sqrt(values[keep])[None, :]

    correct_spectrum = np.linalg.eigvalsh(
        whitener.T @ (correct - mass) @ whitener
    )
    control_spectrum = np.linalg.eigvalsh(
        whitener.T @ (isospectral - mass) @ whitener
    )
    np.testing.assert_allclose(
        control_spectrum, correct_spectrum, rtol=1e-9, atol=1e-10
    )


def test_spectral_energy_curve_matches_direct_solve_and_decreases() -> None:
    rng = np.random.default_rng(20260830)
    raw_mass = rng.normal(size=(8, 8))
    mass = raw_mass @ raw_mass.T + 0.4 * np.eye(8)
    raw_stiffness = rng.normal(size=(8, 8))
    stiffness = raw_stiffness @ raw_stiffness.T
    covector = rng.normal(size=8)
    strengths = np.array([0.01, 0.1, 1.0, 10.0])

    curve = energy_curve(mass, stiffness, covector, strengths)
    direct = np.array(
        [
            covector @ np.linalg.solve(mass + tau * stiffness, covector)
            for tau in strengths
        ]
    )
    np.testing.assert_allclose(curve, direct, rtol=1e-10, atol=1e-11)
    assert np.all(np.diff(curve) <= 0)


def test_haar_isospectral_retention_has_claimed_moments() -> None:
    rng = np.random.default_rng(20260831)
    attenuation = np.array([1.0, 0.9, 0.7, 0.4, 0.2, 0.08, 0.01])
    dimension = len(attenuation)
    sphere = rng.normal(size=(200_000, dimension))
    sphere /= np.linalg.norm(sphere, axis=1, keepdims=True)
    retention = (sphere * sphere) @ attenuation

    expected_mean = attenuation.mean()
    expected_variance = 2.0 / (dimension * (dimension + 2)) * (
        np.sum(attenuation * attenuation)
        - np.sum(attenuation) ** 2 / dimension
    )
    np.testing.assert_allclose(retention.mean(), expected_mean, rtol=3e-3)
    np.testing.assert_allclose(retention.var(), expected_variance, rtol=1e-2)


def test_product_representer_is_covariant_to_factorwise_charts() -> None:
    rng = np.random.default_rng(20260901)
    d1, d2 = 4, 3
    dimension = d1 * d2
    phi = rng.normal(size=dimension)
    raw_mass = rng.normal(size=(dimension, dimension))
    mass = raw_mass @ raw_mass.T + 0.5 * np.eye(dimension)
    raw_stiffness = rng.normal(size=(dimension, dimension))
    stiffness = raw_stiffness @ raw_stiffness.T
    covector = rng.normal(size=dimension)
    chart_1 = random_invertible(rng, d1)
    chart_2 = random_invertible(rng, d2)
    chart = np.kron(chart_1, chart_2)

    operator = mass + 0.7 * stiffness
    value = covector @ np.linalg.solve(operator, phi)
    energy = covector @ np.linalg.solve(operator, covector)
    phi_prime = chart @ phi
    covector_prime = chart @ covector
    operator_prime = chart @ operator @ chart.T

    np.testing.assert_allclose(
        covector_prime @ np.linalg.solve(operator_prime, phi_prime),
        value,
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        covector_prime @ np.linalg.solve(operator_prime, covector_prime),
        energy,
        rtol=1e-10,
        atol=1e-10,
    )


def test_product_isospectral_control_preserves_generalized_spectrum() -> None:
    rng = np.random.default_rng(20260902)
    dimension = 13
    raw_mass = rng.normal(size=(dimension, dimension))
    mass = raw_mass @ raw_mass.T + 0.2 * np.eye(dimension)
    raw_stiffness = rng.normal(size=(dimension, dimension))
    stiffness = raw_stiffness @ raw_stiffness.T
    strength = 0.6
    control = exact_isospectral_operator(mass, stiffness, strength, seed=927)

    values, vectors = np.linalg.eigh(mass)
    whitener = vectors / np.sqrt(values)[None, :]
    semantic_spectrum = np.linalg.eigvalsh(whitener.T @ stiffness @ whitener)
    control_spectrum = np.linalg.eigvalsh(
        whitener.T @ ((control - mass) / strength) @ whitener
    )
    np.testing.assert_allclose(
        control_spectrum, semantic_spectrum, rtol=1e-9, atol=1e-9
    )


def test_empirical_anova_product_is_orthogonal_to_marginal_spaces() -> None:
    rng = np.random.default_rng(20260903)
    first = rng.normal(size=900)
    second = 0.8 * first + 0.4 * rng.normal(size=900)
    train = np.column_stack([first, second])
    clean = {"train": train, "val": train[:100], "test": train[100:200]}
    product, nodes, reference_mass = product_block(
        clean, (0, 1), bins=9, interaction_projection="empirical-anova"
    )

    from support_heat_pilot import hat_basis

    first_basis = hat_basis(first, nodes[0])
    second_basis = hat_basis(second, nodes[1])
    first_basis -= first_basis.mean(axis=0)
    second_basis -= second_basis.mean(axis=0)
    additive = np.column_stack([np.ones(len(first)), first_basis, second_basis])
    cross_moment = additive.T @ product["train"] / len(first)
    np.testing.assert_allclose(cross_moment, 0.0, atol=2e-10)
    assert np.linalg.eigvalsh(reference_mass).min() > -1e-10


def test_reference_mass_completion_yields_a_full_function_space_control() -> None:
    rng = np.random.default_rng(20260907)
    empirical_rows = rng.normal(size=(18, 31))
    empirical_mass = empirical_rows.T @ empirical_rows / len(empirical_rows)
    reference_rows = rng.normal(size=(50, 31))
    reference_mass = reference_rows.T @ reference_rows / len(reference_rows)
    completed_mass = 0.99 * empirical_mass + 0.01 * reference_mass
    raw_stiffness = rng.normal(size=(31, 31))
    stiffness = raw_stiffness @ raw_stiffness.T
    strength = 0.8
    control = exact_isospectral_operator(
        completed_mass, stiffness, strength, seed=20260908
    )

    values, vectors = np.linalg.eigh(completed_mass)
    assert values.min() > 0
    whitener = vectors / np.sqrt(values)[None, :]
    semantic = np.linalg.eigvalsh(whitener.T @ stiffness @ whitener)
    randomized = np.linalg.eigvalsh(
        whitener.T @ ((control - completed_mass) / strength) @ whitener
    )
    np.testing.assert_allclose(randomized, semantic, rtol=1e-8, atol=1e-8)


def test_reference_mass_null_space_is_the_intersection_and_is_covariant() -> None:
    rng = np.random.default_rng(20260909)
    orthogonal, _ = np.linalg.qr(rng.normal(size=(7, 7)))
    empirical = orthogonal @ np.diag([3, 2, 1, 0, 0, 0, 0]) @ orthogonal.T
    reference = orthogonal @ np.diag([0, 2, 1, 4, 3, 2, 0]) @ orthogonal.T
    completed = 0.9 * empirical + 0.1 * reference
    assert np.linalg.matrix_rank(empirical, tol=1e-10) == 3
    assert np.linalg.matrix_rank(reference, tol=1e-10) == 5
    assert np.linalg.matrix_rank(completed, tol=1e-10) == 6
    np.testing.assert_allclose(completed @ orthogonal[:, 6], 0.0, atol=1e-12)

    chart = random_invertible(rng, 7)
    completed_prime = chart @ completed @ chart.T
    assert np.linalg.matrix_rank(completed_prime, tol=1e-10) == 6


def test_support_graph_form_is_psd_and_jointly_calibratable() -> None:
    rng = np.random.default_rng(20260904)
    coordinates = np.column_stack(
        [37.5 + 0.2 * rng.normal(size=500), -122.2 + 0.3 * rng.normal(size=500)]
    )
    phi = rng.normal(size=(500, 17))
    phi -= phi.mean(axis=0)
    mass = phi.T @ phi / len(phi)
    stiffness, metadata = support_graph_stiffness(phi, coordinates, neighbors=9)

    assert metadata["knn_neighbors"] == 9
    assert np.linalg.eigvalsh(stiffness).min() > -1e-10
    scale = generalized_median_scale(mass, stiffness)
    np.testing.assert_allclose(
        generalized_median_scale(mass, stiffness / scale), 1.0, rtol=1e-9
    )


def test_empirical_interaction_purification_commutes_with_chart_changes() -> None:
    rng = np.random.default_rng(20260905)
    samples, product_dimension, additive_dimension = 300, 12, 7
    product = rng.normal(size=(samples, product_dimension))
    additive_base = rng.normal(size=(samples, additive_dimension - 2))
    additive = np.column_stack(
        [
            additive_base,
            additive_base[:, 0] + additive_base[:, 1],
            2.0 * additive_base[:, 2],
        ]
    )
    assert np.linalg.matrix_rank(additive) == additive_dimension - 2
    product_chart = random_invertible(rng, product_dimension)
    additive_chart = random_invertible(rng, additive_dimension)

    purified = product - additive @ (np.linalg.pinv(additive) @ product)
    product_prime = product @ product_chart.T
    additive_prime = additive @ additive_chart.T
    purified_prime = product_prime - additive_prime @ (
        np.linalg.pinv(additive_prime) @ product_prime
    )

    np.testing.assert_allclose(
        purified_prime, purified @ product_chart.T, rtol=1e-10, atol=1e-10
    )
    np.testing.assert_allclose(additive.T @ purified, 0.0, atol=1e-11)


def test_singular_representer_is_chart_invariant_on_operator_range() -> None:
    rng = np.random.default_rng(20260906)
    dimension, rank = 10, 6
    factor = rng.normal(size=(dimension, rank))
    operator = factor @ factor.T
    covector = operator @ rng.normal(size=dimension)
    evaluation = operator @ rng.normal(size=dimension)
    chart = random_invertible(rng, dimension)

    value = covector @ np.linalg.pinv(operator) @ evaluation
    operator_prime = chart @ operator @ chart.T
    value_prime = (
        (chart @ covector)
        @ np.linalg.pinv(operator_prime)
        @ (chart @ evaluation)
    )
    np.testing.assert_allclose(value_prime, value, rtol=1e-9, atol=1e-9)


def test_reference_completion_quotient_quantities_are_chart_invariant() -> None:
    rng = np.random.default_rng(20260910)
    dimension = 8
    orthogonal, _ = np.linalg.qr(rng.normal(size=(dimension, dimension)))
    mass = orthogonal @ np.diag([4.0, 3.0, 2.2, 1.7, 1.1, 0.8, 0.3, 0.0]) @ orthogonal.T
    stiffness = orthogonal @ np.diag([0.1, 0.4, 0.8, 1.3, 2.0, 3.0, 5.0, 0.0]) @ orthogonal.T
    chart = random_invertible(rng, dimension)
    mass_prime = chart @ mass @ chart.T
    stiffness_prime = chart @ stiffness @ chart.T

    null = orthogonal[:, -1]
    null_prime = np.linalg.solve(chart.T, null)
    np.testing.assert_allclose(mass_prime @ null_prime, 0.0, atol=1e-11)
    np.testing.assert_allclose(stiffness_prime @ null_prime, 0.0, atol=1e-11)

    covector = mass @ rng.normal(size=dimension)
    evaluation = mass @ rng.normal(size=dimension)
    strength = 0.9
    operator = mass + strength * stiffness
    operator_prime = mass_prime + strength * stiffness_prime
    value = covector @ np.linalg.pinv(operator) @ evaluation
    value_prime = (
        (chart @ covector)
        @ np.linalg.pinv(operator_prime)
        @ (chart @ evaluation)
    )
    np.testing.assert_allclose(value_prime, value, rtol=1e-9, atol=1e-9)

    def finite_generalized_spectrum(
        local_mass: np.ndarray, local_stiffness: np.ndarray
    ) -> np.ndarray:
        values, vectors = np.linalg.eigh(local_mass)
        keep = values > values[-1] * 1e-10
        whitener = vectors[:, keep] / np.sqrt(values[keep])[None, :]
        return np.linalg.eigvalsh(whitener.T @ local_stiffness @ whitener)

    np.testing.assert_allclose(
        finite_generalized_spectrum(mass_prime, stiffness_prime),
        finite_generalized_spectrum(mass, stiffness),
        rtol=1e-9,
        atol=1e-9,
    )
