import numpy as np
from sklearn.preprocessing import StandardScaler


def test_context_rescaling_cancels_outer_affine_scaling():
    rng = np.random.default_rng(71)
    outer_train = rng.normal(loc=[3.0, -4.0, 10.0], scale=[2.0, 5.0, 0.5], size=(300, 3))
    context = outer_train[:96]
    query = rng.normal(loc=[3.0, -4.0, 10.0], scale=[2.0, 5.0, 0.5], size=(64, 3))

    direct = StandardScaler().fit(context)
    expected_context = direct.transform(context)
    expected_query = direct.transform(query)

    outer = StandardScaler().fit(outer_train)
    outer_context = outer.transform(context)
    outer_query = outer.transform(query)
    episode = StandardScaler().fit(outer_context)

    np.testing.assert_allclose(episode.transform(outer_context), expected_context, atol=1e-12)
    np.testing.assert_allclose(episode.transform(outer_query), expected_query, atol=1e-12)


def test_context_target_rescaling_cancels_outer_standardization():
    rng = np.random.default_rng(73)
    outer_target = rng.normal(loc=20.0, scale=7.0, size=400)
    context = outer_target[:96]
    query = outer_target[200:264]

    outer_scaled_context = (context - outer_target.mean()) / outer_target.std()
    outer_scaled_query = (query - outer_target.mean()) / outer_target.std()
    episode_mean = outer_scaled_context.mean()
    episode_scale = outer_scaled_context.std()

    actual_context = (outer_scaled_context - episode_mean) / episode_scale
    actual_query = (outer_scaled_query - episode_mean) / episode_scale
    expected_context = (context - context.mean()) / context.std()
    expected_query = (query - context.mean()) / context.std()

    np.testing.assert_allclose(actual_context, expected_context, atol=1e-12)
    np.testing.assert_allclose(actual_query, expected_query, atol=1e-12)
