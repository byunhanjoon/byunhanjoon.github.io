from __future__ import annotations

import torch

from neural_benchmark import HPO_TRIALS, MetricTokenizedModel, make_model


def test_metric_tokenizer_transforms_only_trailing_partition_weights() -> None:
    model = make_model(
        "mlp", 37, HPO_TRIALS[0], tokenized_representation_size=32, token_dimension=16
    )
    assert isinstance(model, MetricTokenizedModel)
    x = torch.randn(7, 37)
    transformed = model.transform(x)
    assert transformed.shape == (7, 21)
    assert torch.equal(transformed[:, :5], x[:, :5])
    assert sum(parameter.numel() for parameter in model.tokenizer.parameters()) == 32 * 16


def test_direct_similarity_model_has_no_metric_tokenizer() -> None:
    model = make_model("mlp", 32, HPO_TRIALS[0])
    assert not isinstance(model, MetricTokenizedModel)
    assert model(torch.randn(4, 32)).shape == (4, 1)


def test_categorical_one_hot_can_use_frozen_width_lookup_projection() -> None:
    model = make_model(
        "mlp", 101, HPO_TRIALS[0], tokenized_representation_size=101, token_dimension=32
    )
    assert isinstance(model, MetricTokenizedModel)
    one_hot = torch.eye(101)[:4]
    transformed = model.transform(one_hot)
    assert transformed.shape == (4, 32)
    assert sum(parameter.numel() for parameter in model.tokenizer.parameters()) == 101 * 32
