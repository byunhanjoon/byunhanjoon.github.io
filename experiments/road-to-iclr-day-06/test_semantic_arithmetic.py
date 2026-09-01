from __future__ import annotations

import json
import copy
import itertools
from pathlib import Path

import numpy as np
import torch

import semantic_arithmetic as saa
import precision_delay as h2
import analyze_cross_perturbation as h5
import analyze_semantic_shadow as h4
import audit_day6_integrity as integrity
import audit_analysis_reproducibility as analysis_reproducibility


def test_exact_accum_linear_has_float32_parameters_and_outputs() -> None:
    layer = saa.ExactAccumLinear(7, 5)
    value = torch.randn(3, 7)
    output = layer(value)
    assert layer.weight.dtype == torch.float32
    assert output.dtype == torch.float32


def test_coordinate_conjugacy_in_real_arithmetic() -> None:
    rng = np.random.default_rng(17)
    x = rng.normal(size=(13, 11))
    w = rng.normal(size=(7, 11))
    permutation = rng.permutation(11)
    np.testing.assert_allclose(
        x @ w.T,
        x[:, permutation] @ w[:, permutation].T,
        rtol=1e-14, atol=1e-14,
    )


def test_frozen_matrix_is_three_by_three() -> None:
    config = json.loads((Path(__file__).parent / "hypothesis_01_config.json").read_text())
    assert len(config["datasets"]) == 3
    assert len(config["models"]) == 3
    assert set(config["models"]) == {"mlp", "resnet", "ft_transformer"}


def test_precision_interface_preserves_float32_state() -> None:
    layer = h2.InterfaceLinear(9, 4, accumulation_dtype=torch.bfloat16)
    output = layer(torch.randn(5, 9))
    assert layer.weight.dtype == torch.float32
    assert output.dtype == torch.float32


def test_pairwise_seed_fragility_averages_all_unordered_pairs() -> None:
    predictions = [
        np.asarray([0.0, 0.0]),
        np.asarray([1.0, 1.0]),
        np.asarray([3.0, 3.0]),
    ]
    # Pair MSEs are 1, 9, and 4.
    assert h5.mean_pairwise_mse(predictions) == 14.0 / 3.0


def test_h5_top_quartile_ties_use_average_rank() -> None:
    values = h5.pd.Series([9.0, 8.0, 7.0, 7.0, 6.0])
    assert h5.top_quartile_labels(values).tolist() == [1, 1, 0, 0, 0]


def test_constant_negative_control_has_zero_rank_association() -> None:
    constant = np.ones(12)
    target = np.arange(12)
    assert h4.safe_spearman(constant, target) == 0.0
    assert h5.safe_spearman(constant, target) == 0.0


def test_h4_h5_frozen_matrix_has_324_bundles() -> None:
    config = json.loads((Path(__file__).parent / "hypothesis_04_config.json").read_text())
    count = (
        len(config["datasets"]) * len(config["models"]) * len(config["seeds"])
        * len(config["learning_rates"]) * len(config["weight_decays"])
        * len(config["batch_sizes"])
    )
    assert count == 324


def test_h9_split_leaves_25_bundles_and_75_pairs() -> None:
    h3 = json.loads((Path(__file__).parent / "hypothesis_03_config.json").read_text())
    h9 = json.loads((Path(__file__).parent / "hypothesis_09_config.json").read_text())
    total = len(h3["datasets"]) * len(h3["models"]) * len(h3["seeds"])
    assert len(h9["development_stems"]) == 11
    assert total - len(h9["development_stems"]) == 25
    assert (total - len(h9["development_stems"])) * h3["nonidentity_views"] == 75


def test_h4_shards_mix_every_optimizer_axis() -> None:
    config = json.loads((Path(__file__).parent / "hypothesis_04_config.json").read_text())
    axes = list(itertools.product(
        enumerate(config["batch_sizes"]), enumerate(config["weight_decays"]),
        enumerate(config["learning_rates"]),
    ))
    shards = tuple([
        (batch, weight_decay, learning_rate)
        for (batch_index, batch), (wd_index, weight_decay), (lr_index, learning_rate) in axes
        if (batch_index + wd_index + lr_index) % 2 == shard_index
    ] for shard_index in (0, 1))
    for shard in shards:
        assert len(shard) == 6
        assert [row[0] for row in shard].count(config["batch_sizes"][0]) == 3
        assert [row[1] for row in shard].count(config["weight_decays"][0]) == 3
        assert [row[2] for row in shard].count(config["learning_rates"][0]) == 2


def test_day6_integrity_specs_cover_all_bundle_families() -> None:
    assert set(integrity.SPECS) == {"h1", "h2", "h3", "h4"}


def test_analysis_reproducibility_covers_all_final_analyzers() -> None:
    assert [row[0] for row in analysis_reproducibility.SPECS] == [
        "h3", "h3_dynamics", "h4", "h5", "h6", "h7", "h8", "h9",
    ]
    assert sum(len(row[3]) for row in analysis_reproducibility.SPECS) == 28


def test_iea64_one_update_conjugates_parameters_and_adam_state() -> None:
    config = json.loads(saa.DAY5_CONFIG_PATH.read_text())
    device = torch.device("cpu")
    width, output_width, seed = 13, 2, 271
    coordinate_map = np.asarray([5, 1, 12, 0, 8, 3, 10, 2, 7, 11, 4, 9, 6])
    class_map = np.asarray([0, 1])
    generator = torch.Generator().manual_seed(991)
    canonical_x = torch.randn(19, width, generator=generator)
    transformed_x = canonical_x[:, coordinate_map]
    target = torch.randint(0, output_width, (19,), generator=generator)

    for model_name in ("mlp", "resnet", "ft_transformer"):
        canonical = saa.initialize_model(
            model_name, width, output_width, seed, config, device, "iea64"
        )
        transformed = saa.initialize_model(
            model_name, width, output_width, seed, config, device, "iea64"
        )
        transformed.load_state_dict(saa.completion.matched_state(
            model_name, copy.deepcopy(canonical.state_dict()),
            coordinate_map, class_map,
        ))
        optimizers = [
            torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            for model in (canonical, transformed)
        ]
        for model, optimizer, value in zip(
            (canonical, transformed), optimizers, (canonical_x, transformed_x)
        ):
            torch.manual_seed(1771)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = saa.completion.loss_value(
                saa.completion.forward(model, value, model_name),
                target, "classification", model_name,
            )
            loss.backward()
            optimizer.step()

        expected = saa.completion.matched_state(
            model_name, canonical.state_dict(), coordinate_map, class_map
        )
        for key, value in transformed.state_dict().items():
            assert torch.equal(value, expected[key]), (model_name, key)

        canonical_parameters = dict(canonical.named_parameters())
        transformed_parameters = dict(transformed.named_parameters())
        for name, canonical_parameter in canonical_parameters.items():
            left = optimizers[0].state[canonical_parameter]
            right = optimizers[1].state[transformed_parameters[name]]
            for state_name in ("step", "exp_avg", "exp_avg_sq"):
                expected_state = left[state_name]
                if name == "first.weight" and expected_state.ndim == 2:
                    expected_state = expected_state[:, coordinate_map]
                assert torch.equal(right[state_name], expected_state), (
                    model_name, name, state_name,
                )


def test_exact_canonical_gather_is_bitwise_closed() -> None:
    generator = np.random.default_rng(817)
    canonical = generator.normal(size=(31, 17)).astype(np.float32)
    coordinate_map = generator.permutation(canonical.shape[1])
    transformed = canonical[:, coordinate_map]
    recovered = transformed[:, np.argsort(coordinate_map)]
    assert np.array_equal(recovered.view(np.uint32), canonical.view(np.uint32))


def test_h7_freeze_excludes_five_bundles_and_tests_93_pairs() -> None:
    config = json.loads((Path(__file__).parent / "hypothesis_07_config.json").read_text())
    assert len(config["development_stems"]) == 5
    assert (36 - len(config["development_stems"])) * 3 == 93


def test_h8_freeze_excludes_seven_bundles_and_tests_29() -> None:
    config = json.loads((Path(__file__).parent / "hypothesis_08_config.json").read_text())
    assert len(config["development_stems"]) == 7
    assert 36 - len(config["development_stems"]) == 29
    assert config["early_checkpoints"] == [5, 10, 20]
    assert config["level_log_threshold"] == -5.0
    assert config["acceleration_threshold"] == 0.02
