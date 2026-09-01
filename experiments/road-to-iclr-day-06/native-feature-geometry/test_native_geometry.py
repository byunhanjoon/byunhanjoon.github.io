import json
from pathlib import Path

import numpy as np
import torch

import native_geometry as ng


HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "pilot_config.json").read_text())


def test_native_gram_chart_equivariance() -> None:
    for domain in CONFIG["domains"]:
        embedding, gram = ng.native_embedding(domain, CONFIG["embedding_dim"])
        for chart in ng.charts(991 + ng.stable_offset(domain), 5):
            table = ng.code_table(embedding, chart)
            code_gram = table @ table.T
            expected = np.empty_like(code_gram)
            expected[np.ix_(chart, chart)] = gram
            assert np.max(np.abs(code_gram - expected)) <= 1e-10


def test_all_centered_modes_retained() -> None:
    for domain in CONFIG["domains"]:
        _, gram = ng.native_embedding(domain, CONFIG["embedding_dim"])
        assert np.linalg.matrix_rank(gram, tol=1e-9) == 15
        assert np.max(np.abs(gram.sum(axis=0))) < 1e-8


def test_chart_is_bijection() -> None:
    for chart in ng.charts(1234, 20):
        assert sorted(chart.tolist()) == list(range(16))


def test_transport_patch_specificity() -> None:
    seed = CONFIG["seeds"][0]
    data = ng.make_dataset("cycle16", "category_holdout", seed, CONFIG)
    native, _ = ng.native_embedding("cycle16", CONFIG["embedding_dim"])
    chart = ng.charts(seed, 2)[1]
    ng.seed_all(seed + 701)
    model = ng.FeatureMLP(
        "native_tuned",
        ng.code_table(native, chart),
        CONFIG["embedding_dim"],
        CONFIG["training"]["hidden_width"],
    )
    device = torch.device("cpu")
    model.to(device)
    original = ng.predict(model, data.category["test"], data.continuous["test"], chart, data, device)
    rows = chart[data.held]
    with torch.no_grad():
        model.embedding.weight[torch.as_tensor(rows)] += 1.0
    patched = ng.predict(model, data.category["test"], data.continuous["test"], chart, data, device)
    seen = np.isin(data.category["test"], data.seen)
    held = np.isin(data.category["test"], data.held)
    assert np.array_equal(original[seen], patched[seen])
    assert np.any(original[held] != patched[held])


def test_dataset_holdout_and_balance() -> None:
    for domain in CONFIG["domains"]:
        data = ng.make_dataset(domain, "category_holdout", CONFIG["seeds"][0], CONFIG)
        assert not np.isin(data.category["train"], data.held).any()
        counts = np.bincount(data.category["test"], minlength=16)
        assert counts.min() == counts.max() == 64

