import importlib.util
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("mpb", HERE / "metric_partition_benchmark.py")
mpb = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = mpb
SPEC.loader.exec_module(mpb)


def test_tree_metric_is_symmetric_and_geodesic():
    d = mpb.tree_distance()
    assert np.allclose(d, d.T)
    assert np.allclose(np.diag(d), 0)
    assert d[15, 16] == 2
    assert d[15, 30] == 8


def test_every_feature_map_has_frozen_dimension_and_is_finite():
    for domain_name in mpb.DOMAINS:
        domain = mpb.make_domain(domain_name, 7)
        values = mpb.stored_values(domain, 2, 7)
        for method in mpb.METHODS:
            arrays = mpb.feature_map(domain, values, method, 7)
            for part in ("train", "val", "test"):
                assert arrays[part].shape == (len(domain.semantic[part]), 16)
                assert np.isfinite(arrays[part]).all()


def test_metric_partitions_sum_to_one():
    for domain_name in mpb.DOMAINS:
        domain = mpb.make_domain(domain_name, 8)
        values = mpb.stored_values(domain, 4, 8)
        for method in ("code_rbf", "mpe_native", "mmpe_native", "mpe_corrupt"):
            arrays = mpb.feature_map(domain, values, method, 8)
            assert np.allclose(arrays["train"].sum(axis=1), 1.0)


def test_native_maps_ignore_equivalent_storage_schema():
    for domain_name in mpb.DOMAINS:
        domain = mpb.make_domain(domain_name, 9)
        a = mpb.feature_map(domain, mpb.stored_values(domain, 0, 9), "mmpe_native", 9)
        b = mpb.feature_map(domain, mpb.stored_values(domain, 7, 9), "mmpe_native", 9)
        for part in ("train", "val", "test"):
            assert np.array_equal(a[part], b[part])


def test_nominal_corruption_is_a_true_negative_control():
    domain = mpb.make_domain("nominal", 10)
    native = mpb.native_features(domain, 10, False, False)
    corrupt = mpb.native_features(domain, 10, False, True)
    for part in ("train", "val", "test"):
        assert np.array_equal(native[part], corrupt[part])


def test_feature_construction_does_not_depend_on_targets():
    domain = mpb.make_domain("cycle", 11)
    values = mpb.stored_values(domain, 3, 11)
    before = mpb.feature_map(domain, values, "mmpe_native", 11)
    mutated = mpb.DomainData(domain.name, domain.semantic, {p: y[::-1] for p, y in domain.y.items()}, domain.distance, domain.support, domain.held_out)
    after = mpb.feature_map(mutated, values, "mmpe_native", 11)
    for part in ("train", "val", "test"):
        assert np.array_equal(before[part], after[part])


def test_local_ple_has_same_affine_span():
    import basis_control

    domain = mpb.make_domain("interval", 12)
    stored = mpb.stored_values(domain, 0, 12)
    cumulative = mpb.ple_fit_transform(stored["train"], stored)
    local = basis_control.local_basis(cumulative)
    p = np.column_stack([np.ones(len(cumulative["train"])), cumulative["train"]])
    h = np.column_stack([np.ones(len(local["train"])), local["train"]])
    assert np.linalg.matrix_rank(p) == np.linalg.matrix_rank(h)
    assert np.linalg.norm(p - h @ np.linalg.lstsq(h, p, rcond=None)[0]) < 1e-8
