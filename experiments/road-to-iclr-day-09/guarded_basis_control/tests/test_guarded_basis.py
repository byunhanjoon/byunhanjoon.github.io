from __future__ import annotations

import json

import numpy as np

from guarded_basis.blockguard import (
    coordinate_audit,
    gram_interface,
    greedy_candidates,
    grouped_candidates,
    mixed_representation,
)
from guarded_basis.common import BLACKLIST_PATH, PANEL_PATH, PROTOCOL_PATH, bd, sha256_file
from guarded_basis.gating import guarded_evidence, select_g1, select_g2, select_g3


def test_locked_panel_is_balanced_and_blacklisted_names_are_disjoint() -> None:
    panel = json.loads(PANEL_PATH.read_text())
    blacklist = json.loads(BLACKLIST_PATH.read_text())
    names = {row["key"] for row in panel["datasets"]}
    assert panel["status"] == "LOCKED_BEFORE_GUARDED_DEVELOPMENT_OUTCOME_ACCESS"
    assert len(names) == 12
    assert names.isdisjoint(blacklist["datasets"])
    assert {row["problem_type"] for row in panel["datasets"]} == {"classification", "regression"}
    assert all(1000 <= row["rows"] <= 50000 and row["raw_columns"] <= 100 for row in panel["datasets"])


def test_all_frozen_hash_sidecars_match() -> None:
    for path in (PANEL_PATH, BLACKLIST_PATH, PROTOCOL_PATH):
        assert path.with_suffix(".sha256").read_text().split()[0] == sha256_file(path)


def test_guarded_gates_back_off_for_obvious_harm() -> None:
    y_train = np.array([0, 0, 1, 1] * 20)
    y = np.array([0, 1] * 20)
    raw = np.tile([[0.95, 0.05], [0.05, 0.95]], (20, 1))
    gram = raw[:, ::-1]
    evidence = guarded_evidence(
        "classification", y, y_train, raw, gram,
        alphas=[0.75, 0.5, 0.25, 0.0], resamples=200, seed=1,
    )
    assert select_g1(evidence, tau=0.01)[0] == 0.0
    assert select_g2(evidence, tau=0.01, gamma=1.0) == 0.0
    assert select_g3(evidence)[0] == 0.0


def test_guarded_gates_keep_default_for_safe_invariant_branch() -> None:
    y_train = np.array([0, 0, 1, 1] * 20)
    y = np.array([0, 1] * 20)
    raw = np.tile([[0.8, 0.2], [0.2, 0.8]], (20, 1))
    gram = raw.copy()
    evidence = guarded_evidence(
        "classification", y, y_train, raw, gram,
        alphas=[0.75, 0.5, 0.25, 0.0], resamples=100, seed=2,
    )
    assert select_g1(evidence, tau=0.0)[0] == 0.75
    assert select_g2(evidence, tau=0.005, gamma=1.64) == 0.75
    assert select_g3(evidence)[0] == 0.75


def test_blockguard_replaces_only_selected_contiguous_block() -> None:
    rng = np.random.default_rng(7)
    train = rng.normal(size=(40, 5))
    validation = rng.normal(size=(12, 5))
    test = rng.normal(size=(13, 5))
    reference = bd.Representation(
        "reference", "test", "reference", "reference", -1,
        train, validation, test,
        ["a0", "a1", "pass", "b0", "b1"],
        {"a": [0, 1], "b": [3, 4]}, {},
        {"a": np.eye(2), "b": np.eye(2)}, {}, True,
    )
    orbit = [reference]
    for member in range(3):
        rep_train, rep_validation, rep_test = train.copy(), validation.copy(), test.copy()
        transforms = {}
        for feature, indices in reference.feature_blocks.items():
            transform = bd.orthogonal_matrix(2, bd.stable_seed("synthetic", feature, member))
            transforms[feature] = transform
            for values in (rep_train, rep_validation, rep_test):
                values[:, indices] = values[:, indices] @ transform
        orbit.append(
            bd.Representation(
                f"rotated_{member}", "test", "orthogonal_all", "all", member,
                rep_train, rep_validation, rep_test, reference.columns,
                reference.feature_blocks, {}, transforms, {}, False,
            )
        )
    mixed = [
        mixed_representation(rep, "synthetic", ["a"], gram_rep=gram_interface(rep, "synthetic"))
        for rep in orbit
    ]
    assert mixed[0].X_train.shape[1] == 16 + 1 + 2
    assert np.array_equal(mixed[0].X_train[:, mixed[0].feature_blocks["b"]], train[:, 3:5])
    assert np.array_equal(mixed[0].X_train[:, 16], train[:, 2])
    assert coordinate_audit(mixed, ["a"])["passes_1e_minus_6"]
    assert not np.allclose(
        mixed[0].X_train[:, mixed[0].feature_blocks["b"]],
        mixed[1].X_train[:, mixed[1].feature_blocks["b"]],
    )


def test_blockguard_group_and_greedy_rules_are_bounded_and_stable() -> None:
    rows = [
        {
            "feature": f"f{index}",
            "feature_index": index,
            "normalized_excess_risk": cost,
            "basis_disagreement_benefit": benefit,
        }
        for index, (cost, benefit) in enumerate(
            [(0.0, 0.1), (0.02, 0.9), (0.001, 0.2), (0.01, 0.3), (0.0, 0.05)]
        )
    ]
    grouped, assignment = grouped_candidates(rows)
    greedy = greedy_candidates(rows, maximum_stages=3)
    assert [row["candidate"] for row in grouped] == [
        "raw_only", "very_safe_gram", "very_safe_plus_safe_gram",
        "all_except_dangerous_gram", "all_gram",
    ]
    assert set(assignment) == {row["feature"] for row in rows}
    assert len(greedy) == 4
    assert greedy[-1]["features"] == ["f0", "f4", "f2", "f1", "f3"]
