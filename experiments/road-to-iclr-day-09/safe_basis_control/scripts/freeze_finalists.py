#!/usr/bin/env python3
"""Freeze at most four development finalists before any prospective data access."""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from safe_basis.common import (  # noqa: E402
    PANEL_PATH,
    PROTOCOL_PATH,
    ROOT,
    load_json,
    sha256_file,
    write_json,
)


def main() -> None:
    processed = ROOT / "results" / "processed"
    required = [
        processed / "development_gate_summary.csv",
        processed / "rank_development_summary.csv",
        processed / "embedding_main_method_units.csv",
        processed / "rank_selection.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"cannot freeze finalists; missing development artifacts: {missing}")
    prospective_root = ROOT / "results" / "raw" / "prospective"
    if prospective_root.exists() and any(prospective_root.rglob("*.npz")):
        raise RuntimeError("cannot freeze after prospective prediction artifacts exist")
    gate_summary = pd.read_csv(processed / "development_gate_summary.csv")
    rank_summary = pd.read_csv(processed / "rank_development_summary.csv")
    embedding = pd.read_csv(processed / "embedding_main_method_units.csv")
    rank_selection = load_json(processed / "rank_selection.json")

    def gate_evidence(method: str) -> dict[str, float]:
        row = gate_summary[gate_summary["method"] == method]
        if len(row) != 1:
            raise RuntimeError(f"missing unique gate development evidence for {method}")
        return {key: float(row.iloc[0][key]) for key in ("median_disagreement_reduction", "median_C", "p95_C", "max_C", "raw_fallback_rate")}

    def rank_evidence(method: str) -> dict[str, float]:
        row = rank_summary[rank_summary["method"] == method]
        if len(row) != 1:
            raise RuntimeError(f"missing unique rank development evidence for {method}")
        return {key: float(row.iloc[0][key]) for key in ("median_disagreement_reduction", "median_C", "p95_C", "max_C", "raw_fallback_rate")}

    embedding_evidence = (
        embedding.groupby("method", as_index=False)
        .agg(
            median_disagreement_reduction=("disagreement_reduction", "median"),
            median_C=("normalized_excess_risk", "median"),
            model_families=("model", "nunique"),
        )
        .set_index("method")
    )

    safe_rule = {
        "selection_split": "validation_only",
        "alphas": [0.0, 0.25, 0.5, 0.75, 1.0],
        "criterion": "largest alpha with row-bootstrap UCB95(C_alpha) <= tau",
        "tau": 0.01,
        "bootstrap_resamples": 500,
        "epsilon": 1e-8,
        "fallback": 0.0,
    }
    rank_config = dict(rank_selection["config"])
    finalists = [
        {
            "method_id": "GramAnchor-m16",
            "type": "invariant_interface",
            "interface": "gram_anchor",
            "interface_parameters": {"anchors": 16, "selection": "gram_pivot", "normalize": True, "coordinate_standardization": True},
            "development_evidence": gate_evidence("GramAnchor"),
        },
        {
            "method_id": "RankAdaptiveGram",
            "type": "rank_adaptive_invariant_interface",
            "interface": "rank_adaptive_gram",
            "rank_config": rank_config,
            "development_evidence": rank_evidence("RankAdaptiveGram"),
            "embedding_evidence": embedding_evidence.loc["RankAdaptiveGram-after-embedding"].to_dict(),
        },
        {
            "method_id": "SafeGram-t01",
            "type": "validation_controlled_prediction_hybrid",
            "invariant_branch": "GramAnchor-m16",
            "alpha_rule": safe_rule,
            "development_evidence": gate_evidence("SafeGram-t01"),
            "embedding_evidence": embedding_evidence.loc["SafeGram-after-embedding"].to_dict(),
        },
        {
            "method_id": "SafeRankGram-t01",
            "type": "validation_controlled_rank_adaptive_prediction_hybrid",
            "invariant_branch": "RankAdaptiveGram",
            "rank_config": rank_config,
            "alpha_rule": safe_rule,
            "development_evidence": rank_evidence("SafeRankGram-t01"),
            "embedding_evidence": embedding_evidence.loc["SafeRankGram prediction hybrid"].to_dict(),
        },
    ]
    if len(finalists) > 4:
        raise RuntimeError("finalist cap violated")
    output = ROOT / "configs" / "TAIL_FINALISTS.json"
    config = {
        "status": "FROZEN_BEFORE_PROSPECTIVE_DATA_ACCESS",
        "frozen_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "repository_commit": load_json(PROTOCOL_PATH)["repository_commit"],
        "selection_scope": "eight development datasets; validation-only gates; three embedding backbones; no new prospective outcomes",
        "prospective_outcomes_accessed": False,
        "prospective_panel_sha256": sha256_file(PANEL_PATH),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "development_artifact_hashes": {path.name: sha256_file(path) for path in required},
        "model_hpo": {
            "controlled_mlp": "frozen prior 3x256 GELU AdamW lr=0.001 wd=0.0001; early stopping",
            "tabm_d": "frozen prior TabM-D n_epochs=30 patience=5; native AdamW",
            "catboost": "200 trees depth=7 learning_rate=0.05; early stopping",
            "resnet_tabular_embedding_only": "3x256 residual LayerNorm ReLU dropout=0.1 AdamW lr=0.001 wd=0.0001",
        },
        "finalists": finalists,
    }
    write_json(output, config)
    digest = sha256_file(output)
    (ROOT / "configs" / "TAIL_FINALISTS.sha256").write_text(f"{digest}  TAIL_FINALISTS.json\n")
    print(digest)


if __name__ == "__main__":
    main()
