#!/usr/bin/env python3
"""Score three semantically grounded application datasets.

These datasets are held out from head fitting and HPO.  They are descriptive
case studies, not extra samples in the predeclared 35-dataset primary test.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from common import CACHE, CONFIG, atomic_json
from score_baselines import (
    CLASSICAL_METHODS,
    functional_rows,
    mean_only_point_rows,
    metadata,
    paired_summary,
    point_rows,
    select_tabpfn_temperatures,
)
from score_projective import load_models, score_episode


MAIN_METHODS = [
    "projtabicl",
    "tabiclv2_diagonal",
    "tabpfn3_diagonal",
    "tabpfn25_diagonal",
    "bayesian_linear",
    "gp_rbf",
    "gp_matern32",
    "catboost_process",
]


def main() -> None:
    source_root = CACHE / "tabicl_singleton_episodes" / "app"
    paths = sorted(source_root.glob("*.npz"))
    expected = (
        len(CONFIG["application_datasets"])
        * len(CONFIG["development_splits"])
        * int(CONFIG["development_context_replicates"])
        * len(CONFIG["context_sizes"])
    )
    if len(paths) != expected:
        raise RuntimeError(f"application singleton cache incomplete: {len(paths)} != {expected}")

    tabpfn_temperatures = select_tabpfn_temperatures()
    tabpfn3_temperatures = select_tabpfn_temperatures(
        prediction_root="tabpfn3_episodes",
        source_root="tabicl_singleton_episodes",
        artifact_root="baselines_singleton",
        artifact_prefix="tabpfn3",
        model_label="TabPFN-3",
    )
    head_summary = json.loads((CACHE / "head_singleton" / "training_summary.json").read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models, training_summary = load_models(device, "head_singleton")
    aggregate: list[dict[str, Any]] = []
    points: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    timing: list[dict[str, Any]] = []

    for index, source_path in enumerate(paths):
        projective_rows, audit = score_episode(source_path, models, training_summary, device)
        aggregate.extend(projective_rows)
        audits.append(audit)
        tabpfn_path = CACHE / "tabpfn_episodes" / "app" / source_path.name
        tabpfn3_path = CACHE / "tabpfn3_episodes" / "app" / source_path.name
        classical_path = CACHE / "classical_episodes" / "app" / source_path.name
        tabdpt_path = CACHE / "tabdpt_episodes" / "app" / source_path.name
        if not tabpfn_path.exists() or not tabpfn3_path.exists() or not classical_path.exists():
            raise RuntimeError(f"missing application baseline for {source_path.name}")

        with np.load(source_path, allow_pickle=False) as source, np.load(
            tabpfn_path, allow_pickle=False
        ) as tabpfn, np.load(classical_path, allow_pickle=False) as classical:
            meta = metadata(source)
            target = source["target"].astype(np.float64)
            coefficients = source["coefficients"].astype(np.float64)
            context_size = int(meta["context_size"])
            tabicl_mean = source["mean"].astype(np.float64)
            tabicl_variance = source["variance"].astype(np.float64) * float(
                head_summary["marginal_temperatures"][str(context_size)]
            )
            points.extend(
                point_rows(
                    meta,
                    "tabiclv2_projtabicl_marginal",
                    tabicl_mean,
                    tabicl_variance,
                    target,
                )
            )
            timing.append(
                {
                    "dataset": meta["dataset"],
                    "context_size": context_size,
                    "method": "tabiclv2_backbone_singleton",
                    "elapsed_seconds": float(meta["elapsed_seconds"]),
                }
            )

            with np.load(tabpfn3_path, allow_pickle=False) as tabpfn3:
                tabpfn3_meta = metadata(tabpfn3)
                for key in ("dataset", "split", "replicate", "context_size", "query_index_sha256"):
                    if tabpfn3_meta[key] != meta[key]:
                        raise ValueError(f"TabPFN-3 cache mismatch for {source_path.name}: {key}")
                tabpfn3_mean = tabpfn3["mean"].astype(np.float64)
                tabpfn3_variance = (
                    tabpfn3["variance"].astype(np.float64)
                    * tabpfn3_temperatures[context_size]
                )
                aggregate.extend(
                    functional_rows(
                        meta,
                        "tabpfn3_diagonal",
                        tabpfn3_mean,
                        np.diag(tabpfn3_variance),
                        target,
                        coefficients,
                    )
                )
                points.extend(
                    point_rows(meta, "tabpfn3", tabpfn3_mean, tabpfn3_variance, target)
                )
                timing.append(
                    {
                        "dataset": meta["dataset"],
                        "context_size": context_size,
                        "method": "tabpfn3",
                        "elapsed_seconds": float(tabpfn3_meta["elapsed_seconds"]),
                    }
                )

            tabpfn_meta = metadata(tabpfn)
            classical_meta = metadata(classical)
            for other, name in ((tabpfn_meta, "TabPFN"), (classical_meta, "classical")):
                for key in ("dataset", "split", "replicate", "context_size", "query_index_sha256"):
                    if other[key] != meta[key]:
                        raise ValueError(f"{name} cache mismatch for {source_path.name}: {key}")
            tabpfn_mean = tabpfn["mean"].astype(np.float64)
            tabpfn_variance = tabpfn["variance"].astype(np.float64) * tabpfn_temperatures[context_size]
            aggregate.extend(
                functional_rows(
                    meta,
                    "tabpfn25_diagonal",
                    tabpfn_mean,
                    np.diag(tabpfn_variance),
                    target,
                    coefficients,
                )
            )
            points.extend(point_rows(meta, "tabpfn25", tabpfn_mean, tabpfn_variance, target))
            timing.append(
                {
                    "dataset": meta["dataset"],
                    "context_size": context_size,
                    "method": "tabpfn25",
                    "elapsed_seconds": float(tabpfn_meta["elapsed_seconds"]),
                }
            )

            if list(classical_meta["methods"]) != CLASSICAL_METHODS:
                raise ValueError(f"unexpected classical method order in {classical_path}")
            means = classical["means"].astype(np.float64)
            covariances = classical["covariances"].astype(np.float64)
            for method_index, method in enumerate(CLASSICAL_METHODS):
                covariance = 0.5 * (
                    covariances[method_index] + covariances[method_index].T
                )
                aggregate.extend(
                    functional_rows(
                        meta,
                        method,
                        means[method_index],
                        covariance,
                        target,
                        coefficients,
                    )
                )
                points.extend(
                    point_rows(
                        meta,
                        method,
                        means[method_index],
                        np.diag(covariance),
                        target,
                    )
                )
                timing.append(
                    {
                        "dataset": meta["dataset"],
                        "context_size": context_size,
                        "method": method,
                        "elapsed_seconds": float(classical_meta["elapsed_seconds"][method_index]),
                    }
                )

            if tabdpt_path.exists():
                with np.load(tabdpt_path, allow_pickle=False) as tabdpt:
                    tabdpt_meta = metadata(tabdpt)
                    for key in ("dataset", "split", "replicate", "context_size", "query_index_sha256"):
                        if tabdpt_meta[key] != meta[key]:
                            raise ValueError(f"TabDPT cache mismatch for {source_path.name}: {key}")
                    points.extend(
                        mean_only_point_rows(
                            meta,
                            "tabdpt_turbo_1_2",
                            tabdpt["mean"].astype(np.float64),
                            target,
                        )
                    )
                    timing.append(
                        {
                            "dataset": meta["dataset"],
                            "context_size": context_size,
                            "method": "tabdpt_turbo_1_2",
                            "elapsed_seconds": float(tabdpt_meta["elapsed_seconds"]),
                        }
                    )
        if (index + 1) % 20 == 0:
            print(f"scored applications {index + 1}/{len(paths)}", flush=True)

    cells = pd.DataFrame(aggregate)
    point_cells = pd.DataFrame(points)
    audit_frame = pd.DataFrame(audits)
    timing_frame = pd.DataFrame(timing)
    primary = cells[cells["family"].isin(CONFIG["primary_aggregate_families"])].copy()
    dataset_method = primary.groupby(["dataset", "method"], as_index=False)[
        ["nll", "crps", "squared_error", "coverage_90", "width_90"]
    ].mean()
    main = dataset_method[dataset_method["method"].isin(MAIN_METHODS)].copy()
    main["nll_rank"] = main.groupby("dataset")["nll"].rank()
    main["crps_rank"] = main.groupby("dataset")["crps"].rank()
    method_table = main.groupby("method", as_index=False).agg(
        nll=("nll", "mean"),
        crps=("crps", "mean"),
        squared_error=("squared_error", "mean"),
        coverage_90=("coverage_90", "mean"),
        width_90=("width_90", "mean"),
        nll_rank=("nll_rank", "mean"),
        crps_rank=("crps_rank", "mean"),
    )
    point_dataset = point_cells.groupby(["dataset", "method"], as_index=False).agg(
        mse=("squared_error", "mean"),
        nll=("nll", "mean"),
        crps=("crps", "mean"),
        coverage_90=("coverage_90", "mean"),
    )
    point_dataset["nrmse"] = np.sqrt(point_dataset["mse"])
    point_method = point_dataset.groupby("method", as_index=False).agg(
        nrmse=("nrmse", "mean"),
        nll=("nll", "mean"),
        crps=("crps", "mean"),
        coverage_90=("coverage_90", "mean"),
    )
    wide_nll = dataset_method.pivot(index="dataset", columns="method", values="nll")
    wide_crps = dataset_method.pivot(index="dataset", columns="method", values="crps")
    nll_effect = wide_nll["tabiclv2_diagonal"] - wide_nll["projtabicl"]
    crps_effect = wide_crps["tabiclv2_diagonal"] - wide_crps["projtabicl"]

    out = CACHE / "results" / "applications_singleton"
    out.mkdir(parents=True, exist_ok=True)
    cells.to_parquet(out / "aggregate_cells.parquet", index=False)
    point_cells.to_parquet(out / "point_cells.parquet", index=False)
    dataset_method.to_csv(out / "aggregate_by_dataset.csv", index=False)
    method_table.to_csv(out / "aggregate_main_methods.csv", index=False)
    point_dataset.to_csv(out / "point_by_dataset.csv", index=False)
    point_method.to_csv(out / "point_by_method.csv", index=False)
    audit_frame.to_csv(out / "integrity.csv", index=False)
    timing_frame.to_csv(out / "timing_cells.csv", index=False)
    pd.DataFrame(
        {
            "dataset": nll_effect.index,
            "nll_diagonal_minus_projtabicl": nll_effect.values,
            "crps_diagonal_minus_projtabicl": crps_effect.reindex(nll_effect.index).values,
        }
    ).to_csv(out / "projective_effects_by_dataset.csv", index=False)

    payload = {
        "role": "held-out descriptive application case studies; excluded from primary inference",
        "episodes": expected,
        "datasets": sorted(cells["dataset"].unique()),
        "method_table": method_table.to_dict(orient="records"),
        "point_table": point_method.to_dict(orient="records"),
        "diagonal_comparison": {
            "nll_diagonal_minus_projtabicl": paired_summary(nll_effect),
            "crps_diagonal_minus_projtabicl": paired_summary(crps_effect),
        },
        "integrity": {
            "maximum_diagonal_error": float(audit_frame["diagonal_max_abs"].max()),
            "minimum_eigenvalue": float(audit_frame["min_eigenvalue"].min()),
        },
    }
    atomic_json(out / "summary.json", payload)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
