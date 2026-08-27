#!/usr/bin/env python3
"""Build the compact, auditable result tables used by the Day 4 post."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
BASELINE = "quantile_ple"
DIRECT_CONTROLS = [
    BASELINE,
    "raple_raw",
    "anchor_only",
    "anchor_mass_representer",
    "anchor_wrong_representer",
]
CORRECT = "anchor_riesz_representer"


def read(name: str) -> pd.DataFrame:
    frame = pd.read_csv(RESULTS / name)
    frame["source"] = name
    return frame


def paired_gains(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach relative test-loss gain against the paired PLE baseline."""
    keys = ["dataset", "model", "seed"]
    baselines = (
        frame.loc[frame["method"].eq(BASELINE), keys + ["test_loss"]]
        .drop_duplicates(keys, keep="last")
        .rename(columns={"test_loss": "baseline_test_loss"})
    )
    out = frame.merge(baselines, on=keys, how="left", validate="many_to_one")
    out["test_loss_gain_pct"] = (
        100 * (out["baseline_test_loss"] - out["test_loss"])
        / out["baseline_test_loss"]
    )
    return out


def select_methods(frame: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    return paired_gains(frame.loc[frame["method"].isin(methods)].copy())


def records(frame: pd.DataFrame) -> list[dict[str, object]]:
    """Return strict-JSON records, mapping pandas NaN to JSON null."""
    return json.loads(frame.to_json(orient="records"))


def comparison_counts(
    wide: pd.DataFrame,
    candidate: str = CORRECT,
    controls: list[str] = DIRECT_CONTROLS,
) -> dict[str, dict[str, object]]:
    """Summarize paired percentage gains for one candidate in a loss table."""
    out: dict[str, dict[str, object]] = {}
    for control in controls:
        pairs = wide[[candidate, control]].dropna()
        gains = 100 * (pairs[control] - pairs[candidate]) / pairs[control]
        out[control] = {
            "wins": int((gains > 0).sum()),
            "pairs": int(len(gains)),
            "mean_gain_pct": float(gains.mean()),
            "median_gain_pct": float(gains.median()),
        }
    return out


def direct_rows(wide: pd.DataFrame) -> pd.DataFrame:
    """Create dataset/backbone rows for the shared-anchor direct control."""
    rows: list[dict[str, object]] = []
    for (dataset, model), group in wide.groupby(level=[0, 1]):
        row: dict[str, object] = {
            "dataset": dataset,
            "model": model,
            "seeds": int(len(group)),
            "correct_mean_test_loss": float(group[CORRECT].mean()),
        }
        for control in DIRECT_CONTROLS:
            gains = 100 * (group[control] - group[CORRECT]) / group[control]
            row[f"{control}_mean_test_loss"] = float(group[control].mean())
            row[f"gain_vs_{control}_pct"] = float(gains.mean())
            row[f"wins_vs_{control}"] = int((gains > 0).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    # One fixed seed, parameter matched, across the four tested architecture families.
    architecture = pd.concat(
        [
            read("mass_canonical_ablation_abc.csv"),
            read("tabm_support_abc.csv"),
            read("ft_support_abc.csv"),
        ],
        ignore_index=True,
    )
    architecture = select_methods(
        architecture,
        [
            BASELINE,
            "adaptive_support_whitened",
            "adaptive_support_riesz",
            "adaptive_support_wrong_riesz",
        ],
    )
    architecture[
        [
            "dataset", "model", "seed", "method", "val_loss", "test_loss",
            "test_loss_gain_pct", "parameter_error_fraction", "source",
        ]
    ].to_csv(RESULTS / "day4_architecture_transport.csv", index=False)

    # Three paired seed runs with held-out test splits for the California test.
    geometry = paired_gains(read("california_geometry_repeats.csv"))
    geometry_summary = (
        geometry.groupby(["model", "method"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            mean_test_loss=("test_loss", "mean"),
            sd_test_loss=("test_loss", "std"),
            mean_test_loss_gain_pct=("test_loss_gain_pct", "mean"),
        )
    )
    geometry_summary.to_csv(RESULTS / "day4_california_geometry.csv", index=False)

    # Broad one-seed mass screen, including inactive-field identity controls.
    broad = paired_gains(read("mass_canonical_ablation_broad.csv"))
    broad[
        [
            "dataset", "model", "method", "val_loss", "test_loss",
            "test_loss_gain_pct", "input_size", "baseline_parameter_count",
            "parameter_count",
        ]
    ].to_csv(RESULTS / "day4_broad_mass_screen.csv", index=False)

    # Official temporal splits. Every non-baseline row here is negative evidence.
    temporal = pd.concat(
        [
            read("mass_canonical_tabred_weather.csv"),
            read("field_riesz_churn_weather.csv").query("dataset == 'tabred-weather'"),
            read("field_riesz_tabred_cooking-time.csv"),
            read("field_riesz_tabred_delivery-eta.csv"),
        ],
        ignore_index=True,
    ).drop_duplicates(["dataset", "model", "seed", "method"], keep="last")
    temporal = paired_gains(temporal)
    temporal[
        [
            "dataset", "model", "seed", "method", "val_loss", "test_loss",
            "test_loss_gain_pct", "source",
        ]
    ].to_csv(RESULTS / "day4_temporal_checks.csv", index=False)

    # Single-field intervention: the only changed object is one field's stiffness.
    baseline = read("california_field_baseline.csv").iloc[0]
    field_rows: list[dict[str, object]] = []
    field_names = {
        0: "MedInc", 1: "HouseAge", 3: "AveBedrms",
        6: "Latitude", 7: "Longitude",
    }
    for field_index, field_name in field_names.items():
        correct = read(f"california_field_{field_index}.csv").iloc[0]
        row: dict[str, object] = {
            "field_index": field_index,
            "field": field_name,
            "baseline_test_loss": float(baseline["test_loss"]),
            "correct_test_loss": float(correct["test_loss"]),
            "correct_gain_pct": 100 * (
                float(baseline["test_loss"]) - float(correct["test_loss"])
            ) / float(baseline["test_loss"]),
        }
        wrong_path = RESULTS / f"california_field_{field_index}_wrong.csv"
        if wrong_path.exists():
            wrong = pd.read_csv(wrong_path).iloc[0]
            row["wrong_test_loss"] = float(wrong["test_loss"])
            row["correct_vs_wrong_pct"] = 100 * (
                float(wrong["test_loss"]) - float(correct["test_loss"])
            ) / float(wrong["test_loss"])
        field_rows.append(row)
    field_frame = pd.DataFrame(field_rows)
    field_frame.to_csv(RESULTS / "day4_california_field_ablation.csv", index=False)

    # Independent declared-cyclic test. The ring hypothesis fails to replicate,
    # but empirical mass improves for both backbones over chronological splits.
    bike = paired_gains(read("bike_cyclic_geometry.csv"))
    bike_summary = (
        bike.groupby(["model", "method"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            mean_val_loss=("val_loss", "mean"),
            mean_test_loss=("test_loss", "mean"),
            sd_test_loss=("test_loss", "std"),
            mean_test_loss_gain_pct=("test_loss_gain_pct", "mean"),
        )
    )
    bike_summary.to_csv(RESULTS / "day4_bike_cyclic.csv", index=False)

    residual = pd.concat(
        [
            read("residual_riesz_california.csv"),
            read("residual_riesz_weather.csv"),
            read("residual_riesz_cooking.csv"),
            read("residual_riesz_delivery.csv"),
        ],
        ignore_index=True,
    )
    residual = paired_gains(residual)
    residual_keys = ["dataset", "model", "seed"]
    anchors = (
        residual.loc[
            residual["method"].eq("anchor_only"),
            residual_keys + ["test_loss"],
        ]
        .rename(columns={"test_loss": "anchor_test_loss"})
        .drop_duplicates(residual_keys)
    )
    residual = residual.merge(anchors, on=residual_keys, validate="many_to_one")
    residual["test_loss_gain_vs_anchor_pct"] = 100 * (
        residual["anchor_test_loss"] - residual["test_loss"]
    ) / residual["anchor_test_loss"]
    residual_summary = (
        residual.groupby(["dataset", "model", "method"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            mean_val_loss=("val_loss", "mean"),
            mean_test_loss=("test_loss", "mean"),
            sd_test_loss=("test_loss", "std"),
            mean_gain_vs_ple_pct=("test_loss_gain_pct", "mean"),
            mean_gain_vs_anchor_pct=("test_loss_gain_vs_anchor_pct", "mean"),
        )
    )
    residual_summary.to_csv(RESULTS / "day4_residual_riesz.csv", index=False)
    correct_rows = residual.loc[
        residual["method"].eq("anchor_riesz_representer"),
        residual_keys + ["test_loss"],
    ].rename(columns={"test_loss": "correct_test_loss"})
    residual_pair_wins = {}
    for control in (
        "anchor_only",
        "anchor_mass_representer",
        "anchor_wrong_representer",
    ):
        control_rows = residual.loc[
            residual["method"].eq(control), residual_keys + ["test_loss"]
        ].rename(columns={"test_loss": "control_test_loss"})
        pairs = correct_rows.merge(control_rows, on=residual_keys, validate="one_to_one")
        residual_pair_wins[control] = {
            "wins": int((pairs["correct_test_loss"] < pairs["control_test_loss"]).sum()),
            "pairs": int(len(pairs)),
            "mean_gain_pct": float(
                (100 * (pairs["control_test_loss"] - pairs["correct_test_loss"])
                 / pairs["control_test_loss"]).mean()
            ),
        }

    # Decisive control: all variants below use the exact same RAPLE encoder,
    # out-of-fold LightGBM anchor, residuals, split, seeds, and neural budget.
    # The MLP/ResNet panel and TabM panel use separate runners but the same
    # feature-building implementation and three registered seeds.
    residual_vs_raple = pd.concat(
        [
            read("residual_riesz_vs_raple_california.csv"),
            read("residual_riesz_vs_raple_weather.csv"),
            read("residual_riesz_vs_raple_cooking.csv"),
            read("residual_riesz_vs_raple_delivery.csv"),
            read("residual_riesz_vs_raple_maps.csv"),
            read("tabm_residual_riesz_vs_raple.csv"),
        ],
        ignore_index=True,
    )
    direct_keys = ["dataset", "model", "seed"]
    direct_wide = residual_vs_raple.pivot(
        index=direct_keys, columns="method", values="test_loss"
    )
    direct_pair_wins = comparison_counts(direct_wide)
    direct_pair_wins["mlp_resnet_only"] = comparison_counts(
        direct_wide.loc[direct_wide.index.get_level_values("model") != "tabm"]
    )
    direct_pair_wins["tabm_only"] = comparison_counts(
        direct_wide.loc[direct_wide.index.get_level_values("model") == "tabm"]
    )
    direct_summary = direct_rows(direct_wide)
    direct_summary.to_csv(RESULTS / "day4_residual_vs_raple.csv", index=False)

    # Dataset/backbone means are the independent-looking unit for an
    # exploratory sign audit; seed rows are not treated as 45 independent
    # scientific replications.
    direct_aggregate = (
        residual_vs_raple.groupby(["dataset", "model", "method"], as_index=False)
        .agg(mean_test_loss=("test_loss", "mean"), seeds=("seed", "nunique"))
        .pivot(index=["dataset", "model"], columns="method", values="mean_test_loss")
    )
    direct_aggregate_counts = comparison_counts(direct_aggregate)

    # Sparse Maps screens all reuse the same direct-control reference rows.
    # Selection based on mass (tau=0) is topology-neutral; selecting by the
    # declared Riesz topology is reported separately because it can bias the
    # mechanism comparison.
    maps_screen_files = {
        "dense_all_fields": "residual_riesz_vs_raple_maps.csv",
        "mass_energy_top8": "residual_riesz_maps_top8.csv",
        "mass_energy_top24": "residual_riesz_maps_top24.csv",
        "mass_energy_top64": "residual_riesz_maps_top64.csv",
        "mass_oof_top24": "residual_riesz_maps_mass_oof.csv",
        "riesz_oof_top24": "residual_riesz_maps_riesz_oof.csv",
        "strict_z_screen": "residual_riesz_maps_zscreen.csv",
    }
    if (RESULTS / "residual_riesz_maps_fdr10.csv").exists():
        maps_screen_files["bh_fdr_10pct"] = "residual_riesz_maps_fdr10.csv"
    maps_reference = read("residual_riesz_vs_raple_maps.csv").pivot(
        index=direct_keys, columns="method", values="test_loss"
    )
    maps_screen_rows: list[dict[str, object]] = []
    for screen, filename in maps_screen_files.items():
        screen_frame = read(filename)
        screen_wide = screen_frame.pivot(
            index=direct_keys, columns="method", values="test_loss"
        )
        joined = maps_reference[DIRECT_CONTROLS].join(
            screen_wide[[CORRECT, "anchor_wrong_representer"]],
            how="inner",
            rsuffix="_screen",
        )
        # Avoid the name collision with the reference's dense wrong control.
        candidate = joined[CORRECT]
        controls = {
            BASELINE: joined[BASELINE],
            "raple_raw": joined["raple_raw"],
            "anchor_only": joined["anchor_only"],
            "anchor_mass_representer": joined["anchor_mass_representer"],
            "dense_correct": maps_reference.loc[joined.index, CORRECT],
            "screen_wrong": joined["anchor_wrong_representer_screen"],
        }
        row: dict[str, object] = {"screen": screen, "pairs": int(len(joined))}
        for control, values in controls.items():
            gains = 100 * (values - candidate) / values
            row[f"wins_vs_{control}"] = int((gains > 0).sum())
            row[f"gain_vs_{control}_pct"] = float(gains.mean())
        maps_screen_rows.append(row)
    maps_screen_summary = pd.DataFrame(maps_screen_rows)
    maps_screen_summary.to_csv(RESULTS / "day4_maps_sparse_screens.csv", index=False)

    # A strict family-wise z screen is deliberately allowed to select no
    # representers. In that case both semantic variants reduce exactly to the
    # shared anchor, which measures safe abstention rather than a geometry win.
    strict = read("residual_riesz_zscreen_broad.csv")
    strict_meta = json.loads(
        (RESULTS / "residual_riesz_zscreen_broad.metadata.json").read_text()
    )
    strict_rows: list[dict[str, object]] = []
    for (dataset, model), frame in strict.groupby(["dataset", "model"]):
        index = [dataset, model]
        reference = direct_wide.xs(tuple(index), level=["dataset", "model"])
        correct = frame.loc[frame["method"].eq(CORRECT)].set_index("seed")["test_loss"]
        wrong = frame.loc[
            frame["method"].eq("anchor_wrong_representer")
        ].set_index("seed")["test_loss"]
        row: dict[str, object] = {
            "dataset": dataset,
            "model": model,
            "selected_fields": int(len(strict_meta[dataset]["selected_fields"])),
            "pairs": int(len(correct)),
        }
        for name, values in {
            BASELINE: reference[BASELINE],
            "raple_raw": reference["raple_raw"],
            "anchor_only": reference["anchor_only"],
            "screen_wrong": wrong,
        }.items():
            aligned = correct.to_frame("correct").join(values.rename("control"))
            gains = 100 * (aligned["control"] - aligned["correct"]) / aligned["control"]
            row[f"wins_vs_{name}"] = int((gains > 0).sum())
            row[f"gain_vs_{name}_pct"] = float(gains.mean())
        strict_rows.append(row)
    strict_summary = pd.DataFrame(strict_rows)
    strict_summary.to_csv(RESULTS / "day4_strict_selector.csv", index=False)

    # Robustness of the semantic control to the arbitrary node permutation.
    # The same trained correct-geometry rows are reused, so these are control
    # perturbations, not independent method replications.
    permutation_rows: list[dict[str, object]] = []
    correct_subset = direct_wide.loc[
        direct_wide.index.get_level_values("dataset").isin(
            ["california", "tabred-weather"]
        )
        & direct_wide.index.get_level_values("model").isin(["mlp", "resnet"]),
        [CORRECT],
    ]
    default_wrong = direct_wide.loc[correct_subset.index, ["anchor_wrong_representer"]]
    permutation_frames: list[tuple[str, pd.DataFrame]] = [("991337", default_wrong)]
    for path in sorted(RESULTS.glob("residual_riesz_wrong_perm_*.csv")):
        perm = path.stem.rsplit("_", 1)[-1]
        frame = pd.read_csv(path).pivot(
            index=direct_keys, columns="method", values="test_loss"
        )
        permutation_frames.append((perm, frame[["anchor_wrong_representer"]]))
    for perm, wrong_frame in permutation_frames:
        pairs = correct_subset.join(wrong_frame, how="inner")
        for (dataset, model), group in pairs.groupby(level=[0, 1]):
            gains = 100 * (
                group["anchor_wrong_representer"] - group[CORRECT]
            ) / group["anchor_wrong_representer"]
            permutation_rows.append(
                {
                    "permutation_seed": int(perm),
                    "dataset": dataset,
                    "model": model,
                    "pairs": int(len(gains)),
                    "correct_wins": int((gains > 0).sum()),
                    "mean_correct_gain_pct": float(gains.mean()),
                }
            )
    permutation_summary = pd.DataFrame(permutation_rows)
    permutation_summary.to_csv(
        RESULTS / "day4_wrong_geometry_permutations.csv", index=False
    )
    permutation_cell_frames: list[pd.DataFrame] = []
    for perm, wrong_frame in permutation_frames:
        cells = correct_subset.join(wrong_frame, how="inner").reset_index()
        cells["permutation_seed"] = int(perm)
        permutation_cell_frames.append(cells)
    permutation_cell_long = pd.concat(permutation_cell_frames, ignore_index=True)
    permutation_cell_average = permutation_cell_long.groupby(
        direct_keys, as_index=False
    ).agg(
        correct_test_loss=(CORRECT, "first"),
        mean_permuted_test_loss=("anchor_wrong_representer", "mean"),
        sd_permuted_test_loss=("anchor_wrong_representer", "std"),
        permutations=("permutation_seed", "nunique"),
    )
    permutation_cell_average["correct_gain_pct"] = 100 * (
        permutation_cell_average["mean_permuted_test_loss"]
        - permutation_cell_average["correct_test_loss"]
    ) / permutation_cell_average["mean_permuted_test_loss"]
    permutation_cell_average.to_csv(
        RESULTS / "day4_wrong_geometry_cell_average.csv", index=False
    )
    permutation_cell_summary = pd.DataFrame(
        [
            {
                "unique_cells": int(len(permutation_cell_average)),
                "correct_wins": int(
                    (permutation_cell_average["correct_gain_pct"] > 0).sum()
                ),
                "mean_correct_gain_pct": float(
                    permutation_cell_average["correct_gain_pct"].mean()
                ),
            }
        ]
    )
    permutation_cell_summary.to_csv(
        RESULTS / "day4_wrong_geometry_cell_average_summary.csv", index=False
    )

    # Exact generalized-spectrum control. This is harder than permuting path
    # nodes because it preserves the eigenvalues of (S, M), changing only the
    # allocation of those smoothness costs to field-function modes.
    isospectral_summary = pd.DataFrame()
    isospectral_rotation_summary = pd.DataFrame()
    isospectral_cell_summary = pd.DataFrame()
    isospectral_path = RESULTS / "residual_riesz_isospectral_california_weather.csv"
    if isospectral_path.exists():
        isospectral_frames = [pd.read_csv(isospectral_path)]
        tabm_isospectral_path = RESULTS / "tabm_residual_riesz_isospectral.csv"
        if tabm_isospectral_path.exists():
            isospectral_frames.append(pd.read_csv(tabm_isospectral_path))
        isospectral = pd.concat(isospectral_frames, ignore_index=True).pivot(
            index=direct_keys, columns="method", values="test_loss"
        )
        correct_positive = direct_wide.loc[
            direct_wide.index.get_level_values("dataset").isin(
                ["california", "tabred-weather"]
            ),
            [CORRECT],
        ]
        iso_pairs = correct_positive.join(isospectral, how="inner")
        iso_rows: list[dict[str, object]] = []
        for (dataset, model), group in iso_pairs.groupby(level=[0, 1]):
            gains = 100 * (
                group["anchor_isospectral_representer"] - group[CORRECT]
            ) / group["anchor_isospectral_representer"]
            iso_rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "pairs": int(len(gains)),
                    "correct_wins": int((gains > 0).sum()),
                    "mean_correct_gain_pct": float(gains.mean()),
                }
            )
        isospectral_summary = pd.DataFrame(iso_rows)
        isospectral_summary.to_csv(
            RESULTS / "day4_isospectral_control.csv", index=False
        )
        rotation_frames: list[tuple[int, pd.DataFrame]] = [(991337, isospectral)]
        for path in sorted(RESULTS.glob("residual_riesz_isospectral_[0-9]*.csv")):
            control_seed = int(path.stem.rsplit("_", 1)[-1])
            control_frames = [pd.read_csv(path)]
            tabm_path = RESULTS / f"tabm_residual_riesz_isospectral_{control_seed}.csv"
            if tabm_path.exists():
                control_frames.append(pd.read_csv(tabm_path))
            frame = pd.concat(control_frames, ignore_index=True).pivot(
                index=direct_keys, columns="method", values="test_loss"
            )
            rotation_frames.append((control_seed, frame))
        rotation_rows: list[dict[str, object]] = []
        for control_seed, frame in rotation_frames:
            pairs = correct_positive.join(frame, how="inner")
            for (dataset, model), group in pairs.groupby(level=[0, 1]):
                gains = 100 * (
                    group["anchor_isospectral_representer"] - group[CORRECT]
                ) / group["anchor_isospectral_representer"]
                rotation_rows.append(
                    {
                        "control_seed": control_seed,
                        "dataset": dataset,
                        "model": model,
                        "pairs": int(len(gains)),
                        "correct_wins": int((gains > 0).sum()),
                        "mean_correct_gain_pct": float(gains.mean()),
                    }
                )
        isospectral_rotation_summary = pd.DataFrame(rotation_rows)
        isospectral_rotation_summary.to_csv(
            RESULTS / "day4_isospectral_rotations.csv", index=False
        )

        # Rotations share the same trained semantic model. Average randomized
        # controls within each dataset-model-seed cell before counting wins so
        # the 90 stress comparisons are not mistaken for 90 independent runs.
        cell_frames: list[pd.DataFrame] = []
        for control_seed, frame in rotation_frames:
            cells = correct_positive.join(
                frame[["anchor_isospectral_representer"]], how="inner"
            ).reset_index()
            cells["control_seed"] = control_seed
            cell_frames.append(cells)
        cell_long = pd.concat(cell_frames, ignore_index=True)
        cell_average = cell_long.groupby(direct_keys, as_index=False).agg(
            correct_test_loss=(CORRECT, "first"),
            mean_isospectral_test_loss=("anchor_isospectral_representer", "mean"),
            sd_isospectral_test_loss=("anchor_isospectral_representer", "std"),
            rotations=("control_seed", "nunique"),
        )
        cell_average["correct_gain_pct"] = 100 * (
            cell_average["mean_isospectral_test_loss"]
            - cell_average["correct_test_loss"]
        ) / cell_average["mean_isospectral_test_loss"]
        cell_average.to_csv(
            RESULTS / "day4_isospectral_cell_average.csv", index=False
        )
        cell_rows: list[dict[str, object]] = []
        for model, group in cell_average.groupby("model"):
            cell_rows.append(
                {
                    "model": model,
                    "unique_cells": int(len(group)),
                    "correct_wins": int((group["correct_gain_pct"] > 0).sum()),
                    "mean_correct_gain_pct": float(group["correct_gain_pct"].mean()),
                }
            )
        cell_rows.append(
            {
                "model": "all",
                "unique_cells": int(len(cell_average)),
                "correct_wins": int((cell_average["correct_gain_pct"] > 0).sum()),
                "mean_correct_gain_pct": float(cell_average["correct_gain_pct"].mean()),
            }
        )
        isospectral_cell_summary = pd.DataFrame(cell_rows)
        isospectral_cell_summary.to_csv(
            RESULTS / "day4_isospectral_cell_average_summary.csv", index=False
        )

    spectral_paths = [
        RESULTS / "residual_spectral_profile_california_weather_summary.csv",
        RESULTS / "residual_spectral_profile_cooking_delivery_summary.csv",
        RESULTS / "residual_spectral_profile_maps_summary.csv",
    ]
    spectral_summary = pd.concat(
        [pd.read_csv(path) for path in spectral_paths if path.exists()],
        ignore_index=True,
    ) if any(path.exists() for path in spectral_paths) else pd.DataFrame()
    if not spectral_summary.empty:
        spectral_summary.to_csv(
            RESULTS / "day4_semantic_spectral_profile.csv", index=False
        )

    # Topology-specific localization is allowed as a performance heuristic but
    # not as mechanism evidence: selecting fields by semantic-vs-isospectral
    # gap biases the subsequent control comparison toward the semantic method.
    localizer_rows: list[dict[str, object]] = []
    localizer_files = {
        "top8_positive": "residual_riesz_semantic_localizer_top8.csv",
        "top2_california": "residual_riesz_semantic_localizer_top2_california.csv",
    }
    for label, filename in localizer_files.items():
        path = RESULTS / filename
        if not path.exists():
            continue
        frame = pd.read_csv(path).set_index(direct_keys)["test_loss"]
        for (dataset, model), candidate in frame.groupby(level=[0, 1]):
            reference = direct_wide.xs(
                (dataset, model), level=["dataset", "model"]
            )
            candidate = candidate.droplevel(["dataset", "model"])
            row: dict[str, object] = {
                "localizer": label,
                "dataset": dataset,
                "model": model,
                "pairs": int(len(candidate)),
                "mean_test_loss": float(candidate.mean()),
            }
            for name in [BASELINE, "raple_raw", "anchor_only", CORRECT]:
                aligned = candidate.to_frame("candidate").join(
                    reference[name].rename("control")
                )
                gains = 100 * (
                    aligned["control"] - aligned["candidate"]
                ) / aligned["control"]
                row[f"wins_vs_{name}"] = int((gains > 0).sum())
                row[f"gain_vs_{name}_pct"] = float(gains.mean())
            localizer_rows.append(row)
    localizer_summary = pd.DataFrame(localizer_rows)
    if not localizer_summary.empty:
        localizer_summary.to_csv(
            RESULTS / "day4_semantic_localizer.csv", index=False
        )

    strength_rows: list[dict[str, object]] = []
    strength_files = {
        0.3: RESULTS / "residual_riesz_strength_0p3.csv",
        3.0: RESULTS / "residual_riesz_strength_3.csv",
    }
    for strength, path in strength_files.items():
        if not path.exists():
            continue
        wide = pd.read_csv(path).pivot(
            index=direct_keys, columns="method", values="test_loss"
        )
        for (dataset, model), group in wide.groupby(level=[0, 1]):
            row: dict[str, object] = {
                "strength": strength,
                "dataset": dataset,
                "model": model,
                "pairs": int(len(group)),
                "correct_mean_test_loss": float(group[CORRECT].mean()),
            }
            reference = direct_wide.xs(
                (dataset, model), level=["dataset", "model"]
            )
            for name, values in {
                "node": group["anchor_wrong_representer"],
                "isospectral": group["anchor_isospectral_representer"],
                "raple_raw": reference.loc[
                    group.index.get_level_values("seed"), "raple_raw"
                ],
            }.items():
                gains = 100 * (values.to_numpy() - group[CORRECT].to_numpy()) / values.to_numpy()
                row[f"wins_vs_{name}"] = int((gains > 0).sum())
                row[f"gain_vs_{name}_pct"] = float(gains.mean())
            strength_rows.append(row)
    strength_summary = pd.DataFrame(strength_rows)
    if not strength_summary.empty:
        strength_summary.to_csv(
            RESULTS / "day4_strength_robustness.csv", index=False
        )

    # A predeclared external spatial replication: only latitude and longitude
    # receive representers on a chronological King County house-sales split.
    king_summary = pd.DataFrame()
    king_control_summary = pd.DataFrame()
    king_paths = [
        RESULTS / "king_county_spatial_mlp_tabm.csv",
        RESULTS / "king_county_spatial_resnet.csv",
    ]
    if all(path.exists() for path in king_paths):
        king = pd.concat([pd.read_csv(path) for path in king_paths], ignore_index=True)
        king_wide = king.pivot(
            index=["model", "seed"], columns="method", values="test_loss"
        )
        king_rows: list[dict[str, object]] = []
        for model, group in king_wide.groupby(level="model"):
            row: dict[str, object] = {
                "model": model,
                "pairs": int(len(group)),
                "correct_mean_test_loss": float(group[CORRECT].mean()),
            }
            for name in [
                "raple_raw",
                "anchor_only",
                "anchor_mass_representer",
                "anchor_wrong_representer",
                "anchor_isospectral_representer",
            ]:
                gains = 100 * (group[name] - group[CORRECT]) / group[name]
                row[f"wins_vs_{name}"] = int((gains > 0).sum())
                row[f"gain_vs_{name}_pct"] = float(gains.mean())
            king_rows.append(row)
        king_summary = pd.DataFrame(king_rows)
        king_summary.to_csv(RESULTS / "day4_king_county_spatial.csv", index=False)

        king_controls: list[tuple[int, pd.DataFrame]] = [(991337, king_wide)]
        for path in sorted(RESULTS.glob("king_county_spatial_controls_*.csv")):
            control_seed = int(path.stem.rsplit("_", 1)[-1])
            frame = pd.read_csv(path).pivot(
                index=["model", "seed"], columns="method", values="test_loss"
            )
            king_controls.append((control_seed, frame))
        king_cell_frames: list[pd.DataFrame] = []
        correct_king = king_wide[[CORRECT]]
        for control_seed, frame in king_controls:
            cells = correct_king.join(
                frame[[
                    "anchor_wrong_representer",
                    "anchor_isospectral_representer",
                ]],
                how="inner",
            ).reset_index()
            cells["control_seed"] = control_seed
            king_cell_frames.append(cells)
        king_cell_long = pd.concat(king_cell_frames, ignore_index=True)
        king_cell_average = king_cell_long.groupby(
            ["model", "seed"], as_index=False
        ).agg(
            correct_test_loss=(CORRECT, "first"),
            mean_node_test_loss=("anchor_wrong_representer", "mean"),
            mean_isospectral_test_loss=("anchor_isospectral_representer", "mean"),
            controls=("control_seed", "nunique"),
        )
        for label in ["node", "isospectral"]:
            king_cell_average[f"gain_vs_{label}_pct"] = 100 * (
                king_cell_average[f"mean_{label}_test_loss"]
                - king_cell_average["correct_test_loss"]
            ) / king_cell_average[f"mean_{label}_test_loss"]
        king_cell_average.to_csv(
            RESULTS / "day4_king_county_control_cell_average.csv", index=False
        )
        king_control_rows: list[dict[str, object]] = []
        for model, group in king_cell_average.groupby("model"):
            row = {"model": model, "unique_cells": int(len(group))}
            for label in ["node", "isospectral"]:
                gains = group[f"gain_vs_{label}_pct"]
                row[f"wins_vs_{label}"] = int((gains > 0).sum())
                row[f"gain_vs_{label}_pct"] = float(gains.mean())
            king_control_rows.append(row)
        king_control_summary = pd.DataFrame(king_control_rows)
        king_control_summary.to_csv(
            RESULTS / "day4_king_county_control_summary.csv", index=False
        )

    product_summary = pd.DataFrame()
    surface_sensitivity = pd.DataFrame()
    surface_selection = pd.DataFrame()
    surface_strength = pd.DataFrame()
    surface_reference_mass = pd.DataFrame()
    surface_rho_mixture = pd.DataFrame()
    surface_reference_selection = pd.DataFrame()
    surface_reference_rotations = pd.DataFrame()
    product_paths = [
        RESULTS / "spatial_product_riesz_california.csv",
        RESULTS / "spatial_product_riesz_king_county.csv",
    ]
    if all(path.exists() for path in product_paths):
        product = pd.concat(
            [pd.read_csv(path) for path in product_paths], ignore_index=True
        )
        product_wide = product.pivot(
            index=["dataset", "model", "seed"],
            columns="method",
            values="test_loss",
        )
        product_rows: list[dict[str, object]] = []
        for (dataset, model), group in product_wide.groupby(level=[0, 1]):
            row: dict[str, object] = {
                "dataset": dataset,
                "model": model,
                "pairs": int(len(group)),
                "correct_mean_test_loss": float(
                    group["anchor_product_riesz"].mean()
                ),
            }
            for name in [
                "raple_raw",
                "anchor_only",
                "anchor_product_mass",
                "anchor_product_wrong",
                "anchor_product_isospectral",
            ]:
                gains = 100 * (
                    group[name] - group["anchor_product_riesz"]
                ) / group[name]
                row[f"wins_vs_{name}"] = int((gains > 0).sum())
                row[f"gain_vs_{name}_pct"] = float(gains.mean())
            product_rows.append(row)
        product_summary = pd.DataFrame(product_rows)
        product_summary.to_csv(
            RESULTS / "day4_spatial_product_riesz.csv", index=False
        )

    # The first product pilot omitted a joint generalized-frequency scaling.
    # Keep it as a labeled legacy ablation and separate function-space
    # resolution from the calibrated product and geodesic support-graph forms.
    surface_specs = [
        (
            "product", 8, "legacy-none", "center-only",
            "spatial_product_riesz_california_b8.csv",
        ),
        (
            "product", 12, "legacy-none", "center-only",
            "spatial_product_riesz_california.csv",
        ),
        (
            "product", 16, "legacy-none", "center-only",
            "spatial_product_riesz_california_b16.csv",
        ),
        (
            "product", 12, "joint-generalized-median", "center-only",
            "spatial_product_riesz_california_jointnorm.csv",
        ),
        (
            "haversine-knn", 12, "joint-generalized-median", "center-only",
            "spatial_graph_riesz_california.csv",
        ),
        (
            "product", 12, "joint-generalized-median", "empirical-anova",
            "spatial_product_riesz_california_anova.csv",
        ),
        (
            "haversine-knn", 12, "joint-generalized-median", "empirical-anova",
            "spatial_graph_riesz_california_anova.csv",
        ),
        (
            "product", 12, "joint-generalized-median", "empirical-anova",
            "spatial_product_riesz_king_county_anova.csv",
        ),
        (
            "haversine-knn", 12, "joint-generalized-median", "empirical-anova",
            "spatial_graph_riesz_king_county_anova.csv",
        ),
        (
            "product-token", 12, "joint-generalized-median", "empirical-anova",
            "spatial_product_riesz_california_anova_ft.csv",
        ),
        (
            "product-token", 12, "joint-generalized-median", "empirical-anova",
            "spatial_product_riesz_king_county_anova_ft.csv",
        ),
    ]
    surface_rows: list[dict[str, object]] = []
    for family, bins, normalization, projection, filename in surface_specs:
        path = RESULTS / filename
        if not path.exists():
            continue
        raw_surface = pd.read_csv(path)
        dataset = str(raw_surface["dataset"].iloc[0])
        frame = raw_surface.pivot(
            index=["model", "seed"], columns="method", values="test_loss"
        )
        for model in ["all", *sorted(frame.index.get_level_values("model").unique())]:
            group = frame if model == "all" else frame.xs(model, level="model")
            row: dict[str, object] = {
                "stiffness_family": family,
                "dataset": dataset,
                "product_bins": bins,
                "normalization": normalization,
                "interaction_projection": projection,
                "model": model,
                "pairs": int(len(group)),
            }
            for name in [
                "raple_raw",
                "anchor_only",
                "anchor_product_mass",
                "anchor_product_wrong",
                "anchor_product_isospectral",
            ]:
                gains = 100 * (
                    group[name] - group["anchor_product_riesz"]
                ) / group[name]
                row[f"wins_vs_{name}"] = int((gains > 0).sum())
                row[f"gain_vs_{name}_pct"] = float(gains.mean())
            surface_rows.append(row)
    surface_sensitivity = pd.DataFrame(surface_rows)
    if not surface_sensitivity.empty:
        surface_sensitivity.to_csv(
            RESULTS / "day4_spatial_surface_sensitivity.csv", index=False
        )

    selection_specs = [
        ("product", "spatial_product_riesz_california_anova.csv"),
        ("haversine-knn", "spatial_graph_riesz_california_anova.csv"),
        ("product", "spatial_product_riesz_king_county_anova.csv"),
        ("haversine-knn", "spatial_graph_riesz_king_county_anova.csv"),
        ("product-token", "spatial_product_riesz_california_anova_ft.csv"),
        ("product-token", "spatial_product_riesz_king_county_anova_ft.csv"),
    ]
    selection_candidates = [
        "raple_raw", "anchor_only", "anchor_product_mass",
        "anchor_product_riesz",
    ]
    selection_rows: list[dict[str, object]] = []
    for family, filename in selection_specs:
        path = RESULTS / filename
        if not path.exists():
            continue
        raw = pd.read_csv(path)
        dataset = str(raw["dataset"].iloc[0])
        for model in ["all", *sorted(raw["model"].unique())]:
            group = raw if model == "all" else raw.loc[raw["model"].eq(model)]
            group = group.loc[group["method"].isin(selection_candidates)].copy()
            chosen_indices = group.groupby(["model", "seed"])["val_loss"].idxmin()
            chosen = group.loc[chosen_indices].set_index(["model", "seed"])
            test_wide = group.pivot(
                index=["model", "seed"], columns="method", values="test_loss"
            )
            reference = test_wide["raple_raw"].loc[chosen.index]
            selected_test = chosen["test_loss"]
            gains = 100 * (reference - selected_test) / reference
            oracle = test_wide[selection_candidates].min(axis=1).loc[chosen.index]
            regret = 100 * (selected_test - oracle) / oracle
            counts = chosen["method"].value_counts().sort_index()
            selection_rows.append(
                {
                    "dataset": dataset,
                    "stiffness_family": family,
                    "model": model,
                    "cells": int(len(chosen)),
                    "selected_methods": json.dumps(
                        {str(key): int(value) for key, value in counts.items()},
                        sort_keys=True,
                    ),
                    "wins_vs_raw_raple": int((gains > 0).sum()),
                    "mean_gain_vs_raw_raple_pct": float(gains.mean()),
                    "mean_oracle_regret_pct": float(regret.mean()),
                }
            )
    surface_selection = pd.DataFrame(selection_rows)
    if not surface_selection.empty:
        surface_selection.to_csv(
            RESULTS / "day4_spatial_surface_validation_selection.csv", index=False
        )

    strength_specs = [
        (0.3, "spatial_product_riesz_california_anova_tau03.csv"),
        (1.0, "spatial_product_riesz_california_anova.csv"),
        (3.0, "spatial_product_riesz_california_anova_tau3.csv"),
    ]
    strength_rows: list[dict[str, object]] = []
    strength_reference_path = RESULTS / "spatial_product_riesz_california_anova.csv"
    if strength_reference_path.exists():
        strength_reference = pd.read_csv(strength_reference_path).pivot(
            index=["model", "seed"], columns="method", values="test_loss"
        )
        for strength, filename in strength_specs:
            path = RESULTS / filename
            if not path.exists():
                continue
            frame = pd.read_csv(path).pivot(
                index=["model", "seed"], columns="method", values="test_loss"
            )
            for model in [
                "all", *sorted(frame.index.get_level_values("model").unique())
            ]:
                group = frame if model == "all" else frame.xs(model, level="model")
                reference = (
                    strength_reference
                    if model == "all"
                    else strength_reference.xs(model, level="model")
                )
                row: dict[str, object] = {
                    "strength": strength,
                    "model": model,
                    "pairs": int(len(group)),
                }
                controls = {
                    "raple_raw": reference["raple_raw"],
                    "anchor_product_mass": reference["anchor_product_mass"],
                    "anchor_product_wrong": group["anchor_product_wrong"],
                    "anchor_product_isospectral": group[
                        "anchor_product_isospectral"
                    ],
                }
                for name, values in controls.items():
                    gains = 100 * (
                        values - group["anchor_product_riesz"]
                    ) / values
                    row[f"wins_vs_{name}"] = int((gains > 0).sum())
                    row[f"gain_vs_{name}_pct"] = float(gains.mean())
                strength_rows.append(row)
    surface_strength = pd.DataFrame(strength_rows)
    if not surface_strength.empty:
        surface_strength.to_csv(
            RESULTS / "day4_spatial_surface_strength.csv", index=False
        )

    # Empirical-null modes make the original California surface control only
    # finite-support isospectral.  A chart-covariant reference mass completes
    # the nonredundant interaction space; sweep its predeclared mixture weight
    # rather than reporting only the most favorable completion.
    reference_mass_specs = [
        (0.001, "spatial_product_riesz_refmass0001.csv"),
        (0.01, "spatial_product_riesz_refmass001.csv"),
        (0.1, "spatial_product_riesz_refmass01.csv"),
    ]
    reference_mass_rows: list[dict[str, object]] = []
    for weight, filename in reference_mass_specs:
        path = RESULTS / filename
        metadata_path = path.with_suffix(".metadata.json")
        if not path.exists() or not metadata_path.exists():
            continue
        raw = pd.read_csv(path)
        metadata = json.loads(metadata_path.read_text())
        frame = raw.pivot(
            index=["dataset", "model", "seed"],
            columns="method", values="test_metric",
        )
        for dataset, group in frame.groupby(level="dataset"):
            row: dict[str, object] = {
                "dataset": dataset,
                "reference_mass_weight": weight,
                "pairs": int(len(group)),
                "empirical_mass_rank": int(
                    metadata[dataset]["product_empirical_mass_rank"]
                ),
                "reference_mass_rank": int(
                    metadata[dataset]["product_reference_mass_rank"]
                ),
                "completed_mass_rank": int(
                    metadata[dataset]["product_mass_rank"]
                ),
                "function_space_dimension": 144,
                "isospectral_max_abs_error": float(
                    metadata[dataset]["isospectral_max_abs_error"]
                ),
            }
            for name in [
                "anchor_product_mass",
                "anchor_product_wrong",
                "anchor_product_isospectral",
            ]:
                gains = 100 * (
                    group[name] - group["anchor_product_riesz"]
                ) / group[name]
                row[f"wins_vs_{name}"] = int((gains > 0).sum())
                row[f"gain_vs_{name}_pct"] = float(gains.mean())
            reference_mass_rows.append(row)
    surface_reference_mass = pd.DataFrame(reference_mass_rows)
    if not surface_reference_mass.empty:
        surface_reference_mass.to_csv(
            RESULTS / "day4_spatial_reference_mass.csv", index=False
        )

    reference_frames: list[pd.DataFrame] = []
    for weight, filename in reference_mass_specs:
        path = RESULTS / filename
        if path.exists():
            frame = pd.read_csv(path)
            frame["rho"] = weight
            reference_frames.append(frame)
    if reference_frames:
        reference_long = pd.concat(reference_frames, ignore_index=True)
        selection_keys = ["dataset", "model", "seed"]
        reference_selection_rows: list[dict[str, object]] = []
        for dataset, group in reference_long.groupby("dataset"):
            semantic = group.loc[group["method"].eq("anchor_product_riesz")]
            chosen_semantic = semantic.loc[
                semantic.groupby(selection_keys)["val_metric"].idxmin()
            ].set_index(selection_keys)
            control_wide = group.pivot_table(
                index=selection_keys + ["rho"],
                columns="method", values="test_metric",
            )
            semantic_row: dict[str, object] = {
                "dataset": dataset,
                "selector": "semantic-rho-only",
                "cells": int(len(chosen_semantic)),
                "selected": json.dumps(
                    {
                        str(key): int(value)
                        for key, value in chosen_semantic["rho"]
                        .value_counts().sort_index().items()
                    }, sort_keys=True,
                ),
            }
            for name in [
                "anchor_product_mass", "anchor_product_wrong",
                "anchor_product_isospectral",
            ]:
                gains = []
                for index, selected in chosen_semantic.iterrows():
                    control = control_wide.loc[(*index, selected["rho"]), name]
                    gains.append(
                        100 * (control - selected["test_metric"]) / control
                    )
                semantic_row[f"wins_vs_{name}"] = int(
                    sum(value > 0 for value in gains)
                )
                semantic_row[f"gain_vs_{name}_pct"] = float(sum(gains) / len(gains))
            reference_selection_rows.append(semantic_row)

            candidates = group.loc[group["method"].isin([
                "raple_raw", "anchor_only", "anchor_product_mass",
                "anchor_product_riesz",
            ])].copy()
            chosen = candidates.loc[
                candidates.groupby(selection_keys)["val_metric"].idxmin()
            ].set_index(selection_keys)
            raw = group.loc[group["method"].eq("raple_raw")].set_index(
                selection_keys
            )["test_metric"]
            gains = 100 * (raw - chosen["test_metric"]) / raw
            oracle = candidates.groupby(selection_keys)["test_metric"].min()
            regret = 100 * (chosen["test_metric"] - oracle) / oracle
            reference_selection_rows.append(
                {
                    "dataset": dataset,
                    "selector": "raw-anchor-mass-semantic",
                    "cells": int(len(chosen)),
                    "selected": json.dumps(
                        {
                            f"{method}@{rho:g}": int(value)
                            for (method, rho), value in chosen.groupby(
                                ["method", "rho"]
                            ).size().items()
                        }, sort_keys=True,
                    ),
                    "wins_vs_raple_raw": int((gains > 0).sum()),
                    "gain_vs_raple_raw_pct": float(gains.mean()),
                    "mean_oracle_regret_pct": float(regret.mean()),
                }
            )
        surface_reference_selection = pd.DataFrame(reference_selection_rows)
        surface_reference_selection.to_csv(
            RESULTS / "day4_spatial_reference_selection.csv", index=False
        )

    reference_base_path = RESULTS / "spatial_product_riesz_refmass001.csv"
    if reference_base_path.exists():
        reference_base = pd.read_csv(reference_base_path)
        semantic_reference = reference_base.loc[
            reference_base["method"].eq("anchor_product_riesz"),
            ["dataset", "model", "seed", "test_metric"],
        ].rename(columns={"test_metric": "semantic_test_metric"})
        rotation_frames: list[pd.DataFrame] = []
        default_control = reference_base.loc[
            reference_base["method"].eq("anchor_product_isospectral")
        ].copy()
        default_control["control_seed"] = 991337
        rotation_frames.append(default_control)
        for path in sorted(RESULTS.glob(
            "spatial_product_riesz_refmass001_iso_[0-9]*.csv"
        )):
            control = pd.read_csv(path)
            control["control_seed"] = int(path.stem.rsplit("_", 1)[-1])
            rotation_frames.append(control)
        rotations = pd.concat(rotation_frames, ignore_index=True).merge(
            semantic_reference,
            on=["dataset", "model", "seed"], validate="many_to_one",
        )
        rotations["semantic_gain_pct"] = 100 * (
            rotations["test_metric"] - rotations["semantic_test_metric"]
        ) / rotations["test_metric"]
        rotation_rows: list[dict[str, object]] = []
        for (dataset, control_seed), group in rotations.groupby(
            ["dataset", "control_seed"]
        ):
            rotation_rows.append(
                {
                    "dataset": dataset,
                    "control": str(control_seed),
                    "comparisons": int(len(group)),
                    "semantic_wins": int((group["semantic_gain_pct"] > 0).sum()),
                    "mean_semantic_gain_pct": float(
                        group["semantic_gain_pct"].mean()
                    ),
                }
            )
        cell_average = rotations.groupby(
            ["dataset", "model", "seed"], as_index=False
        ).agg(
            semantic_test_metric=("semantic_test_metric", "first"),
            mean_control_test_metric=("test_metric", "mean"),
            controls=("control_seed", "nunique"),
        )
        cell_average["semantic_gain_pct"] = 100 * (
            cell_average["mean_control_test_metric"]
            - cell_average["semantic_test_metric"]
        ) / cell_average["mean_control_test_metric"]
        for dataset, group in cell_average.groupby("dataset"):
            rotation_rows.append(
                {
                    "dataset": dataset,
                    "control": "within-cell-mean",
                    "comparisons": int(len(group)),
                    "semantic_wins": int((group["semantic_gain_pct"] > 0).sum()),
                    "mean_semantic_gain_pct": float(
                        group["semantic_gain_pct"].mean()
                    ),
                }
            )
        surface_reference_rotations = pd.DataFrame(rotation_rows)
        surface_reference_rotations.to_csv(
            RESULTS / "day4_spatial_reference_rotations.csv", index=False
        )

    mixture_path = RESULTS / "spatial_product_riesz_rho_mixture.csv"
    mixture_reference_path = RESULTS / "spatial_product_riesz_refmass001.csv"
    if mixture_path.exists() and mixture_reference_path.exists():
        mixture = pd.read_csv(mixture_path).pivot(
            index=["dataset", "model", "seed"],
            columns="method", values="test_metric",
        )
        mixture_reference = pd.read_csv(mixture_reference_path).pivot(
            index=["dataset", "model", "seed"],
            columns="method", values="test_metric",
        )
        mixture = mixture.join(
            mixture_reference[["raple_raw", "anchor_only", "anchor_product_mass"]],
            how="inner",
        )
        mixture_rows: list[dict[str, object]] = []
        for dataset, group in mixture.groupby(level="dataset"):
            row: dict[str, object] = {
                "dataset": dataset,
                "rho_mixture": "[0.001,0.01,0.1]",
                "pairs": int(len(group)),
            }
            for name in [
                "raple_raw", "anchor_only", "anchor_product_mass",
                "anchor_product_rho_mixture_wrong",
                "anchor_product_rho_mixture_isospectral",
            ]:
                gains = 100 * (
                    group[name] - group["anchor_product_rho_mixture"]
                ) / group[name]
                row[f"wins_vs_{name}"] = int((gains > 0).sum())
                row[f"gain_vs_{name}_pct"] = float(gains.mean())
            mixture_rows.append(row)
        surface_rho_mixture = pd.DataFrame(mixture_rows)
        surface_rho_mixture.to_csv(
            RESULTS / "day4_spatial_rho_mixture.csv", index=False
        )

    # BH on approximate alignment z-scores is reported as a heuristic screen,
    # not as a finite-sample FDR theorem. The independent-calibration
    # certificate in the theory note is the only formal no-harm statement.
    fdr_summary = pd.DataFrame()
    fdr_path = RESULTS / "residual_riesz_fdr10_broad.csv"
    fdr_meta_path = RESULTS / "residual_riesz_fdr10_broad.metadata.json"
    if fdr_path.exists() and fdr_meta_path.exists():
        fdr = pd.read_csv(fdr_path)
        fdr_meta = json.loads(fdr_meta_path.read_text())
        fdr_rows: list[dict[str, object]] = []
        for (dataset, model), frame in fdr.groupby(["dataset", "model"]):
            reference = direct_wide.xs(
                (dataset, model), level=["dataset", "model"]
            )
            correct = frame.loc[
                frame["method"].eq(CORRECT)
            ].set_index("seed")["test_loss"]
            wrong = frame.loc[
                frame["method"].eq("anchor_wrong_representer")
            ].set_index("seed")["test_loss"]
            row: dict[str, object] = {
                "dataset": dataset,
                "model": model,
                "selected_fields": int(len(fdr_meta[dataset]["selected_fields"])),
                "pairs": int(len(correct)),
            }
            for name, values in {
                BASELINE: reference[BASELINE],
                "raple_raw": reference["raple_raw"],
                "anchor_only": reference["anchor_only"],
                "screen_wrong": wrong,
            }.items():
                aligned = correct.to_frame("correct").join(values.rename("control"))
                gains = 100 * (
                    aligned["control"] - aligned["correct"]
                ) / aligned["control"]
                row[f"wins_vs_{name}"] = int((gains > 0).sum())
                row[f"gain_vs_{name}_pct"] = float(gains.mean())
            fdr_rows.append(row)
        fdr_summary = pd.DataFrame(fdr_rows)
        fdr_summary.to_csv(RESULTS / "day4_fdr_selector.csv", index=False)

    synthetic_completion_summary: dict[str, object] = {}
    synthetic_completion_path = (
        RESULTS / "synthetic_reference_completion.summary.json"
    )
    if synthetic_completion_path.exists():
        synthetic_completion_summary = json.loads(
            synthetic_completion_path.read_text()
        )

    summary = {
        "claim": (
            "Treat each scalar field as a measured function space; normalize its "
            "empirical mass and add semantic stiffness only when the schema declares geometry."
        ),
        "portfolio_decision": {
            "primary": "OrbitANOVA",
            "primary_readiness_out_of_5": 3.5,
            "primary_status": (
                "ICLR-shaped, but requires the frozen broad audit and held-out "
                "audit-guided action transfer before submission."
            ),
            "secondary": "FieldRiesz",
            "secondary_status": (
                "High-risk secondary method or targeted chart-covariant intervention "
                "inside OrbitANOVA; not a standalone submission today."
            ),
        },
        "performance_hypothesis": (
            "For cross-fitted anchor residual r, use the chart-invariant field feature "
            "h_j(x)=c_j^T(M_j+tau_j S_j)^(-1)phi_j(x), where "
            "c_j=E[phi_j(X_j)r]. In the complete direct three-seed panel over five "
            "datasets and MLP, ResNet, and TabM, dense semantic Riesz wins 17/45 "
            "paired cells against raw RAPLE (mean -0.15%) but 33/45 against one "
            "fixed permuted operator (+0.57%). On the positive California/Weather "
            "MLP/ResNet subset it wins 52/60 across five node permutations (+1.05%), "
            "or 11/12 unique cells after control averaging, "
            "and, after adding TabM, 80/90 across five exact M-isospectral "
            "rotations (+0.90%), or 18/18 unique cells after averaging their "
            "five controls. Those datasets were selected after the broad pilot, "
            "so 18/18 is stress evidence rather than a confirmatory test. "
            "Strength probes weaken at tau=3. The "
            "predeclared chronological King County spatial replication is "
            "mixed/negative (2/9 vs raw RAPLE; -0.03% vs mean isospectral "
            "control). A purified, jointly calibrated Latitude/Longitude "
            "residual surface sharpens California (8/9 vs raw and its finite-support "
            "control; +2.63% and +0.55%), but that control is rank-confounded. A "
            "reference-mass repair matches all 144 nonredundant modes: California's "
            "semantic control gap is positive only at rho=0.01 and negative at "
            "rho=0.001 and 0.1. Repeating rho=0.01 across five orientations leaves "
            "only +0.04% after within-cell averaging; King County is +0.19% despite "
            "failing wrong geometry. A "
            "selection-free mixture over all three rho values retains California's "
            "raw gain but loses its matched isospectral mixture on average. The "
            "current result is a "
            "selective mechanism signal, not a broadly superior predictor."
        ),
        "california_geometry": records(geometry_summary),
        "field_ablation": records(field_frame),
        "bike_cyclic": records(bike_summary),
        "residual_riesz": records(residual_summary),
        "residual_riesz_pair_wins": residual_pair_wins,
        "residual_vs_raple": records(direct_summary),
        "residual_vs_raple_pair_wins": direct_pair_wins,
        "residual_vs_raple_dataset_model_counts": direct_aggregate_counts,
        "maps_sparse_screens": records(maps_screen_summary),
        "strict_selector": records(strict_summary),
        "wrong_geometry_permutations": records(permutation_summary),
        "wrong_geometry_unique_cells": records(permutation_cell_summary),
        "isospectral_control": records(isospectral_summary),
        "isospectral_rotations": records(isospectral_rotation_summary),
        "isospectral_unique_cells": records(isospectral_cell_summary),
        "semantic_spectral_profile": records(spectral_summary),
        "semantic_localizer": records(localizer_summary),
        "strength_robustness": records(strength_summary),
        "king_county_spatial": records(king_summary),
        "king_county_controls": records(king_control_summary),
        "spatial_product_riesz": records(product_summary),
        "spatial_surface_sensitivity": records(surface_sensitivity),
        "spatial_surface_validation_selection": records(surface_selection),
        "spatial_surface_strength": records(surface_strength),
        "spatial_reference_mass": records(surface_reference_mass),
        "spatial_reference_selection": records(surface_reference_selection),
        "spatial_reference_rotations": records(surface_reference_rotations),
        "synthetic_reference_completion": synthetic_completion_summary,
        "spatial_rho_mixture": records(surface_rho_mixture),
        "fdr_selector": records(fdr_summary),
        "architecture_cells": int(
            architecture[["dataset", "model", "seed"]].drop_duplicates().shape[0]
        ),
        "temporal_nonbaseline_cells": int((~temporal["method"].eq(BASELINE)).sum()),
        "temporal_nonbaseline_test_wins": int(
            ((~temporal["method"].eq(BASELINE)) & (temporal["test_loss_gain_pct"] > 0)).sum()
        ),
        "status": (
            "Differentiated and mathematically coherent secondary direction, but not "
            "enough evidence for an ICLR submission. OrbitANOVA remains the primary "
            "paper; broad raw-RAPLE superiority, selector power, and reliable semantic "
            "replication remain unsatisfied for FieldRiesz."
        ),
    }
    (RESULTS / "day4_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
