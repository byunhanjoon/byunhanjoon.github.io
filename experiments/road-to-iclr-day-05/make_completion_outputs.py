"""Create the six required tables and ten publication figures for Day 5."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from completion_neural_panel import CONFIG, HERE, make_model, prepare, raw_local, raw_openml


RESULTS = HERE / "results"
TABLES = RESULTS / "completion_tables"
FIGURES = RESULTS / "completion_figures"
CONFIG_DATA = json.loads(CONFIG.read_text())
COLORS = {
    "iid": "#7f8c8d", "srswor": "#34495e", "lhs": "#8e44ad", "sobol": "#d35400",
    "strength1": "#3498db", "strength2": "#16a085", "strength3": "#c0392b",
}


def save_table(frame: pd.DataFrame, number: int, name: str) -> None:
    stem = TABLES / f"table{number}_{name}"
    frame.to_csv(stem.with_suffix(".csv"), index=False)
    def cell(value: object) -> str:
        if isinstance(value, (float, np.floating)):
            rendered = f"{float(value):.6g}"
        else:
            rendered = str(value)
        return rendered.replace("|", "\\|").replace("\n", " ")
    lines = [
        "| " + " | ".join(map(str, frame.columns)) + " |",
        "| " + " | ".join("---" for _ in frame.columns) + " |",
    ]
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None))
    stem.with_suffix(".md").write_text("\n".join(lines) + "\n")


def save_figure(fig: plt.Figure, number: int, name: str) -> None:
    stem = FIGURES / f"figure{number:02d}_{name}"
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def dataset_table() -> pd.DataFrame:
    rows = []
    for name in CONFIG_DATA["datasets"]:
        numerical, categorical, target = (
            raw_openml(name, CONFIG_DATA) if name.startswith("openml-") else raw_local(name, CONFIG_DATA)
        )
        cards = []
        for column in range(categorical.shape[1]):
            cards.append(pd.Series(categorical[:, column]).fillna("__MISSING__").astype(str).nunique())
        task = CONFIG_DATA["dataset_tasks"][name]
        rows.append({
            "dataset": name, "task": task, "n": len(target),
            "d": numerical.shape[1] + categorical.shape[1],
            "# numerical": numerical.shape[1], "# categorical": categorical.shape[1],
            "mean category cardinality": float(np.mean(cards)) if cards else 0.0,
            "classes": int(pd.Series(target).nunique()) if task == "classification" else "—",
        })
    return pd.DataFrame(rows)


def model_table() -> pd.DataFrame:
    parameters: dict[str, list[int]] = {name: [] for name in CONFIG_DATA["models"]}
    for dataset in CONFIG_DATA["datasets"]:
        data = prepare(dataset, int(CONFIG_DATA["split_seeds"][0]), CONFIG_DATA)
        width = data.x_num["train"].shape[1] + sum(size + 1 for size in data.cardinalities)
        output = 2 if data.task == "classification" else 1
        for model in CONFIG_DATA["models"]:
            current = make_model(model, width, output, CONFIG_DATA)
            parameters[model].append(sum(value.numel() for value in current.parameters()))
            del current
    descriptions = {
        "mlp": ("20 epochs AdamW", "dense one-hot blocks", 4),
        "resnet": ("20 epochs AdamW", "dense one-hot blocks", 4),
        "ft_transformer": ("20 epochs AdamW", "dense stem + feature tokens", 4),
        "tabm": ("20 epochs AdamW; k=4", "dense stem + internal members", 4),
        "tabpfn": ("pretrained; 1/8 inference members", "native numeric/categorical indices", 1),
        "onehot_linear": ("converged / ridge closed-form", "dense one-hot blocks", 4),
        "native_histgb": ("80 boosting iterations", "native ordinal categories", 4),
        "catboost_native": ("80 boosting iterations", "native categorical strings", 4),
        "xgboost": ("80 boosting iterations", "ordinal numeric input", 4),
        "lightgbm": ("80 boosting iterations", "declared categorical columns", 4),
    }
    rows = []
    for model in (*CONFIG_DATA["models"], "tabpfn", "onehot_linear", "native_histgb", "catboost_native", "xgboost", "lightgbm"):
        if model in parameters:
            values = parameters[model]
            count = str(values[0]) if min(values) == max(values) else f"{min(values)}–{max(values)}"
        elif model == "tabpfn":
            count = "pretrained checkpoint"
        else:
            count = "data-dependent / n.a."
        budget, handling, seeds = descriptions[model]
        rows.append({"model": model, "parameter count": count, "training budget": budget,
                     "schema handling": handling, "seed count": seeds})
    return pd.DataFrame(rows)


def architecture_risk(neural: pd.DataFrame, classical: pd.DataFrame, tabpfn: pd.DataFrame) -> pd.DataFrame:
    columns = ["total_nuisance_variance", "schema_only_variance", "seed_only_variance", "schema_seed_interaction_variance"]
    parts = [
        neural.groupby(["model", "task"])[columns].mean(),
        classical.groupby(["model", "task"])[columns].mean(),
    ]
    if not tabpfn.empty:
        default = tabpfn[(tabpfn.internal_estimators == 8) & (tabpfn.internal_policy == "default")]
        tabrisk = float(default.drop_duplicates(["dataset", "split_seed"]).schema_risk.mean())
        parts.append(pd.DataFrame({
            "total_nuisance_variance": [tabrisk], "schema_only_variance": [tabrisk],
            "seed_only_variance": [0.0], "schema_seed_interaction_variance": [0.0],
        }, index=pd.MultiIndex.from_tuples([("tabpfn", "classification")], names=["model", "task"])))
    frame = pd.concat(parts).reset_index()
    return frame.rename(columns={
        "total_nuisance_variance": "total risk", "schema_only_variance": "schema-only",
        "seed_only_variance": "seed-only", "schema_seed_interaction_variance": "interaction",
    })


def estimator_table(risk: pd.DataFrame) -> pd.DataFrame:
    methods = [
        ("IID", 16, "iid16_residual_mean"), ("SRSWOR", 16, "srswor16_residual_mean"),
        ("LHS", 16, "lhs16_residual_mean"), ("Sobol", 16, "sobol16_residual_mean"),
        ("strength-1", 16, "strength1_16_residual_mean"),
        ("strength-2", 16, "strength2_16_residual_mean"),
        ("strength-3", 64, "strength3_64_residual_mean"),
    ]
    rows = []
    for label, budget, column in methods:
        values = risk[column]
        rows.append({"method": label, "actual budget": budget, "mean residual": values.mean(),
                     "median residual": values.median(),
                     "relative to equal-budget IID": (
                         values.mean() / risk[f"iid{budget}_residual_mean"].mean()
                     )})
    return pd.DataFrame(rows)


def ranking_table(selection: pd.DataFrame) -> pd.DataFrame:
    labels = {
        "iid16": "IID", "srswor16": "SRSWOR", "lhs16": "LHS", "sobol16": "Sobol",
        "strength1_16": "strength-1", "strength2_16": "strength-2", "strength3_64": "strength-3",
    }
    current = selection[selection.method.isin(labels)].copy()
    current["method"] = current.method.map(labels)
    return current.groupby("method", sort=False)[
        ["winner_agreement", "spearman", "pairwise_accuracy", "selected_test_regret"]
    ].mean().reset_index().rename(columns={"selected_test_regret": "test_regret"})


def figures(risk: pd.DataFrame, selection: pd.DataFrame, matched: pd.DataFrame,
            arch: pd.DataFrame, estimator: pd.DataFrame) -> None:
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    groups = [risk.loc[risk.model == model, "total_nuisance_variance"].clip(lower=1e-16) for model in CONFIG_DATA["models"]]
    ax.boxplot(groups, tick_labels=CONFIG_DATA["models"], showfliers=False)
    ax.set_yscale("log"); ax.set_ylabel("prediction-field nuisance variance")
    ax.set_title("Semantically equivalent representations change fitted predictions")
    save_figure(fig, 1, "schema_prediction_changes")

    fractions = risk.groupby("model")[["main_fraction", "triple_fraction", "higher_fraction"]].mean()
    fractions.insert(1, "pair_fraction", risk.groupby("model").main_pair_fraction.mean() - fractions.main_fraction)
    fig, ax = plt.subplots(figsize=(7.2, 3.8)); bottom = np.zeros(len(fractions))
    for column, label in (("main_fraction", "main"), ("pair_fraction", "pair"),
                          ("triple_fraction", "triple"), ("higher_fraction", "higher")):
        ax.bar(fractions.index, fractions[column], bottom=bottom, label=label); bottom += fractions[column].to_numpy()
    ax.set_ylim(0, 1); ax.set_ylabel("fraction of nuisance variance"); ax.legend(ncol=4, frameon=False)
    ax.set_title("Exact prediction fANOVA")
    save_figure(fig, 2, "fanova_decomposition")

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for prefix, label in (("iid", "IID"), ("srswor", "SRSWOR")):
        budgets = [1, 2, 4, 8, 16, 32, 64]
        values = [risk[f"{prefix}{budget}_residual_mean"].mean() for budget in budgets]
        ax.plot(budgets, values, marker="o", label=label, color=COLORS[prefix])
    for label, budget, key in (("LHS", 16, "lhs16"), ("Sobol", 16, "sobol16"),
                               ("strength-1", 16, "strength1_16"), ("strength-2", 16, "strength2_16"),
                               ("strength-3", 64, "strength3_64")):
        ax.scatter(budget, risk[f"{key}_residual_mean"].mean(), s=50, label=label,
                   color=COLORS[label.lower().replace("-", "")])
    ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xlabel("fits"); ax.set_ylabel("mean quotient residual")
    ax.legend(ncol=3, frameon=False, fontsize=8); ax.set_title("Quotient estimation error versus compute")
    save_figure(fig, 3, "quotient_error_budget")

    iid_budgets = np.asarray([1, 2, 4, 8, 16, 32, 64])
    equivalents = []
    for _, row in risk.iterrows():
        curve = np.asarray([row[f"iid{budget}_residual_mean"] for budget in iid_budgets])
        target = row.strength2_16_residual_mean
        below = np.where(curve <= target)[0]
        equivalents.append(float(iid_budgets[below[0]]) if len(below) else 64.0)
    fig, ax = plt.subplots(figsize=(6.2, 3.8)); ax.hist(equivalents, bins=np.arange(0, 72, 8), color="#16a085")
    ax.axvline(16, color="black", linestyle="--", linewidth=1); ax.set_xlabel("IID fits needed to match strength-2 at 16 fits")
    ax.set_ylabel("dataset × split × model cells"); ax.set_title("IID-equivalent compute")
    save_figure(fig, 4, "iid_equivalent_budget")

    order = ["iid16", "srswor16", "lhs16", "sobol16", "strength1_16", "strength2_16", "strength3_64"]
    selected = selection[selection.method.isin(order)].groupby("method").mean(numeric_only=True).reindex(order)
    labels = ["IID", "SRS", "LHS", "Sobol", "S1", "S2", "S3"]
    fig, ax = plt.subplots(figsize=(6.8, 3.8)); ax.bar(labels, selected.spearman, color="#3498db")
    ax.set_ylim(max(0, selected.spearman.min() - .1), 1); ax.set_ylabel("Spearman correlation"); ax.set_title("Model-ranking fidelity")
    save_figure(fig, 5, "ranking_fidelity")

    fig, ax = plt.subplots(figsize=(6.8, 3.8)); ax.bar(labels, selected.selected_test_regret, color="#d35400")
    ax.axhline(0, color="black", linewidth=.8); ax.set_ylabel("selected-test regret"); ax.set_title("Validation selection does not erase partition shift")
    save_figure(fig, 6, "selected_test_regret")

    pooled = matched.groupby("model")[["ordinary_variance", "matched_variance"]].mean()
    fig, ax = plt.subplots(figsize=(7.2, 3.8)); positions = np.arange(len(pooled)); width = .38
    ax.bar(positions - width/2, pooled.ordinary_variance, width, label="ordinary")
    ax.bar(positions + width/2, pooled.matched_variance, width, label="matched function")
    ax.set_xticks(positions, pooled.index); ax.set_yscale("log"); ax.set_ylabel("nuisance variance"); ax.legend(frameon=False)
    ax.set_title("Matched-initial-function control")
    save_figure(fig, 7, "matched_function")

    advantage = np.log10(risk.iid16_residual_mean.clip(lower=1e-18) / risk.strength2_16_residual_mean.clip(lower=1e-18))
    fig, ax = plt.subplots(figsize=(6.2, 4.2)); points = ax.scatter(risk.main_pair_fraction, advantage, c=risk.higher_fraction, cmap="viridis", s=25, alpha=.8)
    ax.axhline(0, color="black", linewidth=.8); ax.set_xlabel("main + pair fANOVA fraction"); ax.set_ylabel("log10 IID / strength-2 residual")
    fig.colorbar(points, ax=ax, label="higher-order fraction"); ax.set_title("Strength-2 gain follows low-order structure")
    save_figure(fig, 8, "gain_vs_interaction_mass")

    fig, ax = plt.subplots(figsize=(9, 4)); ordered = arch[arch.task == "classification"].sort_values("total risk", ascending=False)
    ax.bar(ordered.model, ordered["total risk"].clip(lower=1e-18), color="#2c3e50")
    ax.set_yscale("log"); ax.tick_params(axis="x", rotation=40); ax.set_ylabel("mean nuisance variance")
    ax.set_title("Architecture boundary comparison")
    save_figure(fig, 9, "architecture_comparison")

    phase = pd.read_csv(RESULTS / "interaction_phase_coefficients.csv")
    pivot = phase.pivot(index="order", columns="method", values="ratio_to_equal_budget_iid")
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for method in [name for name in ("strength1", "strength2", "strength3") if name in pivot]:
        ax.plot(pivot.index, pivot[method], marker="o", label=method, color=COLORS[method])
    ax.axhline(1, color="black", linestyle="--", linewidth=.8); ax.set_xlabel("pure interaction order"); ax.set_ylabel("MSE / equal-budget IID")
    ax.legend(frameon=False); ax.set_title("Synthetic failure boundary")
    save_figure(fig, 10, "synthetic_failure_boundary")


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True)
    neural = pd.read_csv(RESULTS / "completion_neural_risk_cells.csv")
    classical = pd.read_csv(RESULTS / "completion_classical_risk.csv")
    selection = pd.read_csv(RESULTS / "completion_neural_selection.csv")
    matched = pd.read_csv(RESULTS / "completion_matched_function.csv")
    tabpfn_path = RESULTS / "completion_tabpfn_external_cells.csv"
    tabpfn = pd.read_csv(tabpfn_path) if tabpfn_path.exists() else pd.DataFrame()
    datasets = dataset_table(); models = model_table(); arch = architecture_risk(neural, classical, tabpfn)
    estimator = estimator_table(neural); ranking = ranking_table(selection)
    matched_table = matched.assign(**{"fraction remaining": 1 - matched.fraction_removed})[
        ["dataset", "model", "ordinary_variance", "matched_variance", "fraction_removed", "fraction remaining"]
    ].rename(columns={"ordinary_variance": "ordinary nuisance variance",
                      "matched_variance": "matched-function nuisance variance"})
    save_table(datasets, 1, "datasets"); save_table(models, 2, "models")
    save_table(arch, 3, "nuisance_risk"); save_table(estimator, 4, "estimator_comparison")
    save_table(ranking, 5, "ranking_model_selection"); save_table(matched_table, 6, "matched_function")
    figures(neural, selection, matched, arch, estimator)
    manifest = {
        "status": "complete", "tables": 6, "figures": 10,
        "figure_files": 20, "table_files": 12,
        "note": "Strength-3 uses its mathematically valid 64-fit mixed-level design; all other Table-4 methods use 16 fits.",
    }
    (RESULTS / "completion_outputs_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
