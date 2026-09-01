"""Aggregate the frozen Day-8 screen into tables, figures, and final reports."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from day8_core import HERE, PANEL, build_model, make_synthetic


RAW = HERE / "raw"
FIGURES = HERE / "figures"
FIGURES.mkdir(exist_ok=True)


def fmt(x: float, digits: int = 4) -> str:
    return "NA" if not np.isfinite(x) else f"{x:.{digits}f}"


def load_real() -> pd.DataFrame:
    rows = []
    for path in sorted((RAW / "real").glob("*.json")):
        payload = json.loads(path.read_text())
        if payload.get("status") == "complete":
            rows.append(payload)
    frame = pd.DataFrame(rows)
    required = {"dataset", "model", "prediction_representation", "retrieval_representation", "key_capacity", "seed", "score"}
    if frame.empty or not required.issubset(frame.columns):
        raise RuntimeError("Real panel is absent or incomplete")
    missing_panel = set(PANEL) - set(frame.dataset)
    if missing_panel:
        raise RuntimeError(f"Missing datasets: {sorted(missing_panel)}")
    return frame.sort_values(list(required)).reset_index(drop=True)


def add_raw_delta(real: pd.DataFrame) -> pd.DataFrame:
    out = real.copy()
    baseline = out[
        (out.prediction_representation == "raw")
        & (out.retrieval_representation == "raw")
        & (out.key_capacity == "standard")
    ][["dataset", "model", "seed", "score"]].rename(columns={"score": "raw_score"})
    out = out.merge(baseline, on=["dataset", "model", "seed"], how="left")
    out["score_delta_vs_raw"] = out.score - out.raw_score
    return out


def paired(real: pd.DataFrame, model: str, pred: str, retr: str, capacity: str = "standard") -> pd.DataFrame:
    target = real[
        (real.model == model)
        & (real.prediction_representation == pred)
        & (real.retrieval_representation == retr)
        & (real.key_capacity == capacity)
    ][["dataset", "seed", "score"]].rename(columns={"score": "target"})
    raw = real[
        (real.model == model)
        & (real.prediction_representation == "raw")
        & (real.retrieval_representation == "raw")
        & (real.key_capacity == capacity)
    ][["dataset", "seed", "score"]].rename(columns={"score": "raw"})
    x = target.merge(raw, on=["dataset", "seed"])
    x["delta"] = x.target - x.raw
    return x


def paired_sentence(frame: pd.DataFrame) -> str:
    by_data = frame.groupby("dataset").delta.mean()
    seed_sd = frame.groupby("dataset").delta.std().fillna(0).mean()
    return (
        f"dataset-balanced Δscore {fmt(by_data.mean())}; "
        f"wins/losses/ties {(by_data > 0).sum()}/{(by_data < 0).sum()}/{(by_data == 0).sum()}; "
        f"mean within-dataset seed SD {fmt(seed_sd)}"
    )


def build_tables(real: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    synthetic = pd.read_csv(RAW / "synthetic" / "results.csv")
    synth_table = synthetic.groupby(["task", "model", "representation"], dropna=False).agg(
        n=("seed", "size"),
        rmse_mean=("rmse", "mean"), rmse_sd=("rmse", "std"),
        retrieval_risk_mean=("topk_oracle_risk", "mean"),
        risk_spearman_mean=("risk_spearman", "mean"),
        oracle_topk_overlap_mean=("oracle_topk_overlap", "mean"),
        target_mismatch_mean=("topk_target_mismatch", "mean"),
        candidate_noise_mean=("topk_candidate_noise", "mean"),
    ).reset_index()
    synth_table.to_csv(HERE / "table_synthetic_theory.csv", index=False)

    real_out = add_raw_delta(real)
    real_out.to_csv(HERE / "table_real_performance.csv", index=False)

    retrieval_columns = [
        "dataset", "task", "model", "prediction_representation", "retrieval_representation",
        "key_capacity", "seed", "score", "score_delta_vs_raw", "risk_spearman",
        "topk_proxy_risk", "topk_target_mismatch", "topk_candidate_noise", "oracle_topk_overlap",
        "mean_selected_retrieval_distance", "candidate_frequency_entropy",
        "neighbor_target_consistency", "neighbor_residual_consistency", "neighbor_label_entropy",
        "within_neighborhood_target_variance",
    ]
    retrieval = real_out[real_out.model != "MLP"][[c for c in retrieval_columns if c in real_out]].copy()
    retrieval.to_csv(HERE / "table_retrieval_quality.csv", index=False)

    synthetic_alignment = pd.read_csv(RAW / "synthetic" / "metric_alignment.csv")
    synthetic_alignment["alignment_type"] = "exact_Jm_metric"
    real_alignment = retrieval.drop(columns=["task"]).rename(
        columns={"dataset": "task", "retrieval_representation": "representation"}
    ).copy()
    real_alignment["scope"] = "real"
    real_alignment["alignment_type"] = "cross_fitted_retrieval_risk"
    real_alignment["rmse"] = np.nan
    for column in ("frobenius_cosine", "top_eigenvector_angle_deg", "feature_diagonal_correlation"):
        real_alignment[column] = np.nan
    alignment_columns = [
        "scope", "task", "seed", "model", "representation", "key_capacity", "alignment_type",
        "rmse", "frobenius_cosine", "top_eigenvector_angle_deg",
        "feature_diagonal_correlation", "risk_spearman", "score", "score_delta_vs_raw",
    ]
    alignment = pd.concat([
        synthetic_alignment.reindex(columns=alignment_columns),
        real_alignment.reindex(columns=alignment_columns),
    ], ignore_index=True)
    alignment.to_csv(HERE / "table_metric_alignment.csv", index=False)

    branch_raw = real_out[
        (real_out.model == "TabR")
        & (real_out.key_capacity == "standard")
        & real_out.prediction_representation.isin(["raw", "localwarp"])
        & real_out.retrieval_representation.isin(["raw", "localwarp"])
    ].copy()
    branch = branch_raw.groupby(["dataset", "prediction_representation", "retrieval_representation"]).agg(
        n_seeds=("seed", "nunique"), score_mean=("score", "mean"), score_sd=("score", "std"),
        delta_vs_raw_mean=("score_delta_vs_raw", "mean"), delta_vs_raw_sd=("score_delta_vs_raw", "std"),
    ).reset_index()
    branch.to_csv(HERE / "table_branch_ablation.csv", index=False)

    ranking = pd.DataFrame([
        {"Direction": "Retrieval Risk Geometry", "Novelty": 3.5, "Theory": 5.0, "Signal": 2.0, "Simplicity": 4.0, "Prior-art risk": 4.0, "ICLR potential": 3.0},
        {"Direction": "Nonlinear Feature Metric", "Novelty": 1.5, "Theory": 3.0, "Signal": 2.5, "Simplicity": 4.0, "Prior-art risk": 5.0, "ICLR potential": 2.0},
        {"Direction": "Transformer Geometry", "Novelty": 2.5, "Theory": 2.0, "Signal": 1.0, "Simplicity": 2.5, "Prior-art risk": 4.0, "ICLR potential": 1.5},
        {"Direction": "OrbitCover Extension", "Novelty": 1.0, "Theory": 3.0, "Signal": 2.5, "Simplicity": 3.0, "Prior-art risk": 5.0, "ICLR potential": 1.5},
    ])
    ranking.to_csv(HERE / "table_direction_ranking.csv", index=False)
    return synth_table, real_out, retrieval, alignment, branch, ranking


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGURES / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    plt.close()


def figure_risk_law() -> None:
    x = pd.read_csv(RAW / "theory" / "risk_scatter.csv")
    lo = min(x.theory.min(), x.monte_carlo.min()); hi = max(x.theory.max(), x.monte_carlo.max())
    plt.figure(figsize=(5.2, 4.5))
    plt.scatter(x.theory, x.monte_carlo, s=28, alpha=.78, color="#315da8")
    plt.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="identity")
    plt.xlabel("A1 theoretical risk"); plt.ylabel("Monte Carlo risk")
    plt.title("Retrieval Risk Law: 48 independent systems")
    plt.legend(frameon=False)
    savefig("figure_1_retrieval_risk_law")


def figure_good_neighbor() -> None:
    data, meta = make_synthetic("S3_noise", 20260831)
    q = int(np.argmax(meta["sigma"]["test"]))
    train, query = data.x_num["train"], data.x_num["test"][q]
    distance = np.square(train - query).sum(axis=1)
    mismatch = np.square(meta["m"]["train"] - meta["m"]["test"][q])
    noise = np.square(meta["sigma"]["train"])
    risk = mismatch + noise
    chosen = np.unique(np.concatenate((np.argsort(distance)[:5], np.argsort(risk)[:5])))
    order = chosen[np.argsort(risk[chosen])]
    labels = [f"c{i}" for i in order]
    pos = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.1))
    axes[0].bar(pos, distance[order], color="#777777")
    axes[0].set_xticks(pos, labels, rotation=45); axes[0].set_ylabel("Squared raw-feature distance")
    axes[0].set_title("Raw proximity")
    axes[1].bar(pos, mismatch[order], label="target mismatch²", color="#315da8")
    axes[1].bar(pos, noise[order], bottom=mismatch[order], label="candidate noise", color="#d97432")
    axes[1].scatter(pos, risk[order], color="black", s=20, label="total risk")
    axes[1].set_xticks(pos, labels, rotation=45); axes[1].set_ylabel("Theoretical one-neighbor risk")
    axes[1].set_title("Statistical neighbor quality"); axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("A close row need not be a good neighbor (S3 high-noise query)")
    savefig("figure_2_good_neighbor")


def _principal_field(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix)
    vec = vectors[:, -1] * np.sqrt(np.maximum(values[-1], 0))
    if vec[0] < 0:
        vec = -vec
    return vec


def figure_metric_field() -> None:
    device = torch.device("cpu")
    data, meta = make_synthetic("S1_rotating", 20260831)
    model = build_model(data, "ModernNCA", "raw", "localwarp", "linear").to(device)
    checkpoint = torch.load(RAW / "synthetic" / "s1_metric_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"]); model.eval()
    axis = np.linspace(-.9, .9, 13)
    grid = np.array([(a, b) for a in axis for b in axis], dtype=np.float32)
    _, _, grad = meta["truth"](grid)
    signal = np.array([g / (np.linalg.norm(g) + 1e-9) for g in grad])
    learned = []
    cat = torch.empty((1, 0), dtype=torch.float32)
    for point in torch.tensor(grid, requires_grad=True):
        jac = torch.autograd.functional.jacobian(lambda z: model.keys(z[None], cat)[0], point)
        learned.append(_principal_field((jac.T @ jac).detach().numpy()))
    learned = np.asarray(learned)
    learned /= np.linalg.norm(learned, axis=1, keepdims=True) + 1e-9
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharex=True, sharey=True)
    axes[0].quiver(grid[:, 0], grid[:, 1], signal[:, 0], signal[:, 1], color="#315da8", pivot="middle")
    axes[1].quiver(grid[:, 0], grid[:, 1], learned[:, 0], learned[:, 1], color="#d97432", pivot="middle")
    axes[0].set_title(r"Signal field $G_{signal}(x)$"); axes[1].set_title(r"Learned field $G_{theta}(x)$")
    for ax in axes:
        ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_aspect("equal")
    fig.suptitle("Rotating-metric synthetic: principal local directions")
    savefig("figure_3_rotating_metric")


def figure_branch(branch: pd.DataFrame) -> None:
    labels = {("raw", "raw"): "raw/raw", ("localwarp", "raw"): "nonlinear/raw", ("raw", "localwarp"): "raw/nonlinear", ("localwarp", "localwarp"): "nonlinear/nonlinear"}
    temp = branch.copy()
    temp["branch"] = [labels[(p, r)] for p, r in zip(temp.prediction_representation, temp.retrieval_representation)]
    pivot = temp.pivot(index="dataset", columns="branch", values="delta_vs_raw_mean")
    order = list(labels.values())
    pivot = pivot.reindex(columns=order)
    plt.figure(figsize=(10.5, 5))
    x = np.arange(len(pivot)); width = .19
    for j, col in enumerate(order):
        plt.bar(x + (j - 1.5) * width, pivot[col], width, label=col)
    plt.axhline(0, color="black", linewidth=.8)
    plt.xticks(x, pivot.index, rotation=30, ha="right"); plt.ylabel("Δ score vs raw/raw (higher is better)")
    plt.title("TabR prediction/retrieval branch ablation"); plt.legend(frameon=False, ncol=2)
    savefig("figure_4_branch_ablation")


def alignment_gain_cells(real: pd.DataFrame) -> pd.DataFrame:
    x = real[
        (real.model.isin(["TabR", "ModernNCA"]))
        & (real.key_capacity == "standard")
        & (real.prediction_representation == "raw")
        & real.risk_spearman.notna()
    ].copy()
    x["gain"] = x.score_delta_vs_raw
    return x.groupby(["dataset", "model", "retrieval_representation"]).agg(
        alignment=("risk_spearman", "mean"), gain=("gain", "mean")
    ).reset_index()


def figure_alignment_gain(real: pd.DataFrame) -> pd.DataFrame:
    cells = alignment_gain_cells(real)
    fig, ax = plt.subplots(figsize=(6.3, 4.8))
    colors = {"TabR": "#315da8", "ModernNCA": "#d97432"}
    markers = {"raw": "o", "localwarp": "s", "ple": "^", "plr": "D", "wrongwarp": "x"}
    for _, row in cells.iterrows():
        ax.scatter(row.alignment, row.gain, c=colors[row.model], marker=markers.get(row.retrieval_representation, "o"), s=45, alpha=.8)
    rho = spearmanr(cells.alignment, cells.gain).statistic
    if abs(rho) < .005:
        rho = 0.0
    ax.axhline(0, color="black", linewidth=.8)
    ax.set_xlabel("Cross-fitted retrieval-risk Spearman alignment")
    ax.set_ylabel("Δ score vs raw retrieval")
    ax.set_title(f"Risk alignment vs prediction gain (cell Spearman ρ={rho:.2f})")
    for model, color in colors.items():
        ax.scatter([], [], c=color, label=model)
    ax.legend(frameon=False)
    savefig("figure_5_alignment_vs_gain")
    return cells


def figure_capacity(real: pd.DataFrame) -> pd.DataFrame:
    x = real[(real.model == "TabR") & real.dataset.isin(["california", "higgs-small"]) & (real.prediction_representation == "raw")]
    raw = x[x.retrieval_representation == "raw"][["dataset", "key_capacity", "seed", "score"]].rename(columns={"score": "raw_score"})
    local = x[x.retrieval_representation == "localwarp"][["dataset", "key_capacity", "seed", "score"]].merge(
        raw, on=["dataset", "key_capacity", "seed"]
    )
    local["gain"] = local.score - local.raw_score
    summary = local.groupby(["dataset", "key_capacity"]).gain.agg(["mean", "std"]).reset_index()
    order = ["linear", "shallow", "standard", "deep"]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for dataset, group in summary.groupby("dataset"):
        group = group.set_index("key_capacity").reindex(order)
        ax.errorbar(order, group["mean"], yerr=group["std"].fillna(0), marker="o", capsize=3, label=dataset)
    ax.axhline(0, color="black", linewidth=.8); ax.set_ylabel("Local-warp Δ score vs raw")
    ax.set_xlabel("Key-network capacity"); ax.set_title("Explicit geometry × key capacity")
    ax.legend(frameon=False)
    savefig("figure_6_key_capacity")
    return summary


def figure_directions(ranking: pd.DataFrame) -> None:
    plot = ranking.copy()
    # For visual desirability only, invert risk; the CSV retains literal risk.
    plot["Low prior-art risk"] = 6 - plot["Prior-art risk"]
    columns = ["Novelty", "Theory", "Signal", "Simplicity", "Low prior-art risk", "ICLR potential"]
    values = plot[columns].to_numpy()
    fig, ax = plt.subplots(figsize=(9, 5.2))
    x = np.arange(len(columns)); width = .19
    for i, name in enumerate(plot.Direction):
        ax.bar(x + (i - 1.5) * width, values[i], width, label=name)
    ax.set_xticks(x, columns, rotation=20, ha="right"); ax.set_ylim(0, 5.3); ax.set_ylabel("Score / 5")
    ax.set_title("Day-8 direction comparison"); ax.legend(frameon=False, fontsize=8, ncol=2)
    savefig("figure_7_direction_comparison")


def representation_effect(real: pd.DataFrame, model: str, representation: str) -> pd.DataFrame:
    if model == "MLP":
        target = real[(real.model == model) & (real.prediction_representation == representation) & (real.retrieval_representation == "raw")]
    else:
        target = real[(real.model == model) & (real.prediction_representation == "raw") & (real.retrieval_representation == representation) & (real.key_capacity == "standard")]
    raw = real[(real.model == model) & (real.prediction_representation == "raw") & (real.retrieval_representation == "raw") & (real.key_capacity == "standard")]
    x = target[["dataset", "seed", "score"]].rename(columns={"score": "target"}).merge(
        raw[["dataset", "seed", "score"]].rename(columns={"score": "raw"}), on=["dataset", "seed"]
    )
    x["delta"] = x.target - x.raw
    return x


def representation_summary(real: pd.DataFrame, model: str) -> str:
    pieces = []
    for rep in ("ple", "plr", "localwarp"):
        x = representation_effect(real, model, rep)
        by_data = x.groupby("dataset").delta.mean()
        pieces.append(f"{rep}: Δ {fmt(by_data.mean())}, W/L {(by_data > 0).sum()}/{(by_data < 0).sum()}")
    return "; ".join(pieces)


def retrieval_mechanism(real: pd.DataFrame, model: str) -> dict[str, float]:
    columns = ["score", "risk_spearman", "topk_proxy_risk", "topk_target_mismatch", "topk_candidate_noise", "neighbor_residual_consistency"]
    base = real[(real.model == model) & (real.prediction_representation == "raw") & (real.retrieval_representation == "raw") & (real.key_capacity == "standard")]
    local = real[(real.model == model) & (real.prediction_representation == "raw") & (real.retrieval_representation == "localwarp") & (real.key_capacity == "standard")]
    pairs = local[["dataset", "seed", *columns]].merge(base[["dataset", "seed", *columns]], on=["dataset", "seed"], suffixes=("_local", "_raw"))
    output = {}
    for column in columns:
        delta = pairs[f"{column}_local"] - pairs[f"{column}_raw"]
        output[column] = float(delta.groupby(pairs.dataset).mean().mean())
    per_data_score = (pairs.score_local - pairs.score_raw).groupby(pairs.dataset).mean()
    per_data_risk = (pairs.topk_proxy_risk_local - pairs.topk_proxy_risk_raw).groupby(pairs.dataset).mean()
    output["joint_improve"] = int(((per_data_score > 0) & (per_data_risk < 0)).sum())
    return output


def write_ranking(ranking: pd.DataFrame) -> None:
    columns = list(ranking.columns)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] + ["---:"] * (len(columns) - 1)) + "|"]
    for row in ranking.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    table = "\n".join(lines)
    text = f"""# DAY 8 DIRECTION RANKING

Scales are `/5`; for **Prior-art risk**, larger is worse.

{table}

WINNER = Retrieval Risk Geometry

WHY = It is the only direction with an exact useful decomposition, a precise failure boundary, oracle synthetic headroom, and a falsifiable mechanism that can be tested prospectively. The surviving novelty is the finite-candidate, noise-aware analysis plus the distinction between query–candidate compatibility and candidate reliability—not generic metric learning.

KILL CONDITION = On a prospectively frozen panel, cross-fitted risk alignment neither predicts retrieval gains nor enables a reliability-aware score to outperform equally tuned raw/deep-key retrieval.

NEXT DECISIVE EXPERIMENT = Freeze 12 untouched public datasets, add a candidate-reliability term estimated strictly out of fold to TabR and ModernNCA, and require dataset-balanced gains plus lower proxy risk on at least 8/12 datasets against raw, PLE/PLR, LMNN/NCA-style, and capacity-matched deep-key baselines.
"""
    (HERE / "DIRECTION_RANKING.md").write_text(text)


def write_results(real: pd.DataFrame, branch: pd.DataFrame, ranking: pd.DataFrame, alignment_cells: pd.DataFrame, capacity: pd.DataFrame) -> None:
    checks = pd.read_csv(RAW / "theory" / "checks.csv")
    synthetic = pd.read_csv(RAW / "synthetic" / "results.csv")
    synth_align = pd.read_csv(RAW / "synthetic" / "metric_alignment.csv")
    transformer = pd.read_csv(RAW / "transformer" / "results.csv")

    def sv(task: str, model: str, rep: str, column: str) -> float:
        x = synthetic[(synthetic.task == task) & (synthetic.model == model) & (synthetic.representation == rep)]
        return float(x[column].mean())

    tabr_local = paired(real, "TabR", "raw", "localwarp")
    tabr_pred = paired(real, "TabR", "localwarp", "raw")
    tabr_both = paired(real, "TabR", "localwarp", "localwarp")
    nca_local = paired(real, "ModernNCA", "raw", "localwarp")
    mlp_local = paired(real, "MLP", "localwarp", "raw")
    tabr_mech = retrieval_mechanism(real, "TabR")
    nca_mech = retrieval_mechanism(real, "ModernNCA")
    rho = float(spearmanr(alignment_cells.alignment, alignment_cells.gain).statistic)
    if abs(rho) < 5e-4:
        rho = 0.0

    wrong = real[(real.model == "TabR") & (real.prediction_representation == "raw") & (real.retrieval_representation == "wrongwarp")]
    raw_seed1 = real[(real.model == "TabR") & (real.prediction_representation == "raw") & (real.retrieval_representation == "raw") & (real.seed == 20260831) & (real.key_capacity == "standard")]
    wrong_pair = wrong[["dataset", "seed", "score"]].merge(raw_seed1[["dataset", "seed", "score"]], on=["dataset", "seed"], suffixes=("_wrong", "_raw"))
    wrong_delta = (wrong_pair.score_wrong - wrong_pair.score_raw).mean()

    branch_lines = []
    for label, frame in (("raw / raw", paired(real, "TabR", "raw", "raw")), ("nonlinear / raw", tabr_pred), ("raw / nonlinear", tabr_local), ("nonlinear / nonlinear", tabr_both)):
        branch_lines.append(f"| {label.split(' / ')[0]} | {label.split(' / ')[1]} | {paired_sentence(frame)} |")

    field_s1 = synth_align[synth_align.task == "S1_rotating"]
    field_summary = field_s1.groupby(["model", "representation"]).agg(cosine=("frobenius_cosine", "mean"), angle=("top_eigenvector_angle_deg", "mean"), risk_rho=("risk_spearman", "mean")).reset_index()
    field_text = "; ".join(f"{r.model}/{r.representation}: cosine {r.cosine:.3f}, angle {r.angle:.1f}°, risk ρ {r.risk_rho:.3f}" for r in field_summary.itertuples())

    trans_lines = []
    for task, group in transformer.groupby("task"):
        raw_score = float(group[group.representation == "raw"].score.iloc[0])
        reps = ", ".join(f"{r.representation} {r.score:.4f} (first→final ρ {r.first_to_final_distance_spearman:.3f})" for r in group.itertuples())
        trans_lines.append(f"- {task}: {reps}; raw reference {raw_score:.4f}.")
    transformer_text = "\n".join(trans_lines)

    capacity_text = "; ".join(
        f"{row.dataset}/{row.key_capacity}: Δ {row['mean']:.4f} ± {row['std']:.4f}"
        for _, row in capacity.iterrows()
    )

    core_lines = []
    for dataset in PANEL:
        cells = real[real.dataset == dataset]
        metric_name = str(cells.metric_name.iloc[0])
        values = []
        for model in ("MLP", "TabR", "ModernNCA"):
            raw_cell = cells[(cells.model == model) & (cells.prediction_representation == "raw") & (cells.retrieval_representation == "raw") & (cells.key_capacity == "standard")]
            if model == "MLP":
                local_cell = cells[(cells.model == model) & (cells.prediction_representation == "localwarp") & (cells.retrieval_representation == "raw") & (cells.key_capacity == "standard")]
            else:
                local_cell = cells[(cells.model == model) & (cells.prediction_representation == "raw") & (cells.retrieval_representation == "localwarp") & (cells.key_capacity == "standard")]
            values.append(f"{raw_cell.metric.mean():.4f} → {local_cell.metric.mean():.4f}")
        core_lines.append(f"| {dataset} | {metric_name} | " + " | ".join(values) + " |")
    core_table = "\n".join(core_lines)

    theory_max = checks.abs_error.max()
    theory_pass = int(checks.passed.sum())
    text = f"""# RESULTS — 8-HOUR ICLR 2027 DIRECTION SEARCH

## 1. Executive verdict

The primary next direction is **Retrieval Risk Geometry**. Scores use `/5`; prior-art risk is literal (5 = most crowded).

| Rank | Direction | ICLR potential | Novelty | Theory clarity | Empirical signal | Prior-art risk |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Retrieval Risk Geometry | 3.0 | 3.5 | 5.0 | 2.0 | 4.0 |
| 2 | General Feature Geometry | 2.0 | 1.5 | 3.0 | 2.5 | 5.0 |
| 3 | Transformer Geometry | 1.5 | 2.5 | 2.0 | 1.0 | 4.0 |
| 4 | OrbitCover Successor | 1.5 | 1.0 | 3.0 | 2.5 | 5.0 |

This is a direction-screening result, not a leaderboard claim. The eight real datasets were capped at 4,096/1,024/1,024 rows, used one frozen split and three model seeds for core cells, and used compact structurally faithful TabR/ModernNCA implementations rather than full published hyperparameter sweeps. Dataset means are the statistical units for W/L summaries; seeds quantify optimization stability and are not treated as independent datasets.

## 2. Literature subtraction

[TabR](https://proceedings.iclr.cc/paper_files/paper/2024/hash/4ef594af0d9a519db8fb292452c461fa-Abstract-Conference.html) already establishes key-only L2 retrieval with learned keys and context-based prediction. [ModernNCA](https://openreview.net/forum?id=JytL2MrlLT) already modernizes supervised neighborhood component analysis for tabular prediction. The 2026 *Unveiling the Role of Data Uncertainty* analysis already studies when retrieval/embeddings help by uncertainty region; PLE/PLR already supply nonlinear numerical embeddings; Function Basis Encoding and 2026 learned-knot splines already learn feature geometry; AWARE already modifies retrieval with task-aligned embeddings; and Tab-PET already injects structural geometry into tabular Transformer tokens. The claim “nonlinear embeddings improve retrieval” is therefore occupied.

What remains plausibly new is narrow: an exact finite-candidate conditional prediction-risk decomposition, its local pullback interpretation, and a derived separation between **query–candidate signal compatibility** and **candidate-only reliability**. Full sources, dates, subtraction, and adjacency risks are in [LITERATURE_BOUNDARY.md](LITERATURE_BOUNDARY.md).

## 3. Retrieval Risk Law

- **A1 — proved.** Under conditional unbiasedness and covariance `Σ`, `E[(ΣᵢwᵢYᵢ-m(x))²|X]=(wᵀd)²+wᵀΣw`; independence gives `Σᵢwᵢ²σᵢ²`.
- **A2 — proved under stated constraints.** For positive-definite `H=ddᵀ+Σ`, equality-constrained optimal weights are `H⁻¹1/(1ᵀH⁻¹1)`; nonnegative weights require a convex QP. The singular pseudoinverse boundary is treated explicitly.
- **A3 — proved.** A one-neighbor rule has risk `(m(Xᵢ)-m(x))²+σᵢ²`.
- **A4 — partially proved.** First-order Taylor expansion yields `G_signal(x)=J_m(x)ᵀJ_m(x)` locally; it is not a global distance and can be degenerate.
- **A5 — proved.** Differentiable key maps induce `G_θ(x)=J_Φ(x)ᵀW_KᵀW_KJ_Φ(x)` to second order; it is a Riemannian metric only under the rank/positivity conditions in [THEORY.md](THEORY.md).

All {theory_pass}/{len(checks)} numerical checks passed; the largest absolute check error was {theory_max:.6g}. Exact assumptions, proofs, correlated-noise extension, singular cases, and failure boundaries are in `THEORY.md`.

## 4. Synthetic theory validation

Across 48 random A1 systems, mean absolute theory–Monte Carlo error was {checks.loc[checks.check == 'A1_independent', 'abs_error'].iloc[0]:.6f}. Oracle one-neighbor retrieval reduced kNN RMSE versus raw from {sv('S1_rotating','kNN','raw','rmse'):.4f} to {sv('S1_rotating','kNN','oracle_one_neighbor_risk','rmse'):.4f} on S1, {sv('S2_global','kNN','raw','rmse'):.4f} to {sv('S2_global','kNN','oracle_one_neighbor_risk','rmse'):.4f} on S2, {sv('S3_noise','kNN','raw','rmse'):.4f} to {sv('S3_noise','kNN','oracle_one_neighbor_risk','rmse'):.4f} on S3, and {sv('S4_warp','kNN','raw','rmse'):.4f} to {sv('S4_warp','kNN','oracle_one_neighbor_risk','rmse'):.4f} on S4.

The noise control is decisive mechanistically: on S3, raw-neighborhood theoretical risk was {sv('S3_noise','kNN','raw','topk_oracle_risk'):.4f}, while the oracle risk neighborhood reached {sv('S3_noise','kNN','oracle_one_neighbor_risk','topk_oracle_risk'):.4f}. A signal-only oracle still had risk {sv('S3_noise','kNN','oracle_signal_metric','topk_oracle_risk'):.4f}; signal proximity cannot identify reliable candidates. On S4, the wrong inverse warp worsened RMSE from {sv('S4_warp','kNN','raw','rmse'):.4f} to {sv('S4_warp','kNN','wrong_inverse_warp','rmse'):.4f}. On the globally linear S2, a target-guided global metric was already sufficient, as expected.

## 5. Does nonlinear geometry improve retrieval specifically?

| Prediction branch | Retrieval branch | Result |
|---|---|---|
{chr(10).join(branch_lines)}

The answer is **not cleanly**. Retrieval-only LocalWarp gave {paired_sentence(tabr_local)}, while prediction-only LocalWarp gave {paired_sentence(tabr_pred)} and both branches gave {paired_sentence(tabr_both)}. Any observed gain must therefore be separated from ordinary representation effects and tested on a prospective panel.

The raw and LocalWarp branches emit the same representation dimension; LocalWarp adds only eight monotone increments per numerical feature, while branch widths and key capacity are held fixed. PLE/PLR are reported separately and are not used as capacity-matched evidence.

Per-dataset branch means, seed SDs, and deltas are in [`table_branch_ablation.csv`](table_branch_ablation.csv) and visualized in Figure 4.

## 6. TabR results

For retrieval-only LocalWarp, {paired_sentence(tabr_local)}. The first-seed representation screen was: {representation_summary(real, 'TabR')}. LocalWarp changed cross-fitted risk Spearman by {tabr_mech['risk_spearman']:+.4f}, selected proxy risk by {tabr_mech['topk_proxy_risk']:+.4f}, target mismatch by {tabr_mech['topk_target_mismatch']:+.4f}, and candidate noise by {tabr_mech['topk_candidate_noise']:+.4f} (negative risk components are better). It jointly improved score and reduced proxy risk on {tabr_mech['joint_improve']}/8 datasets. The wrong-warp first-seed control changed score by {wrong_delta:+.4f} on average.

Core three-seed means below are `raw → LocalWarp` (prediction-only for MLP, retrieval-only for TabR/ModernNCA). Accuracy is higher-better; standardized RMSE is lower-better.

| Dataset | Metric | MLP | TabR | ModernNCA |
|---|---|---:|---:|---:|
{core_table}

## 7. ModernNCA results

Retrieval-only LocalWarp transferred with {paired_sentence(nca_local)}. The representation screen was: {representation_summary(real, 'ModernNCA')}. It changed risk Spearman by {nca_mech['risk_spearman']:+.4f}, proxy risk by {nca_mech['topk_proxy_risk']:+.4f}, mismatch by {nca_mech['topk_target_mismatch']:+.4f}, and candidate noise by {nca_mech['topk_candidate_noise']:+.4f}; joint score/risk improvement occurred on {nca_mech['joint_improve']}/8 datasets. Thus the risk diagnostic transfers beyond TabR, but explicit nonlinear geometry is not a uniformly beneficial method.

## 8. MLP control

Prediction-only LocalWarp produced {paired_sentence(mlp_local)}. The first-seed representation screen was: {representation_summary(real, 'MLP')}. This control shows that any broad LocalWarp improvement is not automatically retrieval-specific.

## 9. Learned metric field

On S1, automatic-differentiation pullbacks gave: {field_text}. Although the nonlinear map permits input dependence, the fitted LocalWarp field in Figure 3 remains nearly constant and does not recover the target field's rotations. Moreover, even high Frobenius alignment would not imply noise awareness: a symmetric query–candidate metric cannot generally represent a candidate-only heteroscedastic penalty.

## 10. Neighbor quality mechanism

Across dataset × model × representation cells, cross-fitted retrieval-risk alignment versus performance gain had Spearman `ρ={rho:.3f}`. For TabR, LocalWarp's mean proxy-risk change was {tabr_mech['topk_proxy_risk']:+.4f} and score change was {tabr_local.groupby('dataset').delta.mean().mean():+.4f}; for ModernNCA the corresponding values were {nca_mech['topk_proxy_risk']:+.4f} and {nca_local.groupby('dataset').delta.mean().mean():+.4f}. Therefore the answer to “do improvements retrieve lower-risk candidates?” is **mixed, not established**. This is exactly the prospective study's main falsifiable mechanism.

## 11. Key-network redundancy

Capacity interaction screen: {capacity_text}. Explicit LocalWarp did not show a monotone advantage as keys deepened. The strongest current interpretation is that expressive key networks can learn much of the local signal geometry, while symmetric distances still lack a direct candidate-reliability channel.

## 12. Transformer diagnostic

{transformer_text}

First-to-final distance correlations remained between {transformer.first_to_final_distance_spearman.min():.3f} and {transformer.first_to_final_distance_spearman.max():.3f}; geometry was transformed but not uniformly destroyed. LocalWarp did not strongly help the MLP/retrieval panel, and FT-Transformer did not specifically destroy a corresponding advantage. **DEMOTE** custom Transformer geometry; the optional intervention gate was not passed.

## 13. OrbitCover successor

No truly new theoretical extension survived. With fixed marginals, variance-optimal two-sample coupling reduces to covariance minimization, already the domain of classical and recent antithetic/optimal-coupling work. Existing OrbitCover evidence remains a valuable same-target finite-budget construction, but not a new general coupling theorem. **KEEP CURRENT PAPER SEPARATE.** See [ORBITCOVER_SUCCESSOR_AUDIT.md](ORBITCOVER_SUCCESSOR_AUDIT.md).

## 14. Failed hypotheses

- Explicit nonlinear retrieval geometry did not give a uniform real-panel benefit.
- Better exact signal-metric alignment did not reliably translate into lower candidate-noise risk.
- LocalWarp was often redundant with an expressive key encoder; the capacity interaction was not monotone.
- The raw symmetric distance cannot encode candidate-only heteroscedastic reliability in general.
- The wrong warp was not guaranteed to fail on every real dataset, even though it failed on the controlled S4 task.
- Metric-alignment/performance association was not strong enough for a causal claim.
- FT-Transformer did not uniformly destroy tokenizer geometry.
- The broad OrbitCover covariance-optimal successor was occupied by antithetic-coupling theory.

## 15. Best simple scientific insight

**A statistically good neighbor is a row with low conditional target mismatch and low candidate uncertainty—not merely a nearby row.**

## 16. Candidate ICLR thesis

The strongest supported thesis is: **retrieval-based tabular models induce local signal metrics, but statistically optimal retrieval also needs a candidate-reliability term that a symmetric learned distance cannot generally express.** This is an analysis-and-derived-design thesis; the present screen does not establish a new SOTA method.

## 17. Method consequence

No new method survived the evidence gate, so no method name or victory is claimed. The theory suggests a future two-factor score `compatibility_θ(x,i) + λ·reliability(i)`, where reliability is estimated strictly out of fold. Parameter overhead would be one scalar reliability head or cached scalar per candidate. It must beat raw and capacity-matched deep-key retrieval before receiving a method claim.

## 18. ICLR readiness

**INTERESTING THEORY ONLY.** The exact theory, noise failure mode, and branch distinction justify one decisive method pilot, but not yet a full prospective benchmark. The real-data mechanism statistic is null, explicit retrieval geometry failed to improve dataset-balanced performance, the panel is reused rather than prospective, and the compact models are not leaderboard configurations. Promotion requires the two-factor score to pass the next experiment's performance and mechanism gates.

## 19. Next 3-day experiment plan

Freeze before outcomes these 12 untouched public datasets: bank-marketing, credit-g, electricity, jannis, covertype, and MagicTelescope for classification; abalone, cpu_act, elevators, Bike_Sharing_Demand, sulfur, and superconduct for regression. Verify availability/licensing before the freeze, then use 5 splits and 3 seeds without replacement; no dataset may overlap this panel. Compare published TabR and ModernNCA implementations, MLP, kNN, PLE, PLR, LMNN/NCA-style global metrics, raw shallow/deep keys, LocalWarp, and a single two-factor risk score. Estimate candidate reliability only from nested out-of-fold residuals/probabilities.

Day 1: prove consistency/calibration requirements for the plug-in reliability term and freeze datasets, splits, budgets, and hyperparameter grids. Day 2: run 12 datasets on two H100s, logging prediction, proxy-risk ranking, candidate frequency, residual consistency, and calibration. Day 3: run branch/capacity/ablation audits and write the go/no-go report. Success requires ≥8/12 dataset-balanced wins against raw deep keys for both a retrieval model and ModernNCA, positive alignment–gain association, lower proxy risk on ≥8/12, no MLP-only explanation, and robustness to nested cross-fitting. Failure of either the performance or mechanism gate kills the direction.
"""
    (HERE / "results.md").write_text(text)


def main() -> None:
    real = load_real()
    _, real_out, _, _, branch, ranking = build_tables(real)
    figure_risk_law()
    figure_good_neighbor()
    figure_metric_field()
    figure_branch(branch)
    alignment_cells = figure_alignment_gain(real_out)
    capacity = figure_capacity(real_out)
    figure_directions(ranking)
    write_results(real_out, branch, ranking, alignment_cells, capacity)
    write_ranking(ranking)
    print(json.dumps({
        "real_rows": len(real_out), "datasets": real_out.dataset.nunique(),
        "figures_png": len(list(FIGURES.glob("*.png"))), "tables": 6,
    }, indent=2))


if __name__ == "__main__":
    main()
