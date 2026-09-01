"""Generate compact paper-ready figures from frozen result tables."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT = RESULTS / "figures"
COLORS = {
    "strength2_confirmation": "#1f77b4",
    "strength2_openml_external": "#d95f02",
    "strength2_openml_taskbalanced": "#1b9e77",
    "strength2_openml_multiclass": "#7570b3",
}


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def anytime() -> None:
    frame = pd.read_csv(RESULTS / "anytime_nested_cover_cells.csv")
    frame["ratio"] = np.clip(frame.nested_residual / frame.iid_residual, 1e-14, None)
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    offsets = np.linspace(-.15, .15, frame.panel.nunique())
    for offset, (panel, current) in zip(offsets, frame.groupby("panel")):
        for budget, values in current.groupby("budget"):
            x = np.full(len(values), np.log2(budget) + offset)
            ax.scatter(x, values.ratio, s=14, alpha=.55, color=COLORS[panel], edgecolor="none")
        means = current.groupby("budget").ratio.median()
        ax.plot(np.log2(means.index) + offset, means.values, marker="o", lw=2,
                color=COLORS[panel], label=panel.replace("strength2_", "").replace("_", " "))
    ax.axhline(1, color="black", lw=1, ls="--")
    ax.set_xticks(np.log2([4, 16, 64]), ["4 / strength 1", "16 / strength 2", "64 / strength 3"])
    ax.set_ylabel("residual / equal-budget IID")
    ax.set_yscale("log")
    ax.legend(frameon=False)
    ax.set_title("One nested schedule improves at every checkpoint")
    fig.tight_layout()
    save(fig, "anytime_nested_residuals")


def phase() -> None:
    frame = pd.read_csv(RESULTS / "interaction_phase_empirical_cells.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4), sharex=True, sharey=True)
    xline = np.linspace(0, 0.57, 200)
    for panel, current in frame.groupby("panel"):
        label = panel.replace("strength2_", "").replace("_", " ")
        for ax in axes:
            ax.scatter(current.order3_fraction, current.order4_fraction, s=25, alpha=.72,
                       color=COLORS[panel], label=label, edgecolor="white", linewidth=.3)
    axes[0].plot(xline, 27 / 16 - 3 * xline, color="black", lw=1.5)
    axes[0].fill_between(xline, 0, np.clip(27 / 16 - 3 * xline, 0, 1), color="#66c2a5", alpha=.15)
    axes[0].scatter([.99], [.01], marker="X", color="#b2182b", s=70,
                    label="pure triple adverse")
    axes[0].set_title("Strength 2 vs IID-16")
    axes[1].axhline(27 / 64, color="black", lw=1.5)
    axes[1].axhspan(0, 27 / 64, color="#66c2a5", alpha=.15)
    axes[1].scatter([.01], [.99], marker="X", color="#b2182b", s=70,
                    label="pure four-way adverse")
    axes[1].set_title("Strength 3 vs IID-64")
    for ax in axes:
        ax.set_xlim(-.02, 1.02); ax.set_ylim(-.02, 1.02)
        ax.set_xlabel("triple-interaction energy fraction")
    axes[0].set_ylabel("four-way interaction energy fraction")
    handles, labels = [], []
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle); labels.append(label)
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(.5, .005))
    fig.tight_layout(rect=(0, .19, 1, 1))
    save(fig, "interaction_phase_diagram")


def selection_frontier() -> None:
    frame = pd.read_csv(RESULTS / "selection_strength_frontier_means.csv")
    keep = frame[frame.method.isin(["iid_b4", "iid_b16", "iid_b64", "strength1_b4", "strength2_b16", "strength3_b64"])]
    panels = list(keep.groupby("panel"))
    fig, axes = plt.subplots(1, len(panels), figsize=(3.25 * len(panels), 3.1), sharey=True)
    for ax, (panel, current) in zip(axes, panels):
        iid = current[current.method.str.startswith("iid")].sort_values("budget")
        cover = current[current.method.str.startswith("strength")].sort_values("budget")
        ax.plot(iid.budget, iid.selection_agreement, "o--", color="#777777", label="IID")
        ax.plot(cover.budget, cover.selection_agreement, "o-", color="#1f77b4", lw=2, label="nested strength")
        ax.set_xscale("log", base=2); ax.set_xticks([4, 16, 64], [4, 16, 64])
        ax.set_ylim(.82, 1.01); ax.set_title(panel.replace("_", " "))
        ax.set_xlabel("fits per candidate")
    axes[0].set_ylabel("quotient-winner agreement")
    axes[-1].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    save(fig, "selection_strength_frontier")


def selection_shift() -> None:
    frame = pd.read_csv(RESULTS / "selection_error_decomposition_cells.csv")
    panels = ("openml_external", "openml_taskbalanced")
    methods = ("iid16", "srswor16", "strength2")
    labels = ("IID", "SRSWOR", "strength 2")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for ax, panel in zip(axes, panels):
        current = frame[(frame.panel == panel) & frame.method.isin(methods)]
        values = current.groupby("method").mean(numeric_only=True)
        totals = [values.loc[method, "mean_test_quotient_regret"] for method in methods]
        floor = float(values.target_shift_floor.mean())
        ax.bar(labels, totals, color=("#888888", "#b3b3b3", "#1f77b4"), width=.65)
        ax.axhline(floor, color="#b2182b", ls="--", lw=1.6, label="exact target-shift floor")
        ax.set_title(panel.replace("openml_", "").replace("_", " "))
        ax.set_ylabel("held-out quotient selection regret")
        ax.tick_params(axis="x", rotation=18)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Validation fidelity helps only when validation and test rankings align", y=1.02)
    fig.tight_layout()
    save(fig, "selection_target_shift_decomposition")


def cross_score() -> None:
    calibration = pd.read_csv(RESULTS / "cross_score_efficiency_cells.csv")
    cells = pd.read_csv(RESULTS / "cross_quotient_selection_cells.csv")
    panels = list(calibration.panel.drop_duplicates())
    labels = [panel.replace("openml_", "").replace("_repeat", "").replace("_", " ") for panel in panels]
    rmse = calibration.groupby(["panel", "method"]).score_rmse.mean().unstack()
    agreement = cells.groupby(["panel", "method"]).selection_agreement.mean().unstack()
    x = np.arange(len(panels))
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.25))
    ratios = rmse.loc[panels, "strength2_cross32"] / rmse.loc[panels, "iid_u32"]
    axes[0].bar(x, ratios, color="#1f77b4", width=.65)
    axes[0].axhline(1, color="black", ls="--", lw=1)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_yticks([0, .25, .5, .75, 1.0])
    axes[0].set_ylabel("cross-score RMSE / IID-U RMSE")
    axes[0].set_title("Unbiased quotient-loss estimation")
    width = .36
    axes[1].bar(x - width / 2, agreement.loc[panels, "iid_u32"], width,
                color="#888888", label="IID U-statistic")
    axes[1].bar(x + width / 2, agreement.loc[panels, "strength2_cross32"], width,
                color="#1f77b4", label="cover cross-score")
    axes[1].set_ylim(.9, 1.005)
    axes[1].set_ylabel("quotient-winner agreement")
    axes[1].set_title("Equal-budget selection (32 fits)")
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.set_xticks(x, labels, rotation=25, ha="right")
    fig.tight_layout()
    save(fig, "cross_quotient_selection")


def cross_score_frontier() -> None:
    cal32 = pd.read_csv(RESULTS / "cross_score_efficiency_cells.csv")
    cell32 = pd.read_csv(RESULTS / "cross_quotient_selection_cells.csv")
    cal64 = pd.read_csv(RESULTS / "cross_score_budget_frontier_calibration.csv")
    cell64 = pd.read_csv(RESULTS / "cross_score_budget_frontier_cells.csv")
    panels = list(cal32.panel.drop_duplicates())
    labels = [panel.replace("openml_", "").replace("_repeat", "").replace("_", " ") for panel in panels]
    rmse32 = cal32.groupby(["panel", "method"]).score_rmse.mean().unstack()
    rmse64 = cal64.groupby(["panel", "method"]).score_rmse.mean().unstack()
    agree32 = cell32.groupby(["panel", "method"]).selection_agreement.mean().unstack()
    agree64 = cell64.groupby(["panel", "method"]).selection_agreement.mean().unstack()
    ratio32 = rmse32.loc[panels, "strength2_cross32"] / rmse32.loc[panels, "iid_u32"]
    ratio64 = rmse64.loc[panels, "cover_block_u64"] / rmse64.loc[panels, "iid_u64"]
    gap32 = agree32.loc[panels, "strength2_cross32"] - agree32.loc[panels, "iid_u32"]
    gap64 = agree64.loc[panels, "cover_block_u64"] - agree64.loc[panels, "iid_u64"]
    x = np.arange(len(panels)); width = .36
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.25))
    axes[0].bar(x - width / 2, ratio32, width, color="#6baed6", label="32 fits")
    axes[0].bar(x + width / 2, ratio64, width, color="#08519c", label="64 fits")
    axes[0].axhline(1, color="black", ls="--", lw=1)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_yticks([0, .25, .5, .75, 1.0])
    axes[0].set_ylabel("cover-U RMSE / IID-U RMSE")
    axes[0].set_title("Unbiased score efficiency")
    axes[1].bar(x - width / 2, gap32, width, color="#6baed6", label="32 fits")
    axes[1].bar(x + width / 2, gap64, width, color="#08519c", label="64 fits")
    axes[1].axhline(0, color="black", lw=1)
    axes[1].set_ylabel("cover minus IID winner agreement")
    axes[1].set_title("Selection advantage")
    for ax in axes:
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, "cross_score_budget_frontier")


def repeated_holdout() -> None:
    frame = pd.read_csv(RESULTS / "repeated_holdout_shift_cells.csv")
    frame["short"] = (
        frame.dataset.str.replace("openml-", "", regex=False)
        .str.replace(r"-\d+$", "", regex=True)
    )
    frame = frame.sort_values(["panel", "winner_agreement_probability"]).reset_index(drop=True)
    y = np.arange(len(frame))
    color = frame.panel.map({"openml_external": "#d95f02", "openml_taskbalanced": "#1b9e77"})
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 5.3), sharey=True,
                             gridspec_kw={"width_ratios": [1.35, 1]})
    for index, row in frame.iterrows():
        axes[0].plot(
            [row.winner_agreement_probability, float(row.original_winner_agreement)],
            [index, index], color="#cccccc", lw=1, zorder=1,
        )
    axes[0].scatter(frame.winner_agreement_probability, y, c=color, s=30,
                    label="repartition probability", zorder=3)
    axes[0].scatter(frame.original_winner_agreement.astype(float), y,
                    facecolors="none", edgecolors="#333333", s=34,
                    label="original outcome", zorder=4)
    axes[0].axvline(.95, color="black", ls="--", lw=1)
    axes[0].set_xlim(-.03, 1.03)
    axes[0].set_xlabel("validation/test winner agreement")
    axes[0].set_title("Conditional rank stability")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    axes[0].set_yticks(y, frame.short)

    axes[1].scatter(frame.original_floor_mid_percentile, y, c=color, s=30)
    axes[1].axvline(.975, color="#b2182b", ls="--", lw=1.2,
                   label="upper 2.5% tail")
    axes[1].set_xlim(-.03, 1.03)
    axes[1].set_xlabel("original floor mid-percentile")
    axes[1].set_title("Was the original floor exceptional?")
    axes[1].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("Repeated partitions reveal finite held-out sampling instability", y=.995)
    fig.tight_layout()
    save(fig, "repeated_holdout_rank_instability")


def stability_sets() -> None:
    frame = pd.read_csv(RESULTS / "stability_set_cells.csv")
    means = frame.groupby(["panel", "method"]).mean(numeric_only=True)
    panels = list(frame.panel.drop_duplicates())
    labels = [panel.replace("openml_", "").replace("_repeat", "").replace("_", " ") for panel in panels]
    cover = means.xs("cover_cross_union64", level="method").loc[panels]
    iid = means.xs("iid_u_union64", level="method").loc[panels]
    colors = plt.cm.viridis(np.linspace(.12, .88, len(panels)))
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.25))
    for index, (panel, color) in enumerate(zip(panels, colors)):
        axes[0].annotate("", xy=(cover.loc[panel, "set_size"], cover.loc[panel, "validation_winner_covered"]),
                         xytext=(iid.loc[panel, "set_size"], iid.loc[panel, "validation_winner_covered"]),
                         arrowprops={"arrowstyle": "->", "color": color, "lw": 1.5})
        axes[0].scatter(cover.loc[panel, "set_size"], cover.loc[panel, "validation_winner_covered"],
                        color=color, s=35, label=labels[index], zorder=3)
        axes[0].scatter(iid.loc[panel, "set_size"], iid.loc[panel, "validation_winner_covered"],
                        facecolors="none", edgecolors=color, s=35, zorder=3)
    axes[0].set_xlabel("mean set size (smaller is better)")
    axes[0].set_ylabel("exact validation-winner coverage")
    axes[0].set_title("IID-U open circle → cover-U filled")
    axes[0].legend(frameon=False, fontsize=7, loc="lower left")
    x = np.arange(len(panels)); width = .36
    axes[1].bar(x - width / 2, iid.wrong_singleton, width, color="#888888", label="IID-U union")
    axes[1].bar(x + width / 2, cover.wrong_singleton, width, color="#1f77b4", label="cover-U union")
    axes[1].set_xticks(x, labels, rotation=25, ha="right")
    axes[1].set_ylabel("wrong-singleton probability")
    axes[1].set_title("Confidently wrong selections")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Two-replicate stability sets at an equal 64-fit budget", y=1.01)
    fig.tight_layout()
    save(fig, "stability_set_frontier")


def variance_identity() -> None:
    frame = pd.read_csv(RESULTS / "cross_variance_identity_cells.csv")
    frame = frame[frame.nondegenerate & (frame.predicted_variance > 0) & (frame.observed_variance > 0)]
    methods = ("iid_mean16", "strength2_mean16")
    titles = ("IID 16-member mean", "strength-2 16-fit cover")
    panels = list(frame.panel.drop_duplicates())
    palette = dict(zip(panels, plt.cm.viridis(np.linspace(.1, .9, len(panels)))))
    limits = [frame[["predicted_variance", "observed_variance"]].to_numpy().min(),
              frame[["predicted_variance", "observed_variance"]].to_numpy().max()]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), sharex=True, sharey=True)
    for ax, method, title in zip(axes, methods, titles):
        current = frame[frame.method == method]
        for panel, values in current.groupby("panel"):
            ax.scatter(values.predicted_variance, values.observed_variance, s=15,
                       alpha=.68, color=palette[panel], edgecolor="none",
                       label=panel.replace("openml_", "").replace("_repeat", "").replace("_", " "))
        ax.plot(limits, limits, color="black", ls="--", lw=1)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("operator-predicted variance")
    axes[0].set_ylabel("disjoint-stream observed variance")
    axes[1].legend(frameon=False, fontsize=7, loc="lower right")
    fig.suptitle("Proposition 19 predicts real cross-score variance", y=1.01)
    fig.tight_layout()
    save(fig, "cross_variance_identity")


def variance_components() -> None:
    frame = pd.read_csv(RESULTS / "cross_variance_identity_cells.csv")
    panels = list(frame.panel.drop_duplicates())
    labels = [panel.replace("openml_", "").replace("_repeat", "").replace("_", " ") for panel in panels]
    components = ("residual_aligned_variance", "covariance_self_interaction_variance")
    ratios = {component: [] for component in components}
    for panel in panels:
        current = frame[frame.panel == panel]
        means = current.groupby("method").mean(numeric_only=True)
        for component in components:
            ratios[component].append(
                means.loc["strength2_mean16", component] / means.loc["iid_mean16", component]
            )
    x = np.arange(len(panels)); width = .36
    fig, ax = plt.subplots(figsize=(5.6, 3.25))
    ax.bar(x - width / 2, ratios[components[0]], width, color="#3182bd",
           label=r"residual-aligned $2\langle r,Cr\rangle$")
    ax.bar(x + width / 2, ratios[components[1]], width, color="#9ecae1",
           label=r"self-interaction $\mathrm{tr}(C^2)$")
    ax.axhline(1, color="black", ls="--", lw=1)
    ax.set_ylim(0, 1.03)
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylabel("cover component / IID component")
    ax.set_title("Strength balance reduces both cross-score variance terms")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, "cross_variance_components")


def log_jackknife_frontier() -> None:
    cal32 = pd.read_csv(RESULTS / "log_quotient_jackknife_calibration.csv")
    cal64 = pd.read_csv(RESULTS / "log_jackknife_frontier_calibration.csv")
    panels = list(cal64.panel.drop_duplicates())
    labels = [panel.replace("openml_", "").replace("_repeat", "").replace("_", " ") for panel in panels]
    means32 = cal32.groupby(["panel", "method"]).score_rmse.mean().unstack()
    means64 = cal64.groupby(["panel", "method"]).score_rmse.mean().unstack()
    ratio32 = means32.loc[panels, "strength2_jackknife32"] / means32.loc[panels, "iid_jackknife32"]
    ratio64 = means64.loc[panels, "strength2_jackknife64"] / means64.loc[panels, "iid_jackknife64"]
    frontier = means64.loc[panels, "strength2_jackknife64"] / means32.loc[panels, "strength2_jackknife32"]
    x = np.arange(len(panels)); width = .36
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.25))
    axes[0].bar(x - width / 2, ratio32, width, color="#6baed6", label="32 fits")
    axes[0].bar(x + width / 2, ratio64, width, color="#08519c", label="64 fits")
    axes[0].axhline(1, color="black", ls="--", lw=1)
    axes[0].set_ylabel("cover jackknife RMSE / IID")
    axes[0].set_title("Approximate log-quotient efficiency")
    axes[1].bar(x, frontier, color="#31a354", width=.65)
    axes[1].axhline(1, color="black", ls="--", lw=1)
    axes[1].set_ylabel("cover-64 RMSE / cover-32 RMSE")
    axes[1].set_title("Four-block compute frontier")
    for ax in axes:
        ax.set_ylim(0, 1.03)
        ax.set_xticks(x, labels, rotation=25, ha="right")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, "log_jackknife_frontier")


def synthetic_selection_phase() -> None:
    frame = pd.read_csv(RESULTS / "synthetic_selection_phase_cells.csv")
    fig, axes = plt.subplots(1, 4, figsize=(10.2, 2.9), sharey=True)
    styles = {
        "strength2_cross32": ("#1f78b4", "o-", "strength-2 cross"),
        "iid_u32": ("#777777", "s--", "IID-U"),
    }
    for order, ax in zip(range(1, 5), axes):
        current = frame[frame.interaction_order == order]
        for method, (color, style, label) in styles.items():
            values = current[current.method == method].sort_values("quotient_loss_margin")
            ax.plot(values.quotient_loss_margin, values.inversion_rate,
                    style, color=color, lw=1.8, ms=4, label=label)
        ax.set_xscale("log")
        ax.set_xticks([.002, .005, .01, .02], [".002", ".005", ".01", ".02"])
        ax.set_ylim(-.015, .47)
        ax.set_title(f"pure order {order}")
        ax.set_xlabel("quotient-risk gap")
        ax.grid(axis="y", color="#dddddd", lw=.6)
    axes[0].set_ylabel("winner inversion probability")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle("Strength-2 selection follows a non-monotone alias boundary", y=1.02)
    fig.tight_layout()
    save(fig, "synthetic_selection_phase")


def compute_allocator() -> None:
    frame = pd.read_csv(RESULTS / "cheap_screen_precise_deploy_cells.csv")
    panels = list(dict.fromkeys(frame.panel))
    short = [name.replace("openml_", "").replace("_repeat", "") for name in panels]
    proposal = frame[frame.method == "cheap_screen_precise_deploy"].groupby("panel").mean(numeric_only=True)
    control = frame[frame.method == "paired_all_candidate_u64"].groupby("panel").mean(numeric_only=True)
    x = np.arange(len(panels))
    fig, axes = plt.subplots(1, 2, figsize=(8.3, 3.15))
    axes[0].bar(x, 100 * proposal.loc[panels].fit_saving_fraction,
                color="#31a354", width=.68)
    axes[0].axhline(20, color="black", ls="--", lw=1)
    axes[0].set_ylabel("fit saving vs all-candidate U64 (%)")
    axes[0].set_title("Deterministic compute reduction")
    delta = 100 * (proposal.loc[panels].selection_agreement -
                   control.loc[panels].selection_agreement)
    axes[1].bar(x, delta, color=["#b2182b" if value < 0 else "#3182bd" for value in delta], width=.68)
    axes[1].axhline(0, color="black", lw=1)
    axes[1].set_ylabel("agreement change (percentage points)")
    axes[1].set_title("Paired selection effect")
    for ax in axes:
        ax.set_xticks(x, short, rotation=25, ha="right")
    fig.tight_layout()
    save(fig, "cheap_screen_precise_deploy")


def disjoint_packing() -> None:
    cal32 = pd.read_csv(RESULTS / "disjoint_pair32_calibration.csv")
    cal64 = pd.read_csv(RESULTS / "disjoint_pair_cross_calibration.csv")
    pack64 = pd.read_csv(RESULTS / "disjoint_pack64_calibration.csv")
    panels = list(dict.fromkeys(cal32.panel))
    labels = [name.replace("openml_", "").replace("_repeat", "") for name in panels]
    means32 = cal32.groupby(["panel", "method"]).mean(numeric_only=True)
    means64 = cal64.groupby(["panel", "method"]).mean(numeric_only=True)
    means_pack = pack64.groupby(["panel", "method"]).mean(numeric_only=True)
    rmse32 = np.asarray([
        means32.loc[(panel, "disjoint_pair_mean32"), "score_rmse"] /
        means32.loc[(panel, "independent_pair_mean32"), "score_rmse"]
        for panel in panels
    ])
    rmse64 = np.asarray([
        means64.loc[(panel, "disjoint_pair_cross64"), "score_rmse"] /
        means64.loc[(panel, "independent_block_u64"), "score_rmse"]
        for panel in panels
    ])
    residual32 = np.asarray([
        means32.loc[(panel, "disjoint_pair_mean32"), "prediction_residual"] /
        means32.loc[(panel, "independent_pair_mean32"), "prediction_residual"]
        for panel in panels
    ])
    rmse_pack = np.asarray([
        means_pack.loc[(panel, "mutually_disjoint_pack64"), "score_rmse"] /
        means_pack.loc[(panel, "two_disjoint_pairs64"), "score_rmse"]
        for panel in panels
    ])
    residual_pack = np.asarray([
        means_pack.loc[(panel, "mutually_disjoint_pack64"), "prediction_residual"] /
        means_pack.loc[(panel, "two_disjoint_pairs64"), "prediction_residual"]
        for panel in panels
    ])
    x = np.arange(len(panels)); width = .25
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.15), sharey=True)
    axes[0].bar(x - width, rmse32, width, color="#9ecae1", label="pair mean32")
    axes[0].bar(x, rmse64, width, color="#3182bd", label="pair cross64")
    axes[0].bar(x + width, rmse_pack, width, color="#08519c", label="four-pack mean64")
    axes[0].set_title("Quotient-score RMSE")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].bar(x - width / 2, residual32, width, color="#74c476", label="pair mean32")
    axes[1].bar(x + width / 2, residual_pack, width, color="#006d2c", label="four-pack mean64")
    axes[1].set_title("Prediction residual")
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.axhline(1, color="black", ls="--", lw=1)
        ax.set_ylim(0, 1.03)
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.set_ylabel("disjoint packed / independent")
    fig.suptitle("Regular disjoint-cover packing improves equal-budget estimation", y=1.02)
    fig.tight_layout()
    save(fig, "disjoint_cover_packing")


def pack_cross128_power() -> None:
    calibration = pd.read_csv(RESULTS / "disjoint_pack_cross128_calibration.csv")
    power = pd.read_csv(RESULTS / "pack_cross128_power_cells.csv")
    means = calibration.groupby(["panel", "method"]).score_rmse.mean()
    panels = list(dict.fromkeys(calibration.panel))
    labels = [name.replace("openml_", "").replace("_repeat", "") for name in panels]
    ratios = np.asarray([
        means.loc[(panel, "disjoint_pack_cross128")] /
        means.loc[(panel, "independent_cover_u128")]
        for panel in panels
    ])
    curves = power.groupby(["coupling", "gap_multiplier", "method"]).inversion_probability.mean()
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.15))
    axes[0].bar(np.arange(len(panels)), ratios, color="#08519c", width=.68)
    axes[0].axhline(1, color="black", ls="--", lw=1)
    axes[0].set_ylim(0, 1.03)
    axes[0].set_xticks(np.arange(len(panels)), labels, rotation=25, ha="right")
    axes[0].set_ylabel("pack-cross128 / cover-U128 RMSE")
    axes[0].set_title("Unbiased score efficiency")
    colors = {"disjoint_pack_cross128": "#08519c", "independent_cover_u128": "#e6550d"}
    names = {"disjoint_pack_cross128": "pack cross", "independent_cover_u128": "independent U"}
    for coupling, linestyle in (("common", "-"), ("candidate_independent", "--")):
        for method in colors:
            values = [curves.loc[(coupling, gap, method)] for gap in (.25, .5, 1., 2.)]
            label = names[method] + (" (ind. actions)" if coupling != "common" else "")
            axes[1].plot((.25, .5, 1., 2.), values, marker="o", lw=1.8,
                         ls=linestyle, color=colors[method], label=label)
    axes[1].set_xlabel("injected exact gap / control pair SD")
    axes[1].set_ylabel("pairwise inversion probability")
    axes[1].set_title("Gap-calibrated ranking power")
    axes[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, "pack_cross128_ranking_power")


def packed_unbiased_frontier() -> None:
    frame = pd.read_csv(RESULTS / "packed_unbiased_frontier.csv")
    panels = list(dict.fromkeys(frame.panel))
    labels = [name.replace("openml_", "").replace("_repeat", "") for name in panels]
    budgets = np.asarray((32, 64, 128))
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.15))
    colors = plt.cm.Blues(np.linspace(.45, .9, len(panels)))
    for panel, label, color in zip(panels, labels, colors):
        means = frame[frame.panel == panel][["rmse32", "rmse64", "rmse128"]].mean().to_numpy()
        axes[0].plot(budgets, means / means[0], marker="o", lw=1.8, color=color, label=label)
    axes[0].plot(budgets, np.sqrt(32 / budgets), color="black", ls="--", lw=1.3,
                 label=r"independent $B^{-1/2}$")
    axes[0].scatter([128], [0], marker="*", s=95, color="#d7301f", zorder=5,
                    label="exhaustive resolution")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(budgets, ["32", "64", "128"])
    axes[0].set_xlabel("fits")
    axes[0].set_ylabel("RMSE / RMSE at 32")
    axes[0].set_ylim(-.03, 1.04)
    axes[0].set_title("Unbiased packed-score frontier")
    axes[0].legend(frameon=False, fontsize=7)
    x = np.arange(len(panels)); width = .34
    exponent1 = frame.groupby("panel").exponent_32_to_64.median().reindex(panels)
    exponent2 = frame.groupby("panel").exponent_64_to_128.median().reindex(panels)
    axes[1].bar(x - width / 2, exponent1, width, color="#6baed6", label="32→64")
    axes[1].bar(x + width / 2, exponent2, width, color="#08519c", label="64→128")
    axes[1].axhline(.5, color="black", ls="--", lw=1.3, label="MC exponent 0.5")
    axes[1].set_xticks(x, labels, rotation=25, ha="right")
    axes[1].set_ylabel(r"effective exponent $-\log_2(R_{2B}/R_B)$")
    axes[1].set_title("Finite-budget decay")
    axes[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, "packed_unbiased_frontier")


def combined_source_scope() -> None:
    frame = pd.read_csv(RESULTS / "combined_packing_source_effects.csv")
    summary = json.loads((RESULTS / "combined_packing_source_summary.json").read_text())
    order = ("pair32", "pack64", "unbiased_pair_cross64")
    labels = ("pair mean32", "four-pack mean64", "pair cross64")
    colors = ("#6baed6", "#08519c", "#31a354")
    fig, ax = plt.subplots(figsize=(6.3, 3.15))
    for index, (name, color) in enumerate(zip(order, colors)):
        values = frame[frame.comparison == name].percent_reduction.to_numpy()
        offsets = np.linspace(-.12, .12, len(values))
        ax.scatter(values, index + offsets, color=color, s=24, alpha=.72, zorder=2)
        result = summary["comparisons"][name]
        mean = result["equal_source_mean_percent_reduction"]
        low, high = result["bootstrap_95_interval"]
        ax.errorbar(mean, index, xerr=[[mean - low], [high - mean]], fmt="D",
                    color="black", mfc=color, ms=6, capsize=3, lw=1.4, zorder=3)
    ax.axvline(0, color="black", ls="--", lw=1)
    ax.set_yticks(np.arange(3), labels)
    ax.set_xlabel("source-level score-RMSE reduction (%)")
    ax.set_title(f"Packing gains extend to {frame.source.nunique()} unique non-exhaustive sources")
    ax.grid(axis="x", color="#dddddd", lw=.6)
    fig.tight_layout()
    save(fig, "combined_packing_source_scope")


def modern_model_transport() -> None:
    strength = pd.read_csv(RESULTS / "modern_model_strength2_cells.csv")
    strength = strength[strength.split == "test"].copy()
    calibration = pd.read_csv(RESULTS / "modern_model_packing_calibration.csv")
    datasets = list(dict.fromkeys(strength.dataset))
    labels = [name.replace("openml-", "").rsplit("-", 1)[0] for name in datasets]
    x = np.arange(len(datasets)); width = .34
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.35))
    for offset, model, color, label in (
        (-width / 2, "native_histgb", "#9ecae1", "native HistGB"),
        (width / 2, "catboost_native", "#08519c", "CatBoost"),
    ):
        current = strength[strength.model == model].set_index("dataset").loc[datasets]
        ratio = np.maximum(current.strength2_residual.to_numpy(), 0) / current.iid16_residual
        axes[0].bar(x + offset, ratio, width, color=color, label=label)
    axes[0].axhline(1, color="black", ls="--", lw=1)
    axes[0].set_ylabel("strength-2 / IID-16 residual")
    axes[0].set_title("Added-family strength transport")
    axes[0].legend(frameon=False, fontsize=8)
    comparisons = (
        ("pair32", "disjoint_pair_mean32", "independent_pair_mean32", "#6baed6", "pair32"),
        ("pack64", "mutually_disjoint_pack64", "two_disjoint_pairs64", "#08519c", "pack64"),
        ("pair_cross64", "disjoint_pair_cross64", "independent_block_u64", "#31a354", "cross64"),
    )
    offsets = (-.24, 0, .24)
    for offset, (family, action, control, color, label) in zip(offsets, comparisons):
        current = calibration[
            (calibration.family == family)
            & (calibration.model == "catboost_native")
            & calibration.method.isin((action, control))
        ].pivot(index="dataset", columns="method", values="score_rmse").loc[datasets]
        axes[1].bar(x + offset, current[action] / current[control], .23,
                    color=color, label=label)
    axes[1].axhline(1, color="black", ls="--", lw=1)
    axes[1].set_ylabel("packed / equal-fit control RMSE")
    axes[1].set_title("All 8 nondegenerate CatBoost cells improve")
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.set_xticks(x, labels, rotation=28, ha="right")
        ax.grid(axis="y", color="#dddddd", lw=.6)
    fig.tight_layout()
    save(fig, "modern_model_transport")


def repeated_split_transport() -> None:
    summary = json.loads((RESULTS / "repeated_split_modern_summary.json").read_text())
    splits = [str(value) for value in summary["split_seeds"]]
    short = [value[-4:] for value in splits]
    fig, axes = plt.subplots(1, 3, figsize=(11.7, 3.25))
    by_split = summary["strength2"]["by_split"]
    fractions = [
        by_split[value]["material_wins"] / by_split[value]["material_cells"]
        for value in splits
    ]
    fractions.append(summary["strength2"]["material_win_fraction"])
    axes[0].bar(np.arange(4), fractions,
                color=["#9ecae1", "#9ecae1", "#9ecae1", "#08519c"], width=.65)
    axes[0].axhline(.8, color="black", ls="--", lw=1, label="frozen aggregate rule")
    axes[0].set_ylim(.7, 1.01)
    axes[0].set_xticks(np.arange(4), [*short, "all"])
    axes[0].set_ylabel("material-cell win fraction")
    axes[0].set_xlabel("split-seed suffix")
    axes[0].set_title("Strength-2 transport")
    axes[0].legend(frameon=False, fontsize=8)
    comparisons = (
        ("pair32", "disjoint_pair_mean32", "independent_pair_mean32", "#6baed6"),
        ("pack64", "mutually_disjoint_pack64", "two_disjoint_pairs64", "#08519c"),
        ("pair_cross64", "disjoint_pair_cross64", "independent_block_u64", "#31a354"),
    )
    width = .24
    for offset, (family, action, control, color) in zip((-.24, 0, .24), comparisons):
        reductions = []
        for split_seed in splits:
            frame = pd.read_csv(RESULTS / f"modern_split_{split_seed}_packing_calibration.csv")
            current = frame[
                (frame.family == family) & (frame.model == "catboost_native")
                & frame.method.isin((action, control))
            ].pivot(index="dataset", columns="method", values="score_rmse")
            reductions.append(100 * (1 - current[action] / current[control]).mean())
        axes[1].bar(np.arange(3) + offset, reductions, width, color=color,
                    label=family.replace("pair_cross64", "cross64"))
    axes[1].set_xticks(np.arange(3), short)
    axes[1].set_ylabel("mean CatBoost RMSE reduction (%)")
    axes[1].set_xlabel("split-seed suffix")
    axes[1].set_title("Packing repeats on every split")
    axes[1].legend(frameon=False, fontsize=8)
    alternate = pd.read_csv(RESULTS / "repeated_split_modern_transfer.csv")
    original_payload = json.loads((RESULTS / "modern_model_extension_summary.json").read_text())
    original = pd.DataFrame(original_payload["exact_validation_to_test_transfer"]["sources"])
    original["split_seed"] = 2026082831
    transfer = pd.concat([original, alternate], ignore_index=True)
    datasets = list(dict.fromkeys(transfer.dataset))
    split_order = [2026082831, *map(int, splits)]
    matrix = transfer.pivot(index="dataset", columns="split_seed", values="winner_agreement")
    matrix = matrix.loc[datasets, split_order].astype(float)
    axes[2].imshow(matrix, cmap=plt.matplotlib.colors.ListedColormap(["#d7301f", "#66bd63"]),
                   vmin=0, vmax=1, aspect="auto")
    axes[2].set_xticks(np.arange(4), [str(value)[-4:] for value in split_order], rotation=25)
    axes[2].set_yticks(np.arange(len(datasets)),
                       [name.replace("openml-", "").rsplit("-", 1)[0] for name in datasets])
    axes[2].set_xlabel("split-seed suffix")
    axes[2].set_title("Exact validation→test winner")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axes[2].text(column, row, "✓" if matrix.iloc[row, column] else "×",
                         ha="center", va="center", color="white", fontsize=9)
    fig.tight_layout()
    save(fig, "repeated_split_modern_transport")


def prospective_source_c_transport() -> None:
    strength = pd.read_csv(RESULTS / "late_source_c_strength2_cells.csv")
    strength = strength[strength.split == "test"].copy()
    calibration = pd.read_csv(RESULTS / "late_source_c_packing_calibration.csv")
    metric = json.loads((RESULTS / "late_source_c_metric_scope_summary.json").read_text())
    datasets = list(dict.fromkeys(strength.dataset))
    short = [name.replace("openml-", "").rsplit("-", 1)[0] for name in datasets]
    models = list(dict.fromkeys(strength.model))
    model_labels = {
        "onehot_linear": "linear", "ordinal_forest": "forest",
        "native_histgb": "HistGB", "catboost_native": "CatBoost",
        "onehot_adam_mlp": "MLP",
    }
    colors = plt.cm.Blues(np.linspace(.35, .9, len(models)))
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.35))
    x = np.arange(len(datasets)); width = .15
    for index, (model, color) in enumerate(zip(models, colors)):
        current = strength[strength.model == model].set_index("dataset").loc[datasets]
        denominator = current.iid16_residual.to_numpy()
        ratio = np.divide(
            np.maximum(current.strength2_residual.to_numpy(), 0), denominator,
            out=np.zeros_like(denominator), where=denominator > 1e-15,
        )
        axes[0].bar(x + (index - 2) * width, ratio, width, color=color,
                    label=model_labels.get(model, model))
    axes[0].axhline(1, color="black", ls="--", lw=1)
    axes[0].set_ylabel("strength-2 / IID-16 residual")
    axes[0].set_title("Prospective strength-2 gate")
    axes[0].legend(frameon=False, fontsize=6.8, ncol=2)

    comparisons = (
        ("pair32", "disjoint_pair_mean32", "independent_pair_mean32", "#6baed6", "pair32"),
        ("pack64", "mutually_disjoint_pack64", "two_disjoint_pairs64", "#08519c", "pack64"),
        ("pair_cross64", "disjoint_pair_cross64", "independent_block_u64", "#31a354", "cross64"),
    )
    for offset, (family, action, control, color, label) in zip((-.24, 0, .24), comparisons):
        current = calibration[
            (calibration.family == family) & calibration.method.isin((action, control))
        ].groupby(["dataset", "method"]).score_rmse.mean().unstack().loc[datasets]
        axes[1].bar(x + offset, 100 * (1 - current[action] / current[control]), .23,
                    color=color, label=label)
    axes[1].axhline(0, color="black", lw=1)
    axes[1].set_ylabel("source-mean score-RMSE reduction (%)")
    axes[1].set_title("Packing wins all source means")
    axes[1].legend(frameon=False, fontsize=7)

    metric_names = ("brier", "log_loss", "roc_auc", "accuracy")
    metric_labels = ("Brier", "log", "AUC", "accuracy")
    for offset, comparison, color, label in (
        (-.17, "pair32", "#6baed6", "pair32"),
        (.17, "pack64", "#08519c", "pack64"),
    ):
        ratios = [
            metric["comparisons"][comparison]["metrics"][name]["mean_rmse_ratio"]
            for name in metric_names
        ]
        axes[2].bar(np.arange(4) + offset, ratios, .34, color=color, label=label)
    axes[2].axhline(1, color="black", ls="--", lw=1)
    axes[2].set_xticks(np.arange(4), metric_labels, rotation=20, ha="right")
    axes[2].set_ylabel("packed / control mean RMSE")
    axes[2].set_title("Secondary metric scope")
    axes[2].legend(frameon=False, fontsize=7)
    for ax in axes[:2]:
        ax.set_xticks(x, short, rotation=25, ha="right")
    for ax in axes:
        ax.grid(axis="y", color="#dddddd", lw=.6)
    fig.tight_layout()
    save(fig, "prospective_source_c_transport")


def source_c_partition_boundary() -> None:
    original = json.loads((RESULTS / "late_source_c_audit_summary.json").read_text())
    alternate = json.loads((RESULTS / "late_source_c_split_audit_summary.json").read_text())
    audits = (original, alternate)
    labels = ("original 3041", "alternate 3051")
    fig, axes = plt.subplots(1, 3, figsize=(9.7, 3.15))
    material = [row["strength2"]["material_win_fraction"] for row in audits]
    axes[0].bar(np.arange(2), material, color=("#6baed6", "#08519c"), width=.62)
    axes[0].axhline(.8, color="black", ls="--", lw=1)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("material-cell win fraction")
    axes[0].set_title("Strength-2 transports")

    names = ("pair32", "pack64", "unbiased_pair_cross64")
    short = ("pair32", "pack64", "cross64")
    x = np.arange(3); width = .34
    for offset, row, color, label in (
        (-width / 2, original, "#6baed6", "original"),
        (width / 2, alternate, "#08519c", "alternate"),
    ):
        values = [100 * row["packing"][name]["mean_nondegenerate_relative_score_rmse_reduction"]
                  for name in names]
        axes[1].bar(x + offset, values, width, color=color, label=label)
    axes[1].set_xticks(x, short)
    axes[1].set_ylabel("score-RMSE reduction (%)")
    axes[1].set_title("Packing transports")
    axes[1].legend(frameon=False, fontsize=8)

    agreement = [row["validation_test_winner_agreements"] / 4 for row in audits]
    axes[2].bar(np.arange(2), agreement, color=("#66bd63", "#d7301f"), width=.62)
    axes[2].set_ylim(0, 1.05)
    axes[2].set_ylabel("exact winner agreement")
    axes[2].set_title("Partition transfer does not")
    axes[2].text(1, agreement[1] + .05,
                 f"regret {alternate['mean_validation_selected_test_regret']:.4f}",
                 ha="center", fontsize=8)
    for ax in (axes[0], axes[2]):
        ax.set_xticks(np.arange(2), labels, rotation=20, ha="right")
    for ax in axes:
        ax.grid(axis="y", color="#dddddd", lw=.6)
    fig.tight_layout()
    save(fig, "source_c_partition_boundary")


def main() -> None:
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    anytime(); phase(); selection_frontier(); selection_shift(); cross_score(); cross_score_frontier()
    repeated_holdout(); stability_sets(); variance_identity(); variance_components()
    log_jackknife_frontier(); synthetic_selection_phase(); compute_allocator(); disjoint_packing()
    pack_cross128_power()
    packed_unbiased_frontier()
    combined_source_scope()
    modern_model_transport()
    repeated_split_transport()
    prospective_source_c_transport()
    source_c_partition_boundary()
    print({"status": "complete", "figures": 21, "directory": str(OUT)})


if __name__ == "__main__":
    main()
