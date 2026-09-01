#!/usr/bin/env python3
"""Analyze retrospective cells, apply Gate R, and regenerate figures/tables."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


HERE = Path(__file__).resolve().parent
RAW = HERE / "raw" / "retrospective"
FIG = HERE / "figures"
TABLE = HERE / "tables"


def finite_corr(x, y, kind="spearman") -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return float("nan")
    return float((spearmanr if kind == "spearman" else pearsonr)(x[ok], y[ok]).statistic)


def metrics(pred, actual) -> dict:
    pred, actual = np.asarray(pred, float), np.asarray(actual, float)
    ok = np.isfinite(pred) & np.isfinite(actual); pred, actual = pred[ok], actual[ok]
    slope = float(np.polyfit(pred, actual, 1)[0]) if len(pred) > 1 and np.std(pred) else float("nan")
    return {
        "n": len(pred), "pearson": finite_corr(pred, actual, "pearson"),
        "spearman": finite_corr(pred, actual),
        "r2_identity": float(1 - np.sum((actual-pred)**2) / np.sum((actual-actual.mean())**2)),
        "calibration_slope": slope, "mae": float(np.mean(np.abs(actual-pred))),
        "sign_accuracy": float(np.mean(np.sign(pred) == np.sign(actual))),
    }


def source_bootstrap(frame: pd.DataFrame, repeats=2000) -> dict:
    rng = np.random.default_rng(20260829); sources = frame.source.unique(); rows=[]
    for _ in range(repeats):
        selected=rng.choice(sources,len(sources),replace=True)
        sample=pd.concat([frame[frame.source==s] for s in selected],ignore_index=True)
        rows.append(metrics(sample.delta_theory,sample.delta_actual))
    return {key:[float(np.nanquantile([r[key] for r in rows],.025)),float(np.nanquantile([r[key] for r in rows],.975))]
            for key in ("pearson","spearman","calibration_slope","mae","sign_accuracy")}


def loso_prediction(frame: pd.DataFrame, predictor: str, identity=False) -> tuple[np.ndarray, dict]:
    output=np.empty(len(frame),float)
    for source in frame.source.unique():
        train=frame.source!=source; test=~train
        x=frame.loc[train,predictor].to_numpy(float); y=frame.loc[train,"delta_actual"].to_numpy(float)
        xt=frame.loc[test,predictor].to_numpy(float)
        if identity:
            output[test]=xt
        elif np.std(x)>0:
            slope,intercept=np.polyfit(x,y,1); output[test]=intercept+slope*xt
        else:
            output[test]=np.mean(y)
    return output, metrics(output,frame.delta_actual)


def make_figures(frame: pd.DataFrame, states: pd.DataFrame, comparison: pd.DataFrame) -> None:
    FIG.mkdir(exist_ok=True)
    plt.figure(figsize=(8,3.5)); plt.axis("off")
    labels=[("ordinary b(Z)",.08), ("residual signal μs",.28), ("geometry A",.48), ("transferable signal",.68), ("minus noise → help/harm",.88)]
    for label,x in labels:
        plt.text(x,.55,label,ha="center",va="center",bbox=dict(boxstyle="round",fc="#eef4ff"),transform=plt.gca().transAxes)
    for (_,x1),(_,x2) in zip(labels[:-1],labels[1:]): plt.annotate("",xy=(x2-.08,.55),xytext=(x1+.08,.55),arrowprops=dict(arrowstyle="->"),xycoords="axes fraction")
    plt.savefig(FIG/"figure_1_theory_diagram.png",dpi=180,bbox_inches="tight"); plt.savefig(FIG/"figure_1_theory_diagram.pdf",bbox_inches="tight"); plt.close()

    plt.figure(figsize=(6,5))
    for source,g in frame.groupby("source"): plt.scatter(g.delta_theory,g.delta_actual,s=14,alpha=.65,label=source)
    lo=frame[["delta_theory","delta_actual"]].min().min(); hi=frame[["delta_theory","delta_actual"]].max().max(); plt.plot([lo,hi],[lo,hi],"k--",lw=1)
    plt.xlabel("Population plug-in Δ theory");plt.ylabel("Actual state-balanced MSE gain");plt.legend(fontsize=7,ncol=2)
    plt.tight_layout();plt.savefig(FIG/"figure_6_retrospective_law.png",dpi=180);plt.savefig(FIG/"figure_6_retrospective_law.pdf");plt.close()

    order=comparison.sort_values("spearman").predictor
    p=comparison.set_index("predictor").loc[order]
    plt.figure(figsize=(7,4));plt.barh(order,p.spearman);plt.xlabel("LO-source Spearman");plt.axvline(0,color="black",lw=.7)
    plt.tight_layout();plt.savefig(FIG/"figure_7_theory_vs_heuristics.png",dpi=180);plt.savefig(FIG/"figure_7_theory_vs_heuristics.pdf");plt.close()

    representatives=pd.concat([frame.nlargest(3,"delta_actual"),frame.nsmallest(3,"delta_actual")])
    x=np.arange(len(representatives)); possible=representatives.possible_signal.to_numpy(); approx=representatives.approximation_error.to_numpy(); noise=representatives.noise_cost.to_numpy()
    plt.figure(figsize=(9,4));plt.bar(x,possible,label="available signal");plt.bar(x,-approx,label="approximation/mistransfer");plt.bar(x,-noise,bottom=-approx,label="noise cost")
    plt.axhline(0,color="black",lw=.8);plt.xticks(x,[f"{r.task}\n{r.operator}" for r in representatives.itertuples()],rotation=25,ha="right",fontsize=7);plt.legend()
    plt.tight_layout();plt.savefig(FIG/"figure_8_real_decomposition.png",dpi=180);plt.savefig(FIG/"figure_8_real_decomposition.pdf");plt.close()

    sample=states.sample(min(10000,len(states)),random_state=20260829)
    plt.figure(figsize=(6,5));sc=plt.scatter(sample.support_distance,sample.delta_actual_state,c=sample.local_transferable_signal-sample.local_noise_cost,s=7,alpha=.35,cmap="coolwarm")
    plt.xlabel("Nearest support distance");plt.ylabel("Per-state actual Δ");plt.colorbar(sc,label="Local theory Δ")
    plt.tight_layout();plt.savefig(FIG/"figure_9_state_effects.png",dpi=180);plt.savefig(FIG/"figure_9_state_effects.pdf");plt.close()


def main() -> None:
    frame=pd.read_csv(RAW/"cells.csv"); states=pd.read_csv(RAW/"state_cells.csv")
    TABLE.mkdir(exist_ok=True)
    predictors=["delta_theory","nearest_support_distance","cover_radius","raw_smoothness","conditional_smoothness","dirichlet_energy"]
    comparisons=[]; predictions={}
    for p in predictors:
        pred,score=loso_prediction(frame,p,identity=(p=="delta_theory"));predictions[p]=pred
        comparisons.append({"predictor":p,**score})
    comparison=pd.DataFrame(comparisons)
    primary=metrics(frame.delta_theory,frame.delta_actual)
    realized=metrics(frame.delta_realized_oracle,frame.delta_actual)
    ci=source_bootstrap(frame)
    per_source={source:metrics(g.delta_theory,g.delta_actual) for source,g in frame.groupby("source")}
    harmful=frame[frame.delta_actual<0]
    explained=harmful[(harmful.transferable_signal<=0)|(harmful.noise_cost>harmful.transferable_signal)]
    gate={
        "R1_spearman_at_least_070":primary["spearman"]>=.70,
        "R2_sign_accuracy_at_least_075":primary["sign_accuracy"]>=.75,
        "R3_beats_required_heuristics":all(comparison.loc[comparison.predictor=="delta_theory","spearman"].iloc[0] > comparison.loc[comparison.predictor==p,"spearman"].iloc[0] for p in predictors[1:5]),
        "R4_not_one_source_only":sum(np.isfinite(v["spearman"]) and v["spearman"]>0 for v in per_source.values())>=max(3,len(per_source)-1),
        "R5_explains_some_harm":len(explained)>0,
    }
    gate["passed_all"]=all(gate.values())
    summary={"primary":primary,"oracle_realized_arithmetic":realized,"source_bootstrap_95":ci,
             "per_source":per_source,"harmful_cells":len(harmful),"harmful_explained":len(explained),"gate":gate,
             "important_limitation":"delta_theory uses held-out residual means retrospectively; the realized-oracle relation is algebraic, not prospective prediction"}
    (RAW/"analysis.json").write_text(json.dumps(summary,indent=2)+"\n")
    comparison.to_csv(TABLE/"heuristic_comparison.csv",index=False)
    cols=["source","task","operator","delta_actual","delta_theory","transferable_signal","noise_cost","gtr","nearest_support_distance","conditional_smoothness"]
    main=frame[cols].copy();main["sign_correct"]=np.sign(main.delta_actual)==np.sign(main.delta_theory);main.to_csv(TABLE/"main_results.csv",index=False)
    frame.groupby("source")[["transferable_signal","noise_cost","delta_theory","delta_actual"]].mean().to_csv(TABLE/"source_decomposition.csv")
    frame.groupby(["source","task"],as_index=False).agg(
        field=("field","first"),metric=("metric","first"),train_states=("train_states","first"),
        test_states=("test_states","first"),rows=("train_rows","first"),operators=("operator","nunique")
    ).to_csv(TABLE/"retrospective_dataset_panel.csv",index=False)
    for source_name,target_name in (("metric_perturbation.csv","metric_perturbation.csv"),("real_sample_size.csv","sample_size_phase.csv"),("conditioning.csv","conditioning.csv")):
        path=RAW/source_name
        if path.exists(): pd.read_csv(path).to_csv(TABLE/target_name,index=False)
    # Similar-support opposite-outcome state cases.
    cases=[]
    for (task,split,operator),g in states.groupby(["task","split","operator"]):
        if len(g)<2: continue
        ordered=g.sort_values("support_distance").reset_index(drop=True)
        for i in range(len(ordered)-1):
            a,b=ordered.iloc[i],ordered.iloc[i+1]
            if np.sign(a.delta_actual_state)!=np.sign(b.delta_actual_state):
                cases.append({"task":task,"split":split,"operator":operator,"state_a":a.state_id,"state_b":b.state_id,
                              "support_a":a.support_distance,"support_b":b.support_distance,"delta_a":a.delta_actual_state,"delta_b":b.delta_actual_state,
                              "local_theory_a":a.delta_theory_state,"local_theory_b":b.delta_theory_state})
    pd.DataFrame(cases).sort_values(["task","split"]).head(100).to_csv(TABLE/"support_distance_case_studies.csv",index=False)
    make_figures(frame,states,comparison)
    print(json.dumps(summary,indent=2))


if __name__=="__main__": main()
