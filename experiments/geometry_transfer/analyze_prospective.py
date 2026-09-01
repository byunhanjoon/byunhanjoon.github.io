#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr,spearmanr

HERE=Path(__file__).resolve().parent;RAW=HERE/"raw"/"prospective";TABLE=HERE/"tables";FIG=HERE/"figures"


def stats(pred,actual):
    pred=np.asarray(pred,float);actual=np.asarray(actual,float)
    return {"pearson":float(pearsonr(pred,actual).statistic) if len(pred)>2 else float("nan"),
            "spearman":float(spearmanr(pred,actual).statistic) if len(pred)>2 else float("nan"),
            "mae":float(np.mean(np.abs(pred-actual))),"sign_accuracy":float(np.mean(np.sign(pred)==np.sign(actual))),
            "calibration_slope":float(np.polyfit(pred,actual,1)[0]) if np.std(pred)>0 else float("nan"),"n":len(pred)}


def loso(frame,p):
    out=np.zeros(len(frame))
    for source in frame.source.unique():
        tr=frame.source!=source;te=~tr;x=frame.loc[tr,p];y=frame.loc[tr,"actual_delta"]
        if np.std(x)>0:slope,intercept=np.polyfit(x,y,1);out[te]=intercept+slope*frame.loc[te,p]
        else:out[te]=y.mean()
    return stats(out,frame.actual_delta)


def main():
    cells=pd.read_csv(RAW/"cells.csv")
    aggregate=cells.groupby(["source","family","operator"],as_index=False).agg(predicted_delta=("predicted_delta","mean"),actual_delta=("actual_delta","mean"),predicted_se=("predicted_delta","std"),support_distance=("support_distance","mean"),cover_radius=("cover_radius","mean"),raw_smoothness=("raw_smoothness","mean"))
    primary=stats(aggregate.predicted_delta,aggregate.actual_delta)
    heuristic=[]
    for p in ["predicted_delta","support_distance","cover_radius","raw_smoothness"]:
        heuristic.append({"predictor":p,**loso(aggregate,p)})
    heuristic=pd.DataFrame(heuristic)
    source_behavior={}
    for source,g in cells.groupby("source"):
        source_behavior[source]={"spearman":float(spearmanr(g.predicted_delta,g.actual_delta).statistic),"sign_accuracy":float(np.mean(np.sign(g.predicted_delta)==np.sign(g.actual_delta)))}
    harmful=aggregate[aggregate.actual_delta<0];detected=harmful[(harmful.predicted_delta<=0)|(np.abs(harmful.predicted_delta)<=1.96*harmful.predicted_se.fillna(0))]
    p3=heuristic.loc[heuristic.predictor=="predicted_delta","spearman"].iloc[0]>heuristic.loc[heuristic.predictor!="predicted_delta","spearman"].max()
    success={"P1_spearman_at_least_060":primary["spearman"]>=.60,"P2_sign_at_least_075":primary["sign_accuracy"]>=.75,"P3_beats_heuristics":bool(p3),
             "P4_three_sources":sum(v["spearman"]>0 or v["sign_accuracy"]>=2/3 for v in source_behavior.values())>=3,
             "P5_detects_harm":len(harmful)==0 or len(detected)==len(harmful)}
    success["passed_all"]=all(success.values())
    summary={"primary":primary,"source_behavior":source_behavior,"harmful_aggregates":len(harmful),"harmful_detected":len(detected),"success":success,"runnable_sources":cells.source.nunique()}
    (RAW/"analysis.json").write_text(json.dumps(summary,indent=2)+"\n");aggregate.to_csv(TABLE/"prospective_results.csv",index=False);heuristic.to_csv(TABLE/"prospective_heuristics.csv",index=False)
    plt.figure(figsize=(6,5))
    for source,g in aggregate.groupby("source"):plt.scatter(g.predicted_delta,g.actual_delta,s=45,label=source)
    lo=aggregate[["predicted_delta","actual_delta"]].min().min();hi=aggregate[["predicted_delta","actual_delta"]].max().max();plt.plot([lo,hi],[lo,hi],"k--",lw=1);plt.axhline(0,color="gray",lw=.7);plt.axvline(0,color="gray",lw=.7)
    plt.xlabel("Nested state-CV predicted Δ");plt.ylabel("Sealed outer-state actual Δ");plt.legend(fontsize=7);plt.tight_layout();plt.savefig(FIG/"figure_10_prospective_confirmation.png",dpi=180);plt.savefig(FIG/"figure_10_prospective_confirmation.pdf");plt.close()
    print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
