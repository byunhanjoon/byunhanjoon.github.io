#!/usr/bin/env python3
"""Weak/medium/strong base-predictor conditioning diagnostic."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent/"mpe_iclr"))
from representations import load_task,split_state_indices
from run_retrospective import base_residuals,fit_predict
from geometry_transfer import decompose,empirical_gain,operator_family,stable_seed,state_mean_variance,state_means


def medium_bundle(task,split):
    parts=split_state_indices(task,split);t=np.concatenate([parts["train"],parts["validation"]]);u=parts["test"];rs=task.row_state_indices();tr=np.flatnonzero(np.isin(rs,t));ur=np.flatnonzero(np.isin(rs,u))
    raw=pd.to_numeric(task.rows.target).to_numpy(float);center=raw[tr].mean();scale=raw[tr].std() or 1.;y=(raw-center)/scale
    rng=np.random.default_rng(stable_seed("conditioning",task.name));sh=tr.copy();rng.shuffle(sh);folds=np.array_split(sh,3);oof=np.empty(len(tr));pos={r:i for i,r in enumerate(tr.tolist())}
    for k,held in enumerate(folds):
        pred=fit_predict(task,np.setdiff1d(tr,held),held,y,stable_seed("medium",task.name,k),iterations=30);oof[[pos[r] for r in held]]=pred
    pu=fit_predict(task,tr,ur,y,stable_seed("medium-full",task.name),iterations=30)
    return {"t_states":t,"u_states":u,"row_state_t":rs[tr],"row_state_u":rs[ur],"residual_t":y[tr]-oof,"residual_u":y[ur]-pu,"target_t":y[tr],"target_u":y[ur]}


def evaluate(task,bundle,label):
    t,u=bundle["t_states"],bundle["u_states"];st,su=bundle["row_state_t"],bundle["row_state_u"];rt,ru=bundle["residual_t"],bundle["residual_u"]
    mt=state_means(rt,st,t);mu=state_means(ru,su,u);sig=state_mean_variance(rt,st,t);a=operator_family(task.distance,t,u)["rbf"];d=decompose(mu,mt,a,sig);local={s:i for i,s in enumerate(u.tolist())};sul=np.asarray([local[s] for s in su])
    return {"task":task.name,"source":task.manifest["source_unit"],"base_strength":label,"residual_state_signal":float(np.mean(mu**2)),"transferable_signal":d.transferable_signal,"noise_cost":d.noise_cost,"gtr":d.gtr,"delta_theory":d.delta,"delta_actual":empirical_gain(ru,sul,a@mt)}


def main():
    rows=[]
    for name in ["acs_occupation","tlc_pickup_zone","medical_charges"]:
        task=load_task(name);strong=base_residuals(task,0);medium=medium_bundle(task,0)
        weak={k:np.copy(v) for k,v in strong.items()};weak["residual_t"]=strong["target_t"];weak["residual_u"]=strong["target_u"]
        for label,bundle in (("weak_intercept",weak),("medium_catboost_30",medium),("strong_catboost_140",strong)):rows.append(evaluate(task,bundle,label))
    out=HERE/"raw"/"retrospective"/"conditioning.csv";pd.DataFrame(rows).to_csv(out,index=False);print(pd.DataFrame(rows).to_string(index=False))


if __name__=="__main__":main()
