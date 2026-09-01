#!/usr/bin/env python3
"""Run the sealed prospective nested state-held-out confirmation."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from geometry_transfer import (decompose, empirical_gain, harmonic_operator,
    kernel_ridge_operator, knn_operator, median_bandwidth, rbf_operator,
    stable_seed, state_mean_variance, state_means)


HERE=Path(__file__).resolve().parent
DATA=HERE/"prospective_data";RAW=HERE/"raw"/"prospective"
CONFIG=json.loads((HERE/"PROSPECTIVE_CONFIG.json").read_text())


@dataclass
class Source:
    name:str; rows:pd.DataFrame; states:pd.DataFrame; distance:np.ndarray
    domain_distance:np.ndarray; manifest:dict
    @property
    def ids(self):return self.states.state_id.astype(str).tolist()
    def row_state(self):
        lookup={s:i for i,s in enumerate(self.ids)}
        return np.asarray([lookup[str(x)] for x in self.rows.field_state],int)


def load(name):
    folder=DATA/name;manifest=json.loads((folder/"manifest.json").read_text())
    if manifest["status"]!="RUN":raise RuntimeError(manifest["status"])
    d=np.load(folder/"distance_primary.npy");domain=np.load(folder/"distance_domain.npy") if (folder/"distance_domain.npy").exists() else d
    return Source(name,pd.read_parquet(folder/"rows.parquet"),pd.read_csv(folder/"states.csv",dtype={"state_id":str}),d,domain,manifest)


def feature_frame(source:Source,indices):
    x=source.rows.iloc[indices][source.manifest["ordinary_covariates"]].copy();cats=[]
    for i,c in enumerate(x):
        if not pd.api.types.is_numeric_dtype(x[c]):x[c]=x[c].astype("string").fillna("__MISSING__").astype(str);cats.append(i)
        else:x[c]=pd.to_numeric(x[c],errors="coerce")
    return x,cats


def subsample(indices,states,limit,rng):
    if len(indices)<=limit:return indices
    base=[rng.choice(indices[states[indices]==s]) for s in np.unique(states[indices])]
    pool=np.setdiff1d(indices,np.asarray(base));extra=rng.choice(pool,limit-len(base),replace=False)
    return np.sort(np.concatenate([base,extra]).astype(int))


def fit_predict(source,fit,predict,y,seed):
    fit=subsample(fit,source.row_state(),CONFIG["base_model"]["max_fit_rows"],np.random.default_rng(seed))
    combined=np.concatenate([fit,predict]);x,cats=feature_frame(source,combined)
    model=CatBoostRegressor(iterations=100,depth=7,learning_rate=.08,l2_leaf_reg=5.,loss_function="RMSE",
                            random_seed=seed,verbose=False,allow_writing_files=False,thread_count=8,random_strength=.25)
    model.fit(x.iloc[:len(fit)],y[fit],cat_features=cats)
    return np.asarray(model.predict(x.iloc[len(fit):]),float)


def residual_bundle(source,observed,query,seed):
    rs=source.row_state();tr=np.flatnonzero(np.isin(rs,observed));ur=np.flatnonzero(np.isin(rs,query))
    raw=pd.to_numeric(source.rows.target,errors="raise").to_numpy(float);center=raw[tr].mean();scale=raw[tr].std() or 1.;y=(raw-center)/scale
    rng=np.random.default_rng(stable_seed("prospective-rowfold",source.name,seed));sh=tr.copy();rng.shuffle(sh);folds=np.array_split(sh,3)
    oof=np.empty(len(tr));position={row:i for i,row in enumerate(tr.tolist())}
    for k,held in enumerate(folds):
        fit=np.setdiff1d(tr,held);pred=fit_predict(source,fit,held,y,stable_seed("prospective-oof",source.name,seed,k))
        oof[[position[r] for r in held]]=pred
    pu=fit_predict(source,tr,ur,y,stable_seed("prospective-full",source.name,seed))
    return {"tr":tr,"ur":ur,"st":rs[tr],"su":rs[ur],"rt":y[tr]-oof,"ru":y[ur]-pu,"yt":y[tr]}


def operators(source,train,query):
    h=median_bandwidth(source.distance,train)
    result={"knn_3":knn_operator(source.distance,train,query,3),"rbf":rbf_operator(source.distance,train,query,h)}
    if source.name in {"bls_oews_wage","census_cbp_naics"}:result["domain_specific"]=kernel_ridge_operator(source.distance,train,query,h)
    else:
        hd=median_bandwidth(source.domain_distance,train)
        result["domain_specific"]=harmonic_operator(source.domain_distance,train,query,hd)
    return result


def evaluate(bundle,train,query,ops):
    mt=state_means(bundle["rt"],bundle["st"],train);sig=state_mean_variance(bundle["rt"],bundle["st"],train)
    mu=state_means(bundle["ru"],bundle["su"],query);local={s:i for i,s in enumerate(query.tolist())};su=np.asarray([local[s] for s in bundle["su"]])
    output={}
    for name,a in ops.items():
        dec=decompose(mu,mt,a,sig);gain=empirical_gain(bundle["ru"],su,a@mt)
        output[name]={"actual_delta":gain,"oracle_delta":dec.delta,"transferable_signal":dec.transferable_signal,"noise_cost":dec.noise_cost}
    return output


def inner_prediction(source,outer_train,split_seed):
    ordered=outer_train.copy();rng=np.random.default_rng(stable_seed("inner-state-fold",source.name,split_seed));rng.shuffle(ordered);folds=np.array_split(ordered,3)
    scores={name:[] for name in CONFIG["operators"]};supports=[];covers=[];rawsmooth=[]
    for k,held in enumerate(folds):
        train=np.setdiff1d(outer_train,held);bundle=residual_bundle(source,train,held,stable_seed("inner",source.name,split_seed,k))
        result=evaluate(bundle,train,held,operators(source,train,held))
        for name in scores:scores[name].append(result[name]["actual_delta"])
        block=source.distance[np.ix_(held,train)];supports.append(float(np.mean(np.min(block,axis=1))));covers.append(float(np.max(np.min(block,axis=1))))
        means=state_means(bundle["yt"],bundle["st"],train);d=source.distance[np.ix_(train,train)];h=median_bandwidth(d,np.arange(len(train)));w=np.exp(-.5*(d/max(h,1e-12))**2);np.fill_diagonal(w,0)
        rawsmooth.append(float(np.sum(w*(means[:,None]-means[None,:])**2)/max(np.sum(w),1e-12)))
    return {name:{"predicted_delta":float(np.mean(values)),"fold_deltas":values,"se":float(np.std(values,ddof=1)/np.sqrt(len(values)))} for name,values in scores.items()}, {"support_distance":float(np.mean(supports)),"cover_radius":float(np.mean(covers)),"raw_smoothness":float(np.mean(rawsmooth))}


def hash_payload(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def run_source(source):
    all_rows=[];ids=np.arange(len(source.states));state_rows=source.row_state()
    for seed in CONFIG["outer_seeds"]:
        rng=np.random.default_rng(stable_seed("outer-state-split",source.name,seed));order=ids.copy();rng.shuffle(order);cut=max(3,int(round(.7*len(order))));cut=min(cut,len(order)-2);train=np.sort(order[:cut]);test=np.sort(order[cut:])
        prediction_path=RAW/"sealed_predictions"/f"{source.name}__seed{seed}.json"
        if prediction_path.exists():sealed=json.loads(prediction_path.read_text())
        else:
            predicted,heur=inner_prediction(source,train,seed)
            sealed={"source":source.name,"family":source.manifest["family"],"seed":seed,"train_state_ids":[source.ids[i] for i in train],"test_state_ids":[source.ids[i] for i in test],"predictions":predicted,"heuristics":heur,"outer_test_outcomes_accessed":False}
            sealed["payload_sha256"]=hash_payload(sealed);prediction_path.parent.mkdir(parents=True,exist_ok=True)
            tmp=prediction_path.with_suffix(".tmp");tmp.write_text(json.dumps(sealed,indent=2,sort_keys=True)+"\n");os.replace(tmp,prediction_path)
        # Reveal and evaluate only after the prediction artifact exists.
        bundle=residual_bundle(source,train,test,stable_seed("outer",source.name,seed));actual=evaluate(bundle,train,test,operators(source,train,test))
        for name in CONFIG["operators"]:
            all_rows.append({"source":source.name,"family":source.manifest["family"],"seed":seed,"operator":name,
                             "train_states":len(train),"test_states":len(test),"rows":len(source.rows),
                             "predicted_delta":sealed["predictions"][name]["predicted_delta"],"predicted_se":sealed["predictions"][name]["se"],
                             "actual_delta":actual[name]["actual_delta"],"oracle_delta":actual[name]["oracle_delta"],
                             **sealed["heuristics"],"seal_sha256":sealed["payload_sha256"]})
        print(source.name,seed,flush=True)
    return all_rows


def main():
    RAW.mkdir(parents=True,exist_ok=True);rows=[];availability=[]
    for name in CONFIG["sources"]:
        try:
            source=load(name);rows.extend(run_source(source));availability.append({"source":name,"status":"RUN"})
        except Exception as exc:
            availability.append({"source":name,"status":"NOT RUN — SOURCE UNAVAILABLE","error":str(exc)});print(availability[-1],flush=True)
    pd.DataFrame(rows).to_csv(RAW/"cells.csv",index=False);(RAW/"availability.json").write_text(json.dumps(availability,indent=2)+"\n")


if __name__=="__main__":main()
