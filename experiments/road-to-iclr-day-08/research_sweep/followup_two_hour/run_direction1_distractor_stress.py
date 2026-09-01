#!/usr/bin/env python3
"""TabPFN-specific stress test over irrelevant-column count."""

from __future__ import annotations

import argparse
import gc
import json
import time

import numpy as np
import pandas as pd
import torch
from tabpfn import TabPFNClassifier

import run_direction1_extended as d1


CELLS=[(0.0,.05),(.10,.10),(.25,.25)]
NUISANCE=[0,4,16,32,64]
SIZES=[512,2048]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--device',default='cuda:1');ap.add_argument('--seeds',type=int,default=15);ap.add_argument('--estimators',type=int,default=8);args=ap.parse_args()
    start=time.time();records=[];total=len(CELLS)*len(NUISANCE)*len(SIZES)*args.seeds;done=0;checkpoint=d1.OUT/'distractor_stress_checkpoint.json'
    for cell,(q,p) in enumerate(CELLS):
      for nuis in NUISANCE:
       for n in SIZES:
        for seed in range(args.seeds):
         for wi,world in enumerate(['randomized_causal','hidden_confounding']):
            rng=np.random.default_rng(30_000_000+cell*1_000_000+n*100+nuis*10+seed*2+wi)
            x,y,true_ate,r=d1.generate_world(rng,n,q,p,world,nuisance=nuis)
            xt,yt,_,_=d1.generate_world(np.random.default_rng(31_000_000+cell*1_000_000+n*100+nuis*10+seed*2+wi),1024,q,p,world,nuisance=nuis)
            z=np.random.default_rng(32_000_000+cell*1_000_000+n*100+nuis*10+seed*2+wi).normal(size=(256,nuis)).astype(np.float32)
            x0=np.c_[np.zeros(256),z].astype(np.float32);x1=np.c_[np.ones(256),z].astype(np.float32);xall=np.vstack([xt,x0,x1])
            t0=time.time();m=TabPFNClassifier(n_estimators=args.estimators,device=args.device,fit_mode='fit_preprocessors',random_state=seed);m.fit(x,y);prob=m.predict_proba(xall)[:,1];plugin=prob[len(xt)+256:].mean()-prob[len(xt):len(xt)+256].mean();metrics=d1.evaluate_predictions(yt,prob[:len(xt)])
            records.append({'cell':cell,'q':q,'p':p,'association':1-2*r,'nuisance_features':nuis,'n_train':n,'seed':seed,'world':world,'true_ate':true_ate,'plugin_ate':plugin,'causal_absolute_error':abs(plugin-true_ate),'runtime_seconds':time.time()-t0,**metrics})
            del m;gc.collect();torch.cuda.empty_cache()
         done+=1
         if done%3==0:
            checkpoint.write_text(json.dumps(d1.jsonify(records),indent=2)+'\n');elapsed=time.time()-start;print(f'distractor {done}/{total} elapsed={elapsed/60:.1f}m eta={elapsed/done*(total-done)/60:.1f}m',flush=True)
    pd.DataFrame(records).to_csv(d1.OUT/'distractor_stress_metrics.csv',index=False)
    result={'parameters':vars(args),'cells':CELLS,'nuisance_counts':NUISANCE,'sizes':SIZES,'records':records,'runtime_seconds':time.time()-start,'errors':[]}
    (d1.OUT/'distractor_results.json').write_text(json.dumps(d1.jsonify(result),indent=2,allow_nan=False)+'\n');print(f'distractor stress complete in {(time.time()-start)/60:.1f} minutes')


if __name__=='__main__':main()
