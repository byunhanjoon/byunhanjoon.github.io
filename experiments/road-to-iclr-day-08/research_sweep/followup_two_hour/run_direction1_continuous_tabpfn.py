#!/usr/bin/env python3
"""Alternative continuous-SCM TabPFN regression audit."""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tabpfn import TabPFNRegressor


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "direction_1"
CELLS = [(0.25, 0.6), (0.25, 1.0), (0.35, 0.8), (0.35, 1.2), (0.50, 0.6), (0.50, 1.0)]
SIZES = [128, 512, 2048]


def js(v):
    if isinstance(v, dict): return {str(k): js(x) for k,x in v.items()}
    if isinstance(v, (list,tuple)): return [js(x) for x in v]
    if isinstance(v,np.ndarray): return js(v.tolist())
    if isinstance(v,(np.floating,float)):
        x=float(v);return x if math.isfinite(x) else None
    if isinstance(v,(np.integer,int)): return int(v)
    return v


def data(rng,n,rho):
    xy=rng.multivariate_normal([0,0],[[1,rho],[rho,1]],size=n)
    z=rng.normal(size=(n,4))
    return np.c_[xy[:,0],z].astype(np.float32),xy[:,1].astype(np.float32)


def summary(x,y):
    return np.r_[x.mean(0),x.std(0),y.mean(),y.std(),np.corrcoef(x[:,0],y)[0,1],np.quantile(x[:,0],[.1,.5,.9]),np.quantile(y,[.1,.5,.9])]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--device',default='cuda:1');ap.add_argument('--seeds',type=int,default=30);ap.add_argument('--estimators',type=int,default=16);args=ap.parse_args()
    start=time.time();records=[];pairs=[];total=len(CELLS)*len(SIZES)*args.seeds;done=0
    checkpoint=OUT/'continuous_tabpfn_checkpoint.json'
    for cell,(rho,gap) in enumerate(CELLS):
      taus=[rho-gap/2,rho+gap/2]
      # Verify both latent parameterizations have valid positive residual noise.
      for tau in taus:
        gamma=(rho-tau)/.8; residual=1-tau*tau-gamma*gamma-2*tau*gamma*.8
        if residual<=0: raise ValueError((rho,gap,tau,residual))
      for n in SIZES:
       for seed in range(args.seeds):
        summaries=[];tabpfn_plugins=[]
        for wi,(world,tau) in enumerate(zip(['low_effect','high_effect'],taus)):
            rng=np.random.default_rng(7_000_000+cell*100_000+n*10+seed*2+wi)
            x,y=data(rng,n,rho);xt,yt=data(np.random.default_rng(8_000_000+cell*100_000+n*10+seed*2+wi),2048,rho)
            z=np.random.default_rng(9_000_000+cell*100_000+n*10+seed*2+wi).normal(size=(512,4)).astype(np.float32)
            x0=np.c_[np.zeros(512),z].astype(np.float32);x1=np.c_[np.ones(512),z].astype(np.float32);xall=np.vstack([xt,x0,x1])
            summaries.append(summary(x,y))
            for model_name in ['ridge','tabpfn']:
                t0=time.time()
                if model_name=='ridge':
                    model=make_pipeline(StandardScaler(),Ridge(alpha=1.)).fit(x,y);pred=model.predict(xall);qwidth=np.full(len(pred),np.nan)
                else:
                    model=TabPFNRegressor(n_estimators=args.estimators,device=args.device,fit_mode='fit_preprocessors',random_state=seed)
                    model.fit(x,y);full=model.predict(xall,output_type='full');pred=full['mean'];q=full['quantiles'];qwidth=q[-1]-q[0]
                plugin=pred[len(xt)+512:].mean()-pred[len(xt):len(xt)+512].mean()
                if model_name=='tabpfn': tabpfn_plugins.append(plugin)
                records.append({'cell':cell,'rho':rho,'effect_gap':gap,'n_train':n,'seed':seed,'world':world,'model':model_name,'true_ate':tau,'plugin_ate':plugin,'causal_absolute_error':abs(plugin-tau),'predictive_rmse':np.sqrt(mean_squared_error(yt,pred[:len(xt)])),'predictive_mae':mean_absolute_error(yt,pred[:len(xt)]),'predictive_r2':r2_score(yt,pred[:len(xt)]),'mean_80_interval_width':np.nanmean(qwidth[:len(xt)]) if model_name=='tabpfn' else None,'runtime_seconds':time.time()-t0})
                del model;gc.collect();torch.cuda.empty_cache()
            if seed<3:
                shuf=y[rng.permutation(len(y))];m=TabPFNRegressor(n_estimators=max(4,args.estimators//2),device=args.device,fit_mode='fit_preprocessors',random_state=seed+5000);m.fit(x,shuf);pred=m.predict(xt)
                records.append({'cell':cell,'rho':rho,'effect_gap':gap,'n_train':n,'seed':seed,'world':world,'model':'tabpfn_shuffled_outcome','true_ate':tau,'plugin_ate':None,'causal_absolute_error':None,'predictive_rmse':np.sqrt(mean_squared_error(yt,pred)),'predictive_mae':mean_absolute_error(yt,pred),'predictive_r2':r2_score(yt,pred),'mean_80_interval_width':None,'runtime_seconds':None})
                del m;torch.cuda.empty_cache()
        pairs.append({'cell':cell,'rho':rho,'effect_gap':gap,'n_train':n,'seed':seed,'observational_summary_distance':np.linalg.norm(summaries[0]-summaries[1]),'tabpfn_plugin_effect_distance':abs(tabpfn_plugins[0]-tabpfn_plugins[1])})
        done+=1
        if done%3==0:
            checkpoint.write_text(json.dumps(js({'records':records,'pairs':pairs}),indent=2)+'\n');elapsed=time.time()-start;print(f'continuous {done}/{total} elapsed={elapsed/60:.1f}m eta={elapsed/done*(total-done)/60:.1f}m',flush=True)
    pd.DataFrame(records).to_csv(OUT/'continuous_tabpfn_metrics.csv',index=False);pd.DataFrame(pairs).to_csv(OUT/'continuous_pair_metrics.csv',index=False)
    result={'parameters':vars(args),'cells':CELLS,'sizes':SIZES,'records':records,'pairs':pairs,'runtime_seconds':time.time()-start,'errors':[]}
    (OUT/'continuous_results.json').write_text(json.dumps(js(result),indent=2,allow_nan=False)+'\n')
    print(f'continuous direction1 complete in {(time.time()-start)/60:.1f} minutes',flush=True)


if __name__=='__main__':main()
