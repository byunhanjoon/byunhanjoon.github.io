#!/usr/bin/env python3
"""Aggregate the extended runs and evaluate the frozen success gates."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT=Path(__file__).resolve().parent;D1=ROOT/'direction_1';D2=ROOT/'direction_2';FIG=ROOT/'figures'


def py(v):
    if isinstance(v,dict):return {str(k):py(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [py(x) for x in v]
    if isinstance(v,np.ndarray):return py(v.tolist())
    if isinstance(v,(np.floating,float)):
        x=float(v);return x if math.isfinite(x) else None
    if isinstance(v,(np.integer,int)):return int(v)
    if isinstance(v,(np.bool_,bool)):return bool(v)
    return v


def records(df,keys,metrics):
    out=df.groupby(keys).agg(**{m+'_mean':(m,'mean') for m in metrics},**{m+'_std':(m,'std') for m in metrics},n=(metrics[0],'size')).reset_index()
    return out


def main():
    binary=pd.read_csv(D1/'pretrained_model_grid.csv');raw=pd.read_csv(D1/'raw_set_task_metrics.csv');continuous=pd.read_csv(D1/'continuous_tabpfn_metrics.csv');raw_cpairs=pd.read_csv(D1/'continuous_pair_metrics.csv');distractor=pd.read_csv(D1/'distractor_stress_metrics.csv')
    orientation=pd.read_csv(D2/'orientation_metrics.csv');downstream=pd.read_csv(D2/'downstream_metrics.csv');cross_o=pd.read_csv(D2/'crosssplit_orientation_metrics.csv');cross_d=pd.read_csv(D2/'crosssplit_downstream_metrics.csv');real=pd.read_csv(D2/'real_endpoint_transfer_metrics.csv');asym=pd.read_csv(D2/'real_endpoint_asymmetry.csv')
    bmain=binary[~binary.model.str.contains('shuffled')].copy();bshuf=binary[binary.model.str.contains('shuffled')].copy()
    bsum=records(bmain,['model','world'],['auroc','accuracy','log_loss','mean_confidence','causal_absolute_error','plugin_ate','runtime_seconds'])
    bcell=records(bmain,['model','world','cell_id','q','p','n_train'],['auroc','mean_confidence','causal_absolute_error','plugin_ate','paired_observational_summary_distance'])
    bpivot=bmain.pivot_table(index=['cell_id','n_train','data_seed','model'],columns='world',values=['plugin_ate','true_ate']).reset_index()
    bpivot.columns=['_'.join([str(x) for x in c if x!='']) if isinstance(c,tuple) else c for c in bpivot.columns]
    bpivot['predicted_world_difference']=(bpivot.plugin_ate_randomized_causal-bpivot.plugin_ate_hidden_confounding).abs();bpivot['true_world_difference']=(bpivot.true_ate_randomized_causal-bpivot.true_ate_hidden_confounding).abs()
    rsum=records(raw,['with_assumptions'],['absolute_error','ensemble_variance'])
    cmain=continuous[~continuous.model.str.contains('shuffled')].copy();csum=records(cmain,['model','world'],['predictive_rmse','predictive_r2','causal_absolute_error','plugin_ate','mean_80_interval_width','runtime_seconds'])
    cpairs=cmain[cmain.model=='tabpfn'].pivot_table(index=['cell','rho','effect_gap','n_train','seed'],columns='world',values='plugin_ate').reset_index()
    cpairs['tabpfn_plugin_effect_distance']=(cpairs.high_effect-cpairs.low_effect).abs()
    cpairs=cpairs.merge(raw_cpairs[['cell','rho','effect_gap','n_train','seed','observational_summary_distance']],on=['cell','rho','effect_gap','n_train','seed'],how='left',validate='one_to_one')
    cpairs.to_csv(D1/'continuous_pair_metrics.csv',index=False)
    ccell=records(cmain,['model','world','cell','rho','effect_gap','n_train'],['predictive_rmse','predictive_r2','causal_absolute_error','plugin_ate','mean_80_interval_width'])
    dstress=records(distractor,['nuisance_features','n_train','world'],['auroc','accuracy','mean_confidence','causal_absolute_error','plugin_ate','runtime_seconds'])
    osum=records(cross_o,['condition','method'],['correct_orientation','margin'])
    dsum=records(cross_d,['condition','method'],['auroc','log_loss'])
    realsum=records(real,['held_domain','method'],['rmse','r2'])
    for name,df in [('binary_summary',bsum),('binary_cells',bcell),('binary_pairs',bpivot),('rawset_summary',rsum),('continuous_summary',csum),('continuous_cells',ccell),('distractor_summary',dstress),('semantic_orientation_summary',osum),('semantic_downstream_summary',dsum),('real_endpoint_summary',realsum)]:df.to_csv(ROOT/(name+'.csv'),index=False)

    tab=bmain[bmain.model=='tabpfn'];inform=tab[tab.population_observational_association.abs()>=.4];hidden=tab[tab.world=='hidden_confounding'];shauc=bshuf.auroc.mean();rawm=raw.groupby('with_assumptions').absolute_error.mean()
    dclean=cross_d[cross_d.condition=='clean'].groupby('method').auroc.mean();oclean=cross_o[cross_o.condition=='clean'].groupby('method').correct_orientation.mean();dopaque=cross_d[cross_d.condition=='opaque_informative'].groupby('method').auroc.mean();dshuffle=cross_d[cross_d.condition=='shuffled_description'].groupby('method').auroc.mean();dneg=cross_d[cross_d.condition=='negated_description'].groupby('method').auroc.mean()
    encoders=['bge','e5','gte'];best=max(encoders,key=lambda x:oclean[x])
    gates={
      'direction1_tabpfn_informative_auc_ge_075':inform.auroc.mean()>=.75,
      'direction1_hidden_plugin_error_ge_030':hidden.causal_absolute_error.mean()>=.30,
      'direction1_tabpfn_confidence_ge_070':tab.mean_confidence.mean()>=.70,
      'direction1_shuffled_auc_near_chance':.45<=shauc<=.55,
      'direction1_assumptions_cut_rawset_mae_50pct':rawm[True]<=.5*rawm[False],
      'direction2_encoder_beats_tfidf_orientation_15pt':oclean[best]-oclean['tfidf']>=.15,
      'direction2_encoder_downstream_beats_structure_05':dclean[best]-dclean['structure_only']>=.05,
      'direction2_opaque_description_beats_structure_05':dopaque[best]-dopaque['structure_only']>=.05,
      'direction2_shuffle_destroys_gain':dshuffle[best]<dclean[best]-.15,
      'direction2_posthoc_negation_beats_tfidf_15pt':dneg['bge']-dneg['tfidf']>=.15,
    }
    summary={
      'gates':gates,
      'direction1':{'tabpfn_informative_auc':inform.auroc.mean(),'tabpfn_overall_auc':tab.auroc.mean(),'tabpfn_mean_confidence':tab.mean_confidence.mean(),'hidden_world_causal_mae':hidden.causal_absolute_error.mean(),'randomized_world_causal_mae':tab[tab.world=='randomized_causal'].causal_absolute_error.mean(),'shuffled_auc':shauc,'rawset_observation_only_mae':rawm[False],'rawset_assumption_aware_mae':rawm[True],'binary_predicted_world_difference':bpivot[bpivot.model=='tabpfn'].predicted_world_difference.mean(),'binary_true_world_difference':bpivot[bpivot.model=='tabpfn'].true_world_difference.mean(),'continuous_tabpfn_causal_mae':cmain[cmain.model=='tabpfn'].causal_absolute_error.mean(),'continuous_tabpfn_predictive_r2':cmain[cmain.model=='tabpfn'].predictive_r2.mean(),'continuous_predicted_world_difference':cpairs.tabpfn_plugin_effect_distance.mean(),'continuous_true_world_difference':cpairs.effect_gap.mean(),'distractor_by_nuisance':dstress.to_dict(orient='records')},
      'direction2':{'best_encoder_clean_orientation':best,'clean_orientation':oclean.to_dict(),'clean_downstream_auc':dclean.to_dict(),'opaque_downstream_auc':dopaque.to_dict(),'shuffled_downstream_auc':dshuffle.to_dict(),'posthoc_negated_downstream_auc':dneg.to_dict(),'real_endpoint_asymmetry':asym.to_dict(orient='records'),'real_endpoint_rmse':realsum.to_dict(orient='records')},
      'runtime':{'binary_seconds':json.loads((D1/'results.json').read_text())['runtime_seconds'],'continuous_seconds':json.loads((D1/'continuous_results.json').read_text())['runtime_seconds'],'distractor_seconds':json.loads((D1/'distractor_results.json').read_text())['runtime_seconds'],'semantic_main_seconds':json.loads((D2/'results.json').read_text())['runtime_seconds'],'semantic_crosssplit_seconds':json.loads((D2/'crosssplit_results.json').read_text())['runtime_seconds'],'real_panel_seconds':json.loads((D2/'real_panel_results.json').read_text())['runtime_seconds']},
      'errors':json.loads((D1/'results.json').read_text())['errors']+json.loads((D1/'continuous_results.json').read_text())['errors']+json.loads((D1/'distractor_results.json').read_text())['errors']+json.loads((D2/'results.json').read_text())['errors']+json.loads((D2/'crosssplit_results.json').read_text())['errors']+json.loads((D2/'real_panel_results.json').read_text())['errors'],
    }
    (ROOT/'summary.json').write_text(json.dumps(py(summary),indent=2,allow_nan=False)+'\n')

    tcell=bcell[(bcell.model=='tabpfn') & (bcell.world=='hidden_confounding')]
    fig,(a,b)=plt.subplots(1,2,figsize=(10,4.1))
    for n in sorted(tcell.n_train.unique()):
        q=tcell[tcell.n_train==n].sort_values('population_observational_association_mean' if 'population_observational_association_mean' in tcell else 'q')
        x=1-2*(q.q+q.p-2*q.q*q.p)
        a.plot(x,q.auroc_mean,marker='o',label=f'n={n}');b.plot(x,q.causal_absolute_error_mean,marker='o',label=f'n={n}')
    a.set(xlabel='population observational association',ylabel='TabPFN observational AUROC');b.set(xlabel='population observational association',ylabel='confounded-world plug-in ATE error');a.legend(frameon=False)
    fig.tight_layout();fig.savefig(FIG/'followup_direction1_tabpfn_cells.png',dpi=180);plt.close(fig)

    q=dsum[dsum.condition.isin(['clean','opaque_informative','negated_description','ambiguous','shuffled_description']) & dsum.method.isin(['structure_only','tfidf','bge','e5','gte','oracle'])]
    pivot=q.pivot(index='condition',columns='method',values='auroc_mean');methods=[x for x in ['structure_only','tfidf','bge','e5','gte','oracle'] if x in pivot]
    fig,ax=plt.subplots(figsize=(10,4.6));x=np.arange(len(pivot));w=.8/len(methods)
    for j,m in enumerate(methods):ax.bar(x-.4+w/2+j*w,pivot[m],w,label=m)
    ax.set_xticks(x,[s.replace('_','\n') for s in pivot.index]);ax.set_ylim(.2,1);ax.set_ylabel('cross-split downstream AUROC');ax.legend(frameon=False,ncol=3,fontsize=8)
    fig.tight_layout();fig.savefig(FIG/'followup_direction2_crosssplit.png',dpi=180);plt.close(fig)
    print(json.dumps(py(summary),indent=2),flush=True)


if __name__=='__main__':main()
