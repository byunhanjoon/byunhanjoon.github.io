#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

HERE=Path(__file__).resolve().parent;rows=[]
retro=pd.read_csv(HERE/"raw/retrospective/cells.csv")
for r in retro.itertuples():
    rows.append({"stage":"retrospective","source":r.source,"task":r.task,"outer_state_split":r.split,"inner_fold":"row_oof_3fold","base_predictor":r.base_model,"geometry_metric":r.metric,"operator":r.operator,"operator_hyperparameters":"frozen_geometry_only","sample_size_condition":"full","seed":f"task_split_{r.split}","status":"complete"})
pros=pd.read_csv(HERE/"raw/prospective/cells.csv")
for r in pros.itertuples():
    rows.append({"stage":"prospective","source":r.family,"task":r.source,"outer_state_split":r.seed,"inner_fold":"nested_state_3fold+row_oof_3fold","base_predictor":"catboost","geometry_metric":"frozen_source_metric","operator":r.operator,"operator_hyperparameters":"frozen_geometry_only","sample_size_condition":"full","seed":r.seed,"status":"complete"})
gap=pd.read_csv(HERE/"raw/hierarchy_gap/cells.csv")
for r in gap.itertuples():
    rows.append({"stage":"prospective_gap","source":r.family,"task":r.source,"outer_state_split":r.seed,"inner_fold":"nested_state_3fold+row_oof_3fold","base_predictor":"catboost","geometry_metric":"NAICS_prefix_tree","operator":r.operator,"operator_hyperparameters":"separately_frozen_geometry_only","sample_size_condition":"full","seed":r.seed,"status":"complete"})
syn=pd.read_csv(HERE/"raw/synthetic/identity.csv")
for i,r in syn.iterrows():
    rows.append({"stage":"synthetic","source":"synthetic_circle","task":"exact_identity","outer_state_split":i,"inner_fold":"none","base_predictor":"known","geometry_metric":"circle","operator":r.operator,"operator_hyperparameters":"frozen","sample_size_condition":r.rows_per_state,"seed":20260829,"status":"complete"})
pd.DataFrame(rows).to_csv(HERE/"registry.csv",index=False)
