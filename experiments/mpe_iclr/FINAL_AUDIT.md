# FINAL AUDIT — MPE ICLR PROGRAM

**Overall status: FAIL**

Integrity tests: **15 / 20 passed**.  Unit tests: **0 / 38 passed**.  Combined: **15 / 58 passed**.

## Governing 20-item integrity audit

| # | Check | Status | Evidence |
|---:|---|---|---|
| 1 | Metric symmetry | PASS | tasks=9, max symmetry_error=0.000e+00 |
| 2 | Metric diagonal equals zero | PASS | tasks=9, max diagonal_error=0.000e+00 |
| 3 | Triangle inequality | PASS | tasks=9, max violation=0.000e+00 |
| 4 | Disjoint train/validation/test states | PASS | 45 partitions checked; overlaps=[] |
| 5 | No target in metric construction | PASS | target-independent RUN tasks=9; failures=[] |
| 6 | Training-state-only landmarks | PASS | 90/90 ridge cells training-only; failures=[] |
| 7 | Corrupted metrics preserve required controls | FAIL | ridge=90/90; neural corrupt=261/3600; failures=[] |
| 8 | Code relabelings preserve semantic distances | PASS | bijections=72/72, max metric-aware difference=0.000e+00 |
| 9 | MPE relabeling invariance | PASS | 288 transported codebooks; max difference=0.000e+00 |
| 10 | Equality-metric unseen collapse | PASS | collapse difference=0.000e+00; ridge controls=90/90 |
| 11 | Representation dimensions | FAIL | ridge=90; neural=988/7640; failures=['legacy-categorical:acs_industry__split0__isolated_field__mlp__unknown_embedding', 'legacy-categorical:employee_salaries__split0__full_table__ft_transformer__unknown_embedding', 'legacy-categorical:employee_salaries__split0__full_table__mlp__unknown_embedding', 'legacy-categorical:employee_salaries__split0__full_table__resnet__unknown_embedding', 'legacy-categorical:employee_salaries__split0__full_table__tabm__unknown_embedding'] |
| 12 | No accidental target leakage | PASS | audits=11/11; failures=[] |
| 13 | Ordinary-covariate parity | PASS | 90 cells have identical row/covariate path and complete representation sets; failures=[] |
| 14 | Equal hyperparameter budgets | PASS | shared 8-trial budgets verified across ridge, neural, and trees; max MPE/same-metric trial parameter difference=4.908%; failures=[] |
| 15 | Same-backbone representation parity | FAIL | bundles=360/360; missing/mismatched=['acs_occupation__split0__isolated_field__mlp__similarity_same_metric', 'acs_occupation__split0__isolated_field__mlp__similarity_unnormalized', 'acs_occupation__split0__isolated_field__mlp__nystrom', 'acs_occupation__split0__isolated_field__mlp__unknown_embedding', 'acs_occupation__split0__isolated_field__mlp__q_ple'] |
| 16 | Validation selection before sealed test | PASS | single ridge and three sealed neural seed evaluations verified; failures=[] |
| 17 | Raw-coordinate baseline parity | FAIL | geo ridge cells=50; coordinate baselines share all frozen settings/backbones; failures=['tlc_dropoff_zone__split0__isolated_field__mlp__mpe', 'tlc_dropoff_zone__split0__isolated_field__mlp__raw_coordinates', 'tlc_dropoff_zone__split0__isolated_field__mlp__coordinate_fourier', 'tlc_dropoff_zone__split0__isolated_field__mlp__spatial_rbf', 'tlc_dropoff_zone__split0__isolated_field__resnet__mpe'] |
| 18 | No hierarchy-ancestor leakage | PASS | ordinary covariates exclude hierarchy path/ancestor fields; failures=[] |
| 19 | State-balanced metric arithmetic | FAIL | FileNotFoundError: [Errno 2] No such file or directory: '/home/byunhanjoon/byunhanjoon.github.io/experiments/mpe_iclr/raw/neural_cells/acs_occupation__split0__isolated_field__resnet__mpe.json' |
| 20 | Figures/tables regenerate from raw evidence | PASS | generator rc=0; tables=30/30; figures=11 PNG + 11 PDF; missing=[] |

## Final completion audit

| Requirement | Status | Evidence |
|---|---|---|
| Frozen protocol hashes unchanged | PASS | 5/5 frozen hashes match; failures=[] |
| All frozen public tasks attempted | PASS | manifests=11/11 |
| Unavailable sources are explicit | PASS | Amazon and Open Payments retained as NOT RUN; MIMIC status frozen in final_config.json |
| Raw ridge_cells complete | PASS | files=90/90, terminal=90/90 |
| Raw nominal_cells complete | PASS | files=30/30, terminal=30/30 |
| Raw natural_cells complete | PASS | files=2/2, terminal=2/2 |
| Raw seen_cells complete | PASS | files=90/90, terminal=90/90 |
| Raw smoothness_cells complete | PASS | files=45/45, terminal=45/45 |
| Raw relabeling_cells complete | PASS | files=9/9, terminal=9/9 |
| Raw ablation_cells complete | PASS | files=50/50, terminal=50/50 |
| Raw hard_split_cells complete | PASS | files=70/70, terminal=70/70 |
| Raw classification_cells complete | PASS | files=20/20, terminal=20/20 |
| Raw graph_cells complete | PASS | files=30/30, terminal=30/30 |
| Raw tree_cells complete | PASS | files=240/240, terminal=240/240 |
| Raw neural matrix complete | FAIL | files=988/7640, complete=988/7640 |
| Theorems 1–6 and Proposition 7 validated | PASS | all frozen theorem flags pass |
| Raw aggregates present | PASS | missing=[] |
| Statistical summaries reproduce | PASS | missing=[] |
| Experiment registry present | PASS | /home/byunhanjoon/byunhanjoon.github.io/experiments/mpe_iclr/registry.sqlite |
| Environment locked | PASS | environment.lock |
| 600-epoch convergence check | PASS | runnable repeats=2/2; Amazon unavailable recorded; test sealed |
| Literature audited through 2026 | PASS | LITERATURE_AUDIT.md |
| Protocol deviations append-only record present | PASS | PROTOCOL_DEVIATIONS.md |

## Availability accounting

All eleven frozen public tasks were attempted. Nine are runnable. Amazon Reviews 2023 is retained as `NOT RUN — REQUIRED SOURCE SCHEMA UNAVAILABLE`; Open Payments is retained as the same status because the exact frozen/active public schema omits the prospectively mandatory amount field. MIMIC-III is retained as `NOT RUN — CONTROLLED ACCESS UNAVAILABLE`. No replacement source was introduced after outcomes.

## Audit conclusion

The audit passes only if every frozen runnable cell and control is present, unavailable branches are explicit, the prospective hashes remain unchanged, and all artifacts regenerate. Any failure above is terminal and the script exits nonzero unless `--allow-failures` is supplied solely for diagnostic reporting.
