# Numeric real-panel transfer result

Status: **external-transfer gate failed under dataset-level uncertainty**.

The synthetic-development fixed weights and competence temperatures were transferred
without retuning to three binary-classification and four regression datasets. The run
contains 840 fresh `(96 context, 256 query)` episodes and 20,160 expert fits.

| Task | Fixed loss | Competence loss | Dataset-balanced gain (95% hierarchical CI) | Soft gain over hard |
|---|---:|---:|---:|---:|
| Classification | 0.525667 | 0.522206 | 0.003461 [-0.001813, 0.011346] | 0.024674 [0.020399, 0.029715] |
| Regression | 0.710828 | 0.473830 | 0.236997 [-0.009046, 0.735772] | 0.054407 [0.041430, 0.077732] |

Both competence-versus-fixed point estimates are favorable, but neither hierarchical
interval excludes zero after resampling datasets. Strong and scoped transfer therefore
both fail their frozen gates. Repeated splits cannot repair the small number of
independent datasets.

Per-dataset gains reveal heterogeneity: churn (+0.01151), California (+0.00906), diamond
(+0.00312), and house (+0.95081) favor competence; Adult (-0.00135) and HIGGS (+0.00023)
are inconclusive; Black Friday is harmed (-0.01499 [-0.02218, -0.00847]). Soft weighting
does beat hard context-CV selection at the task-panel level, replicating the aggregation
mechanism outside PriorDial, but this does not establish improvement over the fixed
mixture.

The correct conclusion is heterogeneity plus inadequate dataset breadth, not a real-data
performance claim. A breadth follow-up may use the already frozen Day-8 OpenML panel
with unseen dataset identities and no method change; it must not select datasets from
these outcomes.
