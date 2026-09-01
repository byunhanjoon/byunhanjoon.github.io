# Frozen real-panel transfer protocol: context-only competence routing

Frozen: 2026-09-01, before any competence-routing outcome on these real datasets.

## Question

Do the synthetic-development temperature and fixed-mixture weights transfer without
retuning to real tabular data, or is the positive result confined to PriorDial?

## Frozen panel and scope

Use the seven cached binary/regression datasets from the Day-8 panel:

- binary classification: `adult`, `churn`, `higgs-small`;
- regression: `california`, `diamond`, `house`, `black-friday`.

`otto` is excluded before outcomes because the frozen competence expert API is binary,
not multiclass. Only the Day-8 loader's training-standardized numeric columns are used;
the six experts were defined for continuous coordinates, and one-hot categorical
geometry is out of scope. This makes the test a numeric real-tabular transfer check, not
a full mixed-type benchmark.

The Day-8 split seed is 20260831. For each dataset, draw 120 fresh paired episodes using
seed family 125001: 96 context rows from its frozen train split and 256 query rows from
its frozen test split. Binary samples are stratified. Regression targets retain the
Day-8 train-standardized scale.

## Frozen methods

The six experts and three-fold context CV are unchanged. No real query label or dataset
identity tunes a parameter.

- uniform six-expert mixture;
- task-specific fixed weights learned on synthetic development;
- task-specific soft competence weights with the synthetic-development temperature and
  shrinkage;
- hard context-CV argmin, diagnostic;
- per-episode best individual expert, query-label oracle diagnostic.

## Analysis and gate

Primary contrast is fixed loss minus soft-competence loss, positive when competence is
better. Report each dataset and a task-type aggregate. Aggregate confidence intervals
use 10,000 hierarchical paired bootstrap draws: resample datasets with replacement,
then resample paired episodes within every selected dataset. Per-dataset intervals use
paired episode bootstraps.

- Strong transfer: both task-type aggregate 95% intervals are positive.
- Scoped transfer: at least one task interval is positive and the other excludes
  material harm, defined as worse than 0.005 log loss or 0.02 standardized MSE.
- Otherwise kill the external-transfer claim.

Dataset count, not 120 overlapping splits, is the unit supporting breadth. Even a pass
is a small numeric-only panel result, not state-of-the-art accuracy or full external
validity. No method refinement is allowed on these outcomes; any refinement requires a
new protocol and fresh resampling seeds.
