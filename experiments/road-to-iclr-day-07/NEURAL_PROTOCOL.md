# Frozen development protocol — neural-base transfer certificate

Status: **FROZEN BEFORE NEURAL-BASE OUTCOMES**

Freeze date: 2026-08-30 (Asia/Seoul), after the scalar-trust replay failed.

## Scientific label

This is a `DEVELOPMENT BACKBONE-TRANSFER TEST`. All four data sources appeared
in prior MPE/Geometry Transfer experiments, but no outcome from the neural base
defined below exists at freeze time. It can test whether the diagnostic is tied
to CatBoost; it cannot provide fresh-source confirmation.

## Question

Does state-held-out residual transfer remain predictable when the ordinary
covariate base is a neural tabular MLP rather than CatBoost?

## Fixed matrix

- tasks: `acs_occupation`, `tlc_pickup_zone`,
  `airline_origin_airport`, `medical_charges`;
- state splits: 0 and 1;
- operators: the existing nine-member Geometry Transfer operator family;
- base: two-hidden-layer width-128 ReLU MLP, AdamW, 40 fixed epochs;
- design: standardized numeric and one-hot categorical ordinary covariates;
- the semantic field and every derivative of it are excluded from the base;
- row cap: state-preserving 50,000-row subsample per fit;
- residuals: three-fold row-OOF predictions on outer observed states;
- test base: one fit on all outer observed states, with the same fixed budget;
- device shards are operational only and do not change a cell.

## Prediction and selection

Within the observed states, deterministic five-fold state holdout rebuilds
each geometry operator and scores full-strength residual transfer. For every
outer cell and operator, store mean inner gain and its fold standard error.

Two deployable selectors are fixed:

- `mean`: choose the operator with maximum inner mean gain and use it iff that
  mean is positive;
- `pessimistic`: choose the same operator but use it only if
  `mean - 1.96 * SE > 0`.

The outer test states are read only after these decisions are formed.

## Gates

The neural-base diagnostic passes development if all hold:

1. Spearman correlation between inner predicted and outer actual gains is at
   least 0.60 over source-by-operator aggregates;
2. direct gain-sign accuracy is at least 75% over those aggregates;
3. the pessimistic selector has positive source-balanced outer gain;
4. no source has pessimistic mean outer gain below `-0.002` standardized MSE;
5. the pessimistic selector has no more harmful task-split cells than the mean
   selector;
6. fixed-base exclusion, row OOF completeness, and output finiteness audits
   pass.

Passing would support the *diagnostic/formulation*, not an MLP performance
claim. Failure would demote the Geometry Transfer Law to a CatBoost-specific
or insufficiently nested observation until a stronger design resolves it.

