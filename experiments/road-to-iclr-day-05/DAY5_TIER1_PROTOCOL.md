# Day 5 Tier-1 schema-quotient action protocol

Status: frozen before any outcomes from this experiment were inspected.

Freeze time: 2026-08-27 (Asia/Seoul).

## Question

For complete tabular pipelines, can a product audit of exact schema
symmetries predict which nuisance factor should consume a fixed ensemble
budget on a dataset whose schema outcomes were not used to select the action?

The experiment is a conditional prospective extension: these datasets have
appeared in earlier performance studies, but their joint feature/category/
class schema orbits and the leave-one-dataset-out action policy have not been
inspected. It is not labeled an untouched-data benchmark.

## Datasets

- Australian Credit Approval (binary, mixed, small)
- Bank Marketing Subscription (binary, mixed, medium)
- German Credit Risk (binary, mixed, small)
- LendingClub Loan Default (binary, mixed, medium)
- FREM-TPL Claim Count (regression, mixed, medium)

Every dataset uses the existing checksum-audited train/validation/test split.
Training is capped at 20,000 rows and each query split at 3,000 rows using a
fixed stratified or uniform subsample. No split is selected by schema outcome.

## Exact schema factors

- feature positions: identity plus three frozen permutations;
- opaque category IDs: identity plus three independent within-field
  permutations (one level when no categorical field exists);
- target IDs: identity and the binary swap (one level for regression).

Actions are applied jointly to training, validation, and test tables.
Class probabilities are aligned back to semantic target IDs before scoring.
Category strings are treated as opaque: their meanings are not supplied to
the learner. All row order and split membership remain fixed.

## Pipelines and randomness

- standardized one-hot logistic/ridge negative control;
- ordinal-code random forest positive control;
- native HistGradientBoosting;
- native CatBoost;
- standardized one-hot Adam MLP.

Model seeds are `101, 202, 303, 404`. Hyperparameters are fixed in code and
not selected from schema or test outcomes. The full pipeline, including
preprocessing fit, is rerun for each representative.

## Endpoints

For Brier loss (classification) and squared error after train-target
standardization (regression):

1. persistent schema risk `Var_z(E_s p)`;
2. same-seed conditional schema risk `E_s Var_z(p|s)`;
3. exact product fANOVA main and interaction components;
4. hard-label flip rate for classification;
5. quotient-centroid proper loss and schema-risk fraction of mean member loss;
6. leave-one-dataset-out action performance at budgets 2 and 4.

The action policy is selected separately by model family and budget. For a
held-out dataset, average the validation expected residual-risk fraction for
each feasible factor-marginalization-plus-iid-complement action over the other
four datasets, choose the minimum with a deterministic tie break, and evaluate
that action on the held-out test predictions.

Comparators at the same number of fits are:

- iid schema sampling;
- the first `B` ordinary model seeds at every schema representative;
- dataset-specific validation oracle (diagnostic only).

The leave-one-dataset-out action gate is evaluated within the four binary
classification datasets, where all three factors share the same action
semantics. FREM-TPL is a measurement and regression-performance extension;
with no second regression dataset in this frozen tier it is not used to claim
cross-dataset action transfer.

The primary action endpoint is test residual schema risk. The secondary
endpoint is quotient proper loss. The action passes only if the transferred
policy improves residual schema risk over both iid schema sampling and the
ordinary seed ensemble on a majority of material held-out dataset--model
cases, without mean proper-loss degradation above `0.1%` of the reference
member loss.

## Integrity and interpretation

- One-hot logistic/ridge must close category and class relabeling to numerical
  tolerance; any material failure is an implementation audit failure.
- Dataset is the replication unit. Seeds, representatives, and rows are
  repeated measurements.
- Exact Brier/MSE ambiguity and fANOVA reconstruction errors must be below
  `1e-10` in float64 analysis.
- A finding about feature position is pipeline- and coupling-specific, not a
  claim that the represented dataset changed.
- The study is killed as an ICLR action result if attribution-selected actions
  do not transfer beyond generic iid schema or seed ensembles.
