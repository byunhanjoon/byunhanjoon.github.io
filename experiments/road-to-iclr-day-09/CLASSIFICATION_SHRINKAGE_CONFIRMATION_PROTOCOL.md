# Independent classification-shrinkage confirmation

Status: frozen on 2026-09-01 before any episode outcome on the selected tasks.

## Motivation and status

The retrospective 9-dataset path suggested that 10% movement from fixed to competence
can avoid the full router's classification tail failure. Because lambda=0.1 was selected
after real outcomes, this is a real-development-tuned candidate, not the original
synthetic-only rule. Only the independent panel below can confirm it.

## Deterministic panel

From OpenML-CC18 (suite 99), scan task IDs in ascending order and select the first five
binary tasks with at least two numeric features and official repeat-0/fold-0 train/test
sizes supporting 96 context and 64 query rows, after excluding any Day-09 identity.
Multiclass/no-numeric tasks and the already-used Diabetes identity are structural
exclusions. The resulting fixed tasks are:

- breast-w, task 15;
- credit-approval, task 29;
- credit-g, task 31;
- spambase, task 43;
- electricity, task 219.

Use official repeat-0/fold-0 splits, numeric-only preprocessing fit on outer training,
first 32 source-ordered numeric features, 80 fresh stratified episodes per dataset,
`n=96`, `q=64`, three CV folds, and seed 225001. No task may be replaced.

## Frozen candidate and inference

Let `p_0` be the development-tuned fixed prediction and `p_1` the frozen competence
prediction. The sole candidate is `p_0.1 = 0.9 p_0 + 0.1 p_1`. Primary gain is fixed
minus `p_0.1` binary log loss, aggregated with an equal-dataset hierarchical bootstrap
over datasets and episodes (20,000 draws, seed 225501).

Confirmation requires the 95% interval above zero and at least three of five dataset
point estimates positive. Full competence, Brier, AUC, and high-NLL rate are secondary
diagnostics. No hyperparameter may change after the run.

Passing supports a small, real-development-tuned shrinkage improvement on this numeric
binary scope. It does not make the shrinkage rule novel or restore the original
synthetic-only classification-transfer claim.

## Post-result dependence sensitivity

Because the smallest official test folds have 69–70 rows and the query size is 64,
repeated episodes strongly overlap in those datasets. After the primary result, freeze a
100,000-draw bootstrap of the five dataset mean gains only (seed 226001), plus all five
leave-one-dataset-out means. This removes any reliance on episode-level independence and
is reported as sensitivity evidence, not part of the original gate.
