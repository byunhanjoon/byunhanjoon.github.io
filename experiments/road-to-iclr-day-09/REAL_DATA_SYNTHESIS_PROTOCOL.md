# Real-data synthesis protocol

Status: frozen on 2026-09-01 after the component panels were analyzed and before the
combined dataset-level estimates were computed. This is a retrospective synthesis,
not a new confirmation experiment.

## Question

Does the synthetic-development-tuned, three-fold loss-aligned mixture show a
dataset-balanced real-data effect when every completed numeric transfer panel is
pooled without outcome-based dataset exclusion?

## Included evidence

- `real_panel_competence_55553b7ffd`: 3 classification and 4 regression identities;
- `openml_breadth_competence_48170161d0`: 6 classification and 7 regression identities;
- `regression_confirmation_1e4911698d`: 5 regression identities.

The panels use disjoint dataset identities. Only the paired `competence` and `fixed`
rows are used. Each dataset receives equal weight, irrespective of repeat count or
query count. Gain is `fixed loss - competence loss`, so positive is favorable.

## Frozen summaries

For each task separately:

1. average paired gain within each dataset;
2. primary estimator: unweighted mean of dataset gains;
3. 95% percentile interval from 50,000 dataset bootstraps, seed 175001;
4. median dataset gain and number of positive datasets;
5. 10% symmetric trimmed mean, with `floor(0.1 * D)` datasets removed from each tail;
6. leave-one-dataset-out range of the unweighted mean;
7. panel-stratified means, reported descriptively.

The synthesis is considered robust only if the primary interval is above zero, the
median is positive, a strict majority of datasets are positive, the trimmed mean is
positive, and the minimum leave-one-dataset-out mean is positive. Classification and
regression are judged independently. No multiplicity-adjusted universal claim is made.

## Interpretation guardrails

- Passing strengthens a scoped numeric-panel transfer statement; it does not establish
  broad tabular superiority, categorical-feature coverage, or a novel routing method.
- Component-panel results were already known, so this cannot replace the independent
  confirmation panel.
- A failed classification synthesis remains a first-class negative result.
- Dataset-level resampling, not episode-level resampling, is the inferential unit.
