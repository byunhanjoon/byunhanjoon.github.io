# Frozen independent regression confirmation panel

Frozen: 2026-09-01, before model outcomes on these dataset identities.

## Deterministic panel selection

Enumerate cached OpenML task metadata with IDs beginning `361`, sort by task ID, retain
regression tasks with at least 500 rows, and exclude every dataset identity already used
in Day-09 competence experiments. Take the first five remaining identities:

1. Abalone, task 361234;
2. Auction Verification, task 361236;
3. Geographical Origin of Music, task 361243;
4. Solar Flare, task 361244;
5. Naval Propulsion Plant, task 361247.

This rule and resulting list are frozen before outcomes. Missing data may be downloaded
only for these named, small datasets; a failure is recorded rather than replaced.

## Evaluation

Use official repeat-0/fold-0 splits, outer-train median/z-score preprocessing, numeric
columns only, and the first 32 numeric columns by source order. Run 60 fresh episodes per
dataset at seed 145001, with 96 context and 96 query rows.

### Structural v2 addendum

The first execution aborted without artifacts after two buffered dataset summaries were
exposed: the Geographical Origin official test fold has only 106 rows, below the original
128-query request. That partial run is invalid. A label-free split audit found test sizes
418/205/106/107/1194. Version 2 therefore uses 96 queries uniformly, the largest common
round size below the structural minimum. Identities, seed family, repeats, methods,
feature cap, and inference are unchanged.

The method is the original synthetic-development regression competence router
(`temperature=0.1`, no shrinkage), not the later dataset-cross-fitted diagnostic.
Comparators are the synthetic-development fixed mixture, uniform, hard CV selection,
and best-individual oracle. There is no panel tuning.

The primary estimate is dataset-balanced fixed-minus-competence MSE. Use 10,000 paired
hierarchical bootstrap draws over datasets then episodes. Confirmation requires a
strictly positive 95% interval. Report all datasets and retain failures. A pass confirms
only small-context numeric regression transfer for this expert panel; it does not claim
state-of-the-art accuracy or novelty of exponential weighting.
