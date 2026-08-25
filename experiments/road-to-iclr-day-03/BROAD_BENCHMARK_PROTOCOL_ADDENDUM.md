# Day 3 broad-benchmark protocol addendum

This addendum was written while the frozen ten-dataset confirmation tier was
running and before either addendum experiment produced a model outcome. It does
not modify the original 25-dataset protocol or its selection and claim gates.

## A. Five-dataset prospective breadth extension

The original broad screen contains 25 distinct datasets. To meet the requested
30-dataset breadth target without rewriting that completed freeze, a separate
five-dataset extension was selected using only local availability, schema, task,
and compute constraints:

- Covtype and Jannis (multiclass numerical tables);
- Gesture (smaller multiclass numerical table);
- Santander Customer Transactions (binary numerical table);
- Facebook Comments Volume (mixed numerical/categorical regression table).

Learning-to-rank tasks, copies of datasets already in the original tier,
feature width above 500, and output cardinality above 100 were excluded. The
extension uses the same training-only representation construction, condition
numbers `{1,1000}`, geometric-energy control, frozen AdamW calibration, three
seeds, and four model families. It is a replication of the primary phenomenon,
not another remedy-selection tier.

The exact configuration is
`experiments/day3/configs/broad_extension_preregistered.json`. Code,
hyperparameters, source arrays, and metadata were frozen before outcomes in
`results/day3/broad_benchmark/broad_extension_freeze.json` (42 protected files,
aggregate SHA-256
`58ed19b7c12e1d24524f6f9e5e1acd160e7f65715da5310a23aff7f916fd60af`).

The honest statistical labels are therefore:

- **original prospective screen:** 25 datasets;
- **prospective replication extension:** five additional datasets;
- **combined descriptive result:** 30 distinct datasets.

The five were appended after original Phase 1 outcomes existed, so the paper
must not imply that all 30 were frozen in one simultaneous preregistration.

## B. Same-table distribution-shift comparison

The original finance tasks use official chronological purged splits. Merely
comparing them with unrelated non-temporal datasets would confound dataset and
split type, so the robustness tier adds a fixed row-random 70/15/15 resplit of
the exact same three source tables:

- Compustat Korea direction;
- KDD17 stock direction;
- KDD17 stock return.

The random permutation seed is `20260825`; post-split row caps and subsampling
match the broad protocol. The comparison uses MLP, AdamW, three seeds, and the
two controlled basis endpoints. Its purpose is to ask whether basis sensitivity
survives a change from easier row-random evaluation to chronological deployment
shift.

This row-random split may mix entities and adjacent dates. It is intentionally
an easier comparator, not a recommended finance evaluation protocol. The exact
configuration is
`experiments/day3/configs/distribution_shift_preregistered.json`. Its 34
protected files were frozen before random-split outcomes with aggregate SHA-256
`103efe48919b99aef195ae229e573af52d6cf3d202b1b99525ac90a44d78fc95`.

## C. Analysis-only corrections

Post-freeze analysis corrections are enumerated in
`results/day3/broad_benchmark/analysis_fix_addendum.json`. They exclude epoch
curve tables from run-table globs, enforce complete coverage and failure-aware
gates, normalize cross-task regression differences, and export preregistered
clustered, robustness, runtime, and memory summaries. They do not alter data,
models, optimization, hyperparameters, metrics, gates, or method selection.
