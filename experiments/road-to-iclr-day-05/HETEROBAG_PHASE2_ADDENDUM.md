# HeteroBag Phase-2 addendum

Status: frozen after the Phase-1 gate passed and before either additional
seed-triplet or any mechanism-control outcome was computed.

Freeze time: 2026-08-28T00:13:00+09:00 (Asia/Seoul).

The eight datasets, three architectures, splits, preprocessing, representations,
parameter-matching procedure, training budget, metrics, and averaging rule remain
unchanged from `HETEROBAG_PHASE1_PROTOCOL.md`.

## Seed robustness

Run two additional independent triplets:

- triplet 2: `20261234, 20261335, 20261436`;
- triplet 3: `20261537, 20261638, 20261739`.

Every triplet compares the same fixed HeteroBag candidate with `T+T+T`.
Aggregate once per dataset after averaging architecture and triplet strata.

## Frozen controls on triplet 3

Also train and evaluate:

- `Q+Q+Q` for classification or `Midrank+Midrank+Midrank` for regression;
- `T(A)+T(B)+transformed-T(C)`, where transformed-T reverses numerical field
  order but retains the identical T-PLE field charts. This is a coordinate-only
  representation placebo, not a semantic alternative.

All are fixed one-third averages with three fits. The alternate-rank width
matcher remains unchanged; any active-parameter mismatch is measured and
reported rather than repaired after seeing Phase-1 outcomes.

## Mechanism endpoint

On raw held-out predictions, compute prediction correlation, error correlation,
absolute disagreement, squared disagreement, and ensemble gain for a
same-representation pair, a cross-representation pair, and the transformed-T
placebo pair. Across the 24 dataset-architecture cells, compare Spearman
association with HeteroBag gain. This is descriptive mechanism evidence because
the eight datasets were used to establish the Phase-1 effect; it is not a new
untouched predictor test.

Phase 2 supports a surviving secondary direction if HeteroBag has positive
dataset-level mean gain in at least two of three triplets, positive pooled means
in both tasks, at least 65% pooled cell wins, and triplet-3 gain exceeds both the
homogeneous-alternate and coordinate-placebo controls in panel mean. Lack of
TabM and the conditional nature of the reused dataset panel continue to block a
standalone-primary promotion in this sprint.

