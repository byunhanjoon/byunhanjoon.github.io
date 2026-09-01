# Day 5 deterministic canonical-orbit comparator

Status: frozen before canonicalized performance outcomes.

This experiment answers the strongest low-compute objection to schema
quotient ensembling: map every exact representation to one deterministic
representative before fitting.

The map uses training data only:

1. within each categorical field, sort levels lexicographically by their
   packed training-row membership vector and replace them by ranks;
2. sort fields by type, missingness mask, and the byte representation of the
   resulting training column;
3. sort target IDs by their packed training-row membership vectors.

Distinct levels necessarily have distinct membership vectors. Duplicate
fields may tie, but swapping identical canonical columns leaves the rendered
matrix unchanged. All Day-5 orbit representatives must therefore produce
byte-identical canonical train/validation/test matrices and targets. With a
fixed seed and deterministic learner, aligned outputs have exactly zero
declared schema risk.

The five Tier-1 datasets and four non-control families (ordinal forest,
HistGradientBoosting, native CatBoost, and Adam MLP) use the frozen Tier-1
subsamples, seeds, and hyperparameters. Canonical inputs are checked over the
entire 4 x 4 x 2 orbit, but only one canonical model per seed is fit because
all checked inputs are identical.

Primary performance comparison is the four-seed canonical ensemble versus
the four-seed identity-representation ensemble at equal fit count. Secondary
comparisons are one canonical seed versus one identity seed and the canonical
ensemble versus the much more expensive full schema-and-seed quotient. The
accuracy gate is descriptive: no universal improvement is assumed. Brier/MSE
changes above 0.1% of reference loss are material.

Scope limitations: the map depends on fixed training-row identity, does not
cover row permutations, and is not proposed for fields whose externally
meaningful order must be preserved. It is a necessary comparator, not the
main novelty claim.

