# Day 5 TabPFN v2.5 exact-schema audit

Status: frozen before outcomes.

This supporting experiment tests whether TabPFN v2.5's default internal
preprocessing/permutation ensemble removes the same exact Tier-1 product risk
seen by a single estimator. It is not the primary novelty claim.

- datasets: Australian Credit, Bank Marketing, German Credit, LendingClub;
- train/validation/test caps: 1,000 rows each, fixed stratified subsamples;
- factors: four feature orders, four opaque category-ID maps, both class IDs;
- configurations: one estimator with internal feature/class shifts disabled;
  one estimator with defaults; eight estimators with shifts disabled; eight
  estimators with defaults;
- checkpoint: local TabPFN v2.5 classifier checkpoint;
- seed: 4201; all outputs aligned to semantic class IDs.

Endpoints are Brier loss, total and factor-attributed schema risk, hard-label
flips, and the residual-risk fraction of default eight-member ensembling. The
expected result is reduction, not necessarily exact closure. The experiment
must not claim the discovery of TabPFN permutation sensitivity or a new
equivariance gap.

