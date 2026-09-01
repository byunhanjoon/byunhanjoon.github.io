# Implementation repair log

## 2026-09-01 — Natural hat-basis duplicate knots

- Stage: development natural equivalent bases; prospective outcomes remained locked and unaccessed.
- Trigger: all three `pendigits` bundles for each already-running foundation-model process (TabICLv2 and TabPFN-2.6; six attempts total) raised `no valid natural pair` because the frozen highest-variance feature had fewer than eight distinct training quantiles. No destination bundle was written for a failed job.
- Repair: for C3 only, traverse continuous features in the same deterministic highest-training-variance-then-lexicographic order and choose the first feature with eight distinct training quantile knots. Rejected features are recorded in representation metadata. Other datasets retain their original selected feature.
- Scientific settings unchanged: basis dimension 8, training-only quantiles, DCT transform, model settings, seeds, dataset panel, and development protocol hashes.
- Recovery: rerun only absent natural-basis bundles; immutable completed bundles are not overwritten.
