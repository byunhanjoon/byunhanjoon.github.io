# Expanded-model source audit

Status: post-outcome scope diagnostic, specified after the modern-model
extension. It does not replace the frozen 11-source analysis.

Append the native HistGB and CatBoost tensors to the original three candidates
on the eight late OpenML sources. Recompute each source's equal-candidate mean
score-RMSE reduction for pair32, pack64, and unbiased pair-cross64. Earlier
financial/credit-g source rows are unchanged. Report all eleven source effects,
the equal-source mean, a deterministic 100,000-resample percentile interval,
and the exact sign test.

This asks whether adding stronger model families reverses a source-level
conclusion. Because both the source and original outcomes were already known,
the result is sensitivity evidence only.
