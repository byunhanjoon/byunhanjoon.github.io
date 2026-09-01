# Repeated-split metric scope

Status: post-primary diagnostic specified after the repeated-split quadratic
transport result. No new gate is imposed.

On the three frozen alternate split tensors, repeat pair32 and pack64
calibration for Brier, clipped log loss, ROC-AUC, and accuracy. Aggregate at the
dataset×split unit, separately report nondegenerate CatBoost candidate wins,
losses, and mean RMSE ratios, and preserve exact/tied HistGB cases. This checks
whether the known nonlinear/ranking boundary changes under repartitioning; it
cannot retroactively broaden the primary quadratic claim.
