# Frozen post-gate protocol: metric scope on late sources

Use all 12 candidates from the four late OpenML sources and the same 1,024
pair32/fourpack64 actions and controls.  For Brier score, clipped log loss,
ROC-AUC, and accuracy, estimate RMSE around the exact complete-product metric.

For each comparison report strict candidate wins and source-mean wins.  The
probabilistic/ranking scope passes if Brier and log win at least 10/12
candidates, AUC wins at least 9/12, and all four source means are no higher for
each of those three metrics. Accuracy has no gate because its discontinuous
margin boundary was already identified; it is a prospective descriptive
repeat. This protocol was frozen after seeing Brier packing RMSE but before
computing log, AUC, or accuracy outcomes on these sources.
