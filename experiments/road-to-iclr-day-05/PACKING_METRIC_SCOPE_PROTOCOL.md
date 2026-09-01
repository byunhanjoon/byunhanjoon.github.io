# Frozen protocol: disjoint-packing metric scope

Use every binary-classification candidate in the five frozen selection panels
and the already frozen 1,024 pair/pack actions. Compare disjoint pair32 with
two independent covers, and mutually-disjoint pack64 with two independent
disjoint pairs. For each action prediction, estimate the complete quotient's
Brier loss, clipped log loss (`1e-12`), ROC AUC, and accuracy. Report RMSE from
the exact quotient metric by panel and candidate, separating 128-cell products
from exact closures.

The probabilistic/ranking scope gate passes if Brier and log-loss panel-mean
RMSE are no higher on all represented panels, AUC is no higher on at least four
panels, and at least 70% of non-exhaustive candidate comparisons are strict
wins for each of Brier, log loss, and AUC at both budgets. Accuracy has no pass
clause and is retained as a falsification metric. A universal-metric label is
allowed only if accuracy also meets the same 70% non-exhaustive threshold.
