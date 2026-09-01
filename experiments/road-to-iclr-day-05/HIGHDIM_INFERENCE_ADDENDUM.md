# High-dimensional repetition-uncertainty addendum

Status: frozen after the nine field-only point estimates were opened and
before uncertainty intervals were computed.

The high-dimensional experiments have only eight independent randomized
design repetitions per method. To expose estimator uncertainty, use a
nonparametric bootstrap over design repetitions, independently within each
cell and method, with 50,000 draws and seed `2026082811`.

For squared prediction risk, compute each bootstrap sample variance exactly
from the repetition prediction Gram matrix. For predictive loss, bootstrap
the mean per-repetition Brier score. Report per-cell 95% percentile intervals
for `OA risk / control risk` and the bootstrap probability that OA is lower.
Also report a fixed-panel pooled ratio by averaging cell risks within each
draw. This interval is conditional on the selected datasets/models and is not
a dataset-population confidence interval.

Apply the same procedure to field-only OA versus iid and, once complete, to
field-plus-row OA versus both marginally balanced and iid controls.

