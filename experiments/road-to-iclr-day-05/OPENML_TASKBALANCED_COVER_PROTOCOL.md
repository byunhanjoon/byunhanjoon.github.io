# Task-balanced OpenML cover replication

Status: frozen before task-balanced cover outcomes.

Post-gate enumeration audit: report results separately for nuisance products
larger than budget 16 and products of exactly 16 cells. The latter are complete
enumeration, not evidence of variance reduction beyond exhaustive averaging.

## Evidence status

These eight OpenML sources were previously used by the separate HeteroBag
development line, whose semantic-specificity claim failed. They were not used
to construct or tune the nuisance cover, its strengths, or this panel's
thresholds. This is therefore an outcome-independent cross-task replication
for OrbitCover, but not a wholly untouched dataset panel.

## Panel

Four binary classification and four regression sources, three fixed learning
algorithms (linear, random forest, and Adam MLP), four feature-order views, up
to four independently generated category-ID views, task-appropriate target-ID
views, and four seeds. Store the exact full product for validation and test.

Apply the existing validation-material threshold and exact covariance
analyses without modification. The strength-2 gate requires lower held-out
pooled residual than IID-16, four strength-1 blocks, and four seed blocks;
at least 75% of validation-material cells beating all; at least 6/8 source
means beating all when all sources are represented; and positive pooled
reduction separately for classification and regression. Strength-3 and
anytime results are supporting hierarchy checks. No model-selection claim is
predeclared because the genuinely untouched OpenML panel already falsified
external validation-to-test selection transfer.
