# Cross-score selection-regret bound audit

Status: post-theory diagnostic, aggregation specified before computation.

For each dataset and for cover-cross32 and IID-U32, estimate the
Proposition-21 bound as twice the sum of candidate score standard deviations.
Compare it with observed mean quotient validation regret. Report validity,
tightness, paired source counts, and source-bootstrap intervals for the cover
minus IID bound.

The diagnostic is favorable if the cover has a smaller mean bound in all five
panels. The bound is expected to be conservative; a large bound/regret ratio is
retained and limits how the theorem is presented.
