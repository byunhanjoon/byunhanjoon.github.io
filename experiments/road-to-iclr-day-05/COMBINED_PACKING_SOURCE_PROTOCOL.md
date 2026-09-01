# Post-extension protocol: combined non-exhaustive source uncertainty

Frozen after the late-source primary gate and therefore labeled a post-gate
scope addendum.

## Population and clustering

Combine the prior 128-cell classification tensors with the four new late
OpenML sources.  Treat dataset/source identity—not panel repeat or candidate—as
the inferential unit.  Collapse confirmation, changed-menu, and
changed-subsample measurements of the same financial source before inference.
This yields seven unique sources: two financial, credit-g, and the four late
OpenML datasets.

## Comparisons and gate

For pair32, pack64, and unbiased pair-cross64, compute each source's percentage
reduction in mean candidate score RMSE versus its frozen equal-budget control.
Use 100,000 equal-source bootstrap resamples for the mean reduction and report
the exact two-sided sign-test p-value.

A comparison passes this scope addendum if at least 6/7 source reductions are
positive and the percentile bootstrap 95% interval has a strictly positive
lower endpoint.  This is conditional evidence over seven sources, not a claim
that repeated panels are independent or a substitute for a larger benchmark.

## Prespecified-rule extension

After freezing a second four-source block, append every completed source and
rerun the identical equal-source calculation. Scale the positive-source rule
from `6/7` to `ceil((6/7)S)` for `S` unique sources; the interval rule is
unchanged. This extension is post-gate and cannot erase either block's primary
strength-2 pass/fail result.
