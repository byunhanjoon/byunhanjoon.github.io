# Frozen addendum: non-exhaustive source uncertainty for pack-cross128

Exclude every exact product and retain only the 23 candidates on 128-cell
products. Within each represented panel, average the paired candidate RMSE
difference (`pack-cross128 - independent-cover-U128`) by dataset and form the
same deterministic source-cluster bootstrap 95% interval used elsewhere.

The strict addendum passes if every represented panel has a negative mean,
every source is favorable, and every interval's upper endpoint is below zero.
Panels with no 128-cell products are reported as out of scope, not successes.
