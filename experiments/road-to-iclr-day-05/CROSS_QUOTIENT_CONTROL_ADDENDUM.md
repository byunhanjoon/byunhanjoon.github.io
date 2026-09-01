# Cross-quotient equal-budget control addendum

Status: **frozen before these control outcomes**; added after the primary
strength-2-vs-IID-U gate passed.

At the same 32-fit budget, form independent-half cross-scores from:

- two SRSWOR-16 samples;
- two four-block strength-1 actions;
- two four-block seed-only actions;
- two scrambled Sobol-16 actions;
- two scrambled Latin-hypercube-16 actions.

Use the identical five panels, 1,024 validation-only decisions, and held-out
test evaluation as the primary cross-score protocol.  The post-gate extension
passes if strength-2 cross-score has lower mean validation quotient regret than
each named control on at least four of five panels.  Dataset counts and test
losses are reported descriptively; this addendum cannot retroactively enlarge
the primary gate.
