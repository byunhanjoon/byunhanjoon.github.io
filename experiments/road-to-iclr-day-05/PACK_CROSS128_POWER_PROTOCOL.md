# Frozen protocol: gap-calibrated ranking power at 128 fits

Restrict to the 23 non-exhaustive candidates on five 128-cell datasets. Reuse
the frozen 512 actions for pack-cross128 and complete eight-cover U128. For
every within-dataset candidate pair, subtract its exact quotient score to get
two score-error streams. Let `s` be the standard deviation of the control
pairwise error difference. At injected exact gaps `delta/s in
{0.25,0.5,1,2}`, estimate the symmetrized ranking-inversion probability by
averaging both candidate orientations.

Run the comparison twice: once with the original common action per candidate,
and once after a deterministic independent permutation of each candidate's
draw stream. The latter destroys candidatewise common-random-number coupling
while preserving every marginal error distribution.

The frozen gate passes if pack-cross128 has lower mean inversion on every
represented panel at all four gaps under both couplings (32/32 clauses) and on
at least 80% of pair-gap-coupling cells. This is a calibrated diagnostic, not
new held-out predictive evidence.
