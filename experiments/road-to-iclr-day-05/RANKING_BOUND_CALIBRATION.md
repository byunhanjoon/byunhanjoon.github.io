# Pairwise-ranking bound calibration

Status: post-confirmation theory audit, specified before computation.

Proposition 16 converts a candidate's cover residual into an upper bound on
the probability that a pair of quotient validation losses is inverted.  This
audit asks whether that finite bound is informative or merely formally true.

For every non-tied candidate pair in the binary-classification selection
panels, draw 512 strength-2 and IID-16 actions.  Record the empirical inversion
rate, the exact quotient-loss gap, the exact validation residuals, the raw
Proposition-16 upper bound, and the corresponding empirical second-moment
Markov bound.  Regression is excluded because Proposition 16's stated
constant uses the bounded diameter of probability vectors.

Report results separately for each panel and method, including the fraction of
pair bounds below one.  This is a bound-calibration diagnostic, not a new
confirmatory performance gate.  If the bound is vacuous, retain that result
and use the theorem only as qualitative grounding.
