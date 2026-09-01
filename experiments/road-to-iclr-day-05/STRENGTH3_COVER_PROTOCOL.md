# Budget-64 strength-3 nuisance-cover hierarchy

Status: frozen before strength-3 outcomes.

This analysis reuses complete prediction tensors and performs no new fitting.
It tests a pre-specified strength hierarchy, not a new selected model.

Construct a 64-row mixed OA from all `(u,v,w) in GF(4)^3`:

- feature = `u`;
- category = `v` (or the singleton for regression/no-category factors);
- seed = `w`;
- binary class = `Tr(u + alpha*v + alpha^2*w)` (or its singleton).

Every one-, two-, and three-factor margin is balanced. Randomize independent
factor level names exhaustively, which preserves marginal uniformity and makes
the prediction average unbiased for the full quotient.

At 64 fits compare its exact expected residual with:

- iid-64 joint draws;
- four independent strength-2 OA-16 blocks;
- sixteen independent strength-1 four-run blocks;
- sixteen four-seed blocks.

Compute residual both from the randomized-design covariance Gram form and from
the exact fANOVA coefficients; their discrepancy must be numerical. Screen
materiality on validation, then report paired test results and source-group
means. Run on the frozen confirmation panel, the disjoint nuisance-menu repeat,
and the changed-subsample repeat once available.

