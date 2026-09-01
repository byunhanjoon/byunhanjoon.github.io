# Frozen protocol — Context-Only Loss-Aligned Routing

Freeze time: 2026-09-01T03:52:23+09:00  
Status: frozen before implementation outcomes  
Scope: new fallback diagnostic; not E4/M6 and not a revision of the killed Day-09 method

## Question

The prior fallback showed that generator-family identification can approach 100% while
predictive routing worsens. Test the replacement hypothesis:

> Route experts using their inner-context predictive competence, not their nominal
> generator-family match.

This is a performance and mechanism test. Cross-validated stacking/model selection is
classical and is not claimed as a standalone novelty. The potentially new scientific
result is the controlled separation between identifiable prior metadata and usable
predictive information under the fixed-marginal dial.

## Frozen experts and information boundary

Reuse all six E1 family-labelled experts without modification: linear, additive spline,
random forest, quadratic interaction, shallow partition tree, and periodic features.
Each expert is fit on the full context for query prediction.

The loss-aligned router obtains one competence number per expert from three-fold
cross-validation within the labeled context. It may use context features and labels, but
never query rows, query labels, rho, mechanism identity, warp identity, or the coupling
bit. Classification competence is cross-entropy; regression competence is MSE.

Convert competence losses `L_j` to weights

`w_j proportional to exp(-(L_j-min_k L_k)/temperature)`

and shrink toward uniform by a global coefficient selected on development episodes.
Temperature and shrinkage are selected separately by task type, then frozen.

## Data and splits

- unchanged generator: `prior_dial_v1_1`;
- task types: classification and regression;
- rhos: `{0,.25,.5,.75,1}`;
- regimes: `(n,d)` in `{(64,8),(96,8),(64,12),(96,12)}`;
- query size: 128;
- development: 120 tasks per rho/task/regime, seed family 95001;
- untouched test: 240 tasks per rho/task/regime, seed family 105001;
- all task counts are balanced across six mechanisms and warp marginals;
- test outcomes cannot change experts, grids, estimands, or exclusions.

The primary unit is an independently generated episode. No task, mechanism, rho, or
regime may be filtered after outcomes.

## Comparators

1. uniform six-expert mixture;
2. development-optimized fixed global simplex mixture (same full-context expert fits);
3. label-free marginal-shape mechanism router trained on development episodes;
4. matched-family one-hot route (generator-information diagnostic only);
5. context-CV loss-aligned route;
6. query-label best individual expert (headroom diagnostic only).

The fixed mixture and router hyperparameters use development query labels. Test query
labels are used only for final metrics and the diagnostic oracle.

## Primary estimands and gates

- Primary: test loss of loss-aligned routing minus the development-tuned fixed mixture,
  aggregated equally over declared rho/regime cells within each task type.
- Secondary: the same contrast in the two `d=12,rho>=.75` classification cells, where
  family routing weakened; family-route and uniform contrasts; win rates; and routing
  headroom capture relative to the best-individual query oracle.
- Report paired episode-bootstrap 95% intervals with 10,000 draws, after equal cell
  weighting.

Performance opportunity passes only if:

1. loss-aligned routing improves over the fixed mixture with an interval excluding zero
   in at least one task type;
2. it causes no statistically clear average harm larger than 0.001 log loss or 0.01 MSE
   in the other task type;
3. it captures at least 20% of fixed-to-best-individual headroom in the passing task; and
4. any high-dimensional classification claim is made only if its predeclared slice also
   improves with an interval excluding zero.

If these gates fail, stop. Do not add router features, tune per regime, or replace
experts. Regardless of outcome, retain the result as a test of objective alignment.

## Compute interpretation

The route requires three extra context-only fits per expert. Report wall-clock and fit
count. It is not compute-matched to a fixed mixture, so any gain is an opportunity signal,
not a deployable Pareto claim.
