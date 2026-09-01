# OrbitCover incumbent audit for the Day-6 ranking

Audit time: 2026-08-29 ~01:10 Asia/Seoul  
Status: **EVIDENCE EXTRACTION ONLY — NO PREMATURE SCORE OR RANK**

The authoritative source is Day 5 `results.md`, which supersedes the earlier
long-form `DAY5_FINAL_REPORT.md` counts and verdict.  This note prevents the
Day-6 ranking from comparing against a remembered or selectively favorable
version of OrbitCover.

## Evidence strength

- final verdict: **PARTIALLY SUPPORTED**;
- 144 neural dataset×split×architecture cells and 12 source units in the final
  independent-seed showdown;
- MLP, ResNet, FT-Transformer, and TabM in the primary neural closure, with
  separate TabPFN and tree-model evidence;
- 140,592 audited unique fit keys, 232.005 summed fit-hours, and 116/116 final
  audit tests;
- frozen mandatory experiments complete, including realistic-size,
  convergence, matched-function, coupling-ablation, and failure panels.

This breadth is substantially beyond Day 6's three-dataset matrix.  OrbitCover
is not subject to the Day-6 three-dataset empirical cap.

## Strongest positive result

At budget 16, coupled OC2 versus canonical independent wins 144/144 cells and
12/12 sources, with equal-source mean residual reduction 55.9% and clustered
95% interval `[38.7%, 73.8%]`.  The best finite coupling ablation is
`all_factors`.  The final report's defensible thesis is efficient estimation
of a distinct semantic quotient predictor through interaction-balanced
coupling—not generic schema robustness or predictive SOTA.

## Decisive negative boundaries

- OC2 with independent nuisance draws wins only 5/144 cells and is 7.0% worse
  on average than canonical independent.  Coupling, not schema-only balancing,
  carries the finite-budget result.
- Canonical-independent and schema×independent expectations differ by mean
  squared distance `2.632e-4`; OrbitCover partly changes the target rather than
  merely estimating the canonical predictor more accurately.
- At convergence, mean OC2/SRS residual ratio is `1.002`; the finite-budget
  relative advantage does not persist on average.
- Strength-3 recovers none of the recorded strength-2/SRS losses.  Interaction
  order is a boundary condition, not a reliable cellwise oracle.
- Exact matched-function residual is negligible for MLP and ResNet; the
  mechanism is architecture-specific rather than universal.
- Validation winner agreement improves from 96.69% to 99.41%, but held-out
  selected-test regret moves only from `0.005029` to `0.004906`, with exact
  validation/test winners agreeing in 19/36 partitions.

These failures limit practical utility and must appear beside the 144/144
finite coupled wins.

## Novelty subtraction

Orthogonal arrays, randomized-OA integration, fANOVA, group/frame averaging,
antithetic sampling, U-statistics, jackknife bias correction, and low-variance
model-selection criteria are established.  The closest 2026 collision is
antithetic Gaussian cross-validation, followed by a newer optimality result
for equicorrelated Gaussian CV; OrbitCover cannot claim generic antithetic-risk
estimation or minimax optimality.

Its remaining distinct composition is:

1. a finite product of exact complete-pipeline semantic nuisances and learner
   randomness;
2. aligned prediction-space quotient/fANOVA accounting;
3. interaction-matched dependent retraining actions; and
4. independent outer cross-scores with an end-to-end covariance-to-selection
   audit.

Whether this composition earns novelty above the close-work cap is decided
only under the frozen final rubric after Day-6 evidence is complete.

## Ranking implication held in reserve

OrbitCover enters the final comparison as the incumbent with exceptional
evidence breadth, strong theory/execution, and a credible ICLR thesis.  Its
weak points are classical-component collisions, dependence on coupling,
finite-budget rather than converged gains, target shift, and weak held-out
predictive/selection utility.  No Day-6 direction displaces it merely by being
newer or having a cleaner plot.
