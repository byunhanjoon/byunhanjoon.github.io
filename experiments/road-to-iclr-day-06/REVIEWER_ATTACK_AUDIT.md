# Day-6 reviewer attack audit

Status: living audit; written before the active H3–H9 evidence completed.

## 1. “This is the JMLR 2024 numerical-sensitivity result on tables.”

This is the most serious novelty objection.  Bit-level differences between
mathematically equivalent optimizer updates, fixed random tapes, macroscopic
path divergence, and stability boundaries are established.  Day 6 can claim
only the exact supervised tabular schema conjugacy, causal localization to the
schema-facing affine reduction, architecture/horizon boundary, and any
prospectively validated semantic-shadow consequence.  H1 alone is likely a
strong case study, not a safe standalone ICLR novelty claim.

## 2. “Just canonicalize the schema.”

For feature order and category IDs with reliable semantic metadata, this is a
valid and usually cheaper intervention.  Sorting fields and canonicalizing
category vocabularies before rendering restores a fixed contraction order and
does not require float64.  IEA64 is useful when pipeline components expose
different but semantically aligned dense layouts, canonicalization metadata is
not carried to the kernel, or one wants a local arithmetic commutation device.
Day 6 must benchmark or at least acknowledge canonicalization; it cannot claim
IEA64 is the universally preferred engineering fix.

This objection sharply caps standalone practical utility.  A successful
H4/H5/H6 diagnostic could remain useful because a schema shadow deliberately
uses noncanonical layouts to probe optimizer fragility, even if deployment is
canonicalized.

## 3. “Float64 GEMM is hardware-specific and expensive.”

Correct.  Exact closure is conditional on tested PyTorch/CUDA kernels and the
float32 rounding-cell condition, not hardware-independent.  H3's ≤25% gate is
an engineering check with fixed arm order, not a randomized benchmark.  A
credible submission needs at least a second hardware/software stack or must
frame IEA64 as a causal instrument rather than a production primitive.

## 4. “You did not prove the full optimizer path is bitwise conjugate.”

Long artifacts store aligned predictions at nine checkpoints, not every
parameter and optimizer state.  The direct one-update CPU audit verifies exact
parameter and AdamW-moment conjugacy for all three models.  Reports must say
“checkpoint-prediction closure plus one-update state closure,” not imply every
intermediate GPU state was recorded.

## 5. “There is no accuracy improvement.”

H1 agrees.  IEA64 selects a different chaotic canonical path with signed test
loss changes.  Reproducibility is a legitimate property but is weaker than a
consistent predictive or decision benefit.  H3's equal-dataset loss gate tests
systematic harm only.  H5 is the strongest route to consequential utility if
an early shadow forecasts ordinary seed multiplicity.

## 6. “Three datasets and three small dense-stem models are not ICLR breadth.”

The matrix satisfies the Day-6 validation requirement but caps empirical
strength under the frozen ranking rubric.  It omits native categorical
tokenizers, TabM in the current matrix, TabPFN, multiclass tasks, missingness
stress, large industrial tables, and independent codebases.  A passed Day-6
idea still requires external datasets/architectures before submission.

## 7. “Your stable MLP/ResNet control was cherry-picked.”

Partial H3 already falsifies a universal version: Bank ResNet is near numerical
scale at epoch 20 and material by epoch 50.  This counterexample is retained.
The honest boundary is horizon- and optimization-dependent amplification, not
an architecture taxonomy saying transformers are unstable and residual MLPs
are stable.

## 8. “H6 calls an extrapolated three-point slope a Lyapunov exponent.”

H6 explicitly does not estimate a tangent-space spectrum or global exponent.
The name is mnemonic; the estimand is a finite-difference log prediction-orbit
growth screen.  Its numeric forecasts are uncalibrated even on development
data.  It survives only if it prospectively improves AUROC over the raw
epoch-20 level by the frozen margin.

## 9. “H4/H5 are ordinary early learning-curve prediction.”

Early-dynamics forecasting and multiplicity diagnostics are established.  The
only distinction is the controlled perturbation: exact parameter-conjugated
schema twins start as the same real-valued function and isolate interface
arithmetic.  If that perturbation does not transfer to seed fragility, H5 is
discarded.  If it transfers, new-seed/new-optimizer confirmation is still
required.

## 10. “The 200-epoch constant-learning-rate paths are unrealistic.”

They are an intentional stress horizon for pathwise commutation, not a claim
that every tabular practitioner should train this way.  Canonical losses and
delayed divergence must both be shown.  A future confirmation should include
early stopping or a common modern schedule frozen independently of Day 6.

## 11. “H8 was invented after H6 made a mistake.”

Correct: H8 is a declared successor, not a rescue of H6.  H6's original 33
test bundles, thresholds, and verdict remain unchanged.  H8 names and excludes
all seven artifacts available when its rule was created, then requires 29
future bundles plus a 0.10 accuracy improvement over H6.  Its log-convex modal
mixture is an interpretable local approximation, not a novel general theorem.
Even a pass is prospective evidence for a diagnostic refinement, not evidence
that the seven development bundles independently confirmed it.

The H8 variable called acceleration is explicitly a difference of two interval
slopes, not a normalized second derivative.  Proposition 10 motivates a
positive slope change under modal takeover; it does not derive the frozen
`0.02` cutoff.

## 12. “H9 was invented when H7's delay result weakened.”

Correct.  H9 is an explicitly selected successor, not independent confirmation
of a pre-Day-6 theory.  H7 keeps its original 31-bundle verdict.  H9 excludes
all eleven bundles available when the distinction between hitting time and
final damage was made, tests only the remaining 25, and cannot become a new
ranking candidate.  Proposition 11 requires shared linear response maps and
covariance ordering that need not hold after nonlinear paths separate.  Even a
pass supports only a narrow post-breach attenuation clause inside Semantic
Arithmetic and remains subject to canonicalization and no-accuracy-gain caps.
The split is also ordered by bundle runtime, not randomized, so its dataset/
model composition is unbalanced; any positive result needs a newly randomized
confirmation rather than a larger claim from these 25 bundles.

## 13. “Feature-permutation metamorphic tests and conjugate training dynamics already exist.”

Correct.  Supervised metamorphic-testing work has used attribute and class
permutations since at least 2011, and NeurIPS 2024 explicitly studies neural
training-dynamics equivalence through topological conjugacy.  The residual
claim is not either ingredient: Day 6 supplies an explicit parameter action
that makes a schema-permuted tabular predictor the same real-valued function,
then isolates and intervenes on finite-precision failure of that pathwise
commutation.  Any H4/H5 forecasting result would be a consequence of this
controlled numerical orbit, not a claim to have invented metamorphic testing
or conjugacy.  This collision keeps novelty moderate even when empirical gates
pass.

## 14. “IEA64 is ordinary mixed-precision accumulation.”

Wider accumulators during training and conditioning-guided selective
higher-precision accumulation are established.  IEA64 earns no novelty credit
for computing an inner product in a wider format.  The only distinct object is
how the interface is selected and evaluated: a known tabular schema group
action determines the affine boundary, conjugate parameters make the two
real-valued functions exactly identical, and the intervention tests semantic
path commutation rather than ordinary accuracy under quantization.  Without
that exact-orbit causal construction and a prospective H7/H9 consequence,
IEA64 is not a paper contribution.

## Final reviewer-risk verdict

Semantic Arithmetic is mathematically clean, causally strong, and now has two
positive prospective long-horizon consequences: H7 survival extension and H9
post-breach attenuation.  It still has only moderate novelty after subtraction,
three-dataset breadth, no accuracy gain, one hardware/software stack, and a
canonicalization baseline that weakens engineering necessity.  H9 is also a
development-selected successor with a runtime-ordered test composition.

H4/H5/H6/H8 do not add the practical forecasting utility needed to overcome
those objections.  Semantic Arithmetic is therefore the honest alternative,
not the lead: it merits a randomized multi-dataset/multi-hardware confirmation
but does not displace the broader OrbitCover incumbent under the frozen rubric.
