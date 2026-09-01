# H6 — Metric-Corruption Dose Response

Status: **FROZEN BEFORE INTERMEDIATE-DOSE OUTCOMES**

This successor was motivated by the complete H1–H5 pilot.  The endpoint
comparison (`alpha=0` correct versus `alpha=1` shuffled) is already known from
H5 and cannot count as new confirmation.  The new evidence is the shape at
intermediate doses within the same frozen network.

## Intervention

For each stored semantic kernel `K` and the outcome-independent row corruption
`P` already used by H5, define

`K_alpha = (1-alpha) K + alpha P K P^T`

for `alpha in [0, .25, .50, .75, 1]`.  Positive semidefiniteness is preserved
by convexity.  Compile the complete centered spectral coordinates of each
`K_alpha`, fit the same affine ridge chart transport on observed categories,
and patch only the four held embedding rows.

The downstream model, observed embedding rows, data, chart, optimizer result,
and test examples remain fixed across dose.  Models are deterministically
replayed from the frozen pilot code, and their unpatched predictions must match
the original artifacts exactly.

## Matrix

- domains: cycle16, ordinal16, tree16, and nominal16 control;
- seeds: 7301–7303;
- five charts per domain × seed;
- interfaces: unconstrained `learned` and `native_tuned`;
- 12 replay bundles, 120 trained paths, 600 dose interventions.

The structured domain × seed cell is the independent summary unit.  Charts and
doses are paired repeated measurements.

## Frozen gate

H6 passes only if all conditions hold:

1. chart-mean held MSE at `alpha=0` is below `alpha=1` in all 9 structured
   cells for each interface (known endpoint sanity check, not new evidence);
2. Spearman correlation between dose and chart-mean held MSE is at least 0.80
   in at least 7/9 cells separately for `learned` and `native_tuned`;
3. the pooled median relative endpoint reduction is at least 50% for each
   interface;
4. deterministic replay error and maximum seen-category prediction change are
   exactly zero in stored float32 predictions;
5. nominal16 is invariant to the dose up to maximum relative MSE range
   `1e-5`, because permuting the equality kernel leaves it unchanged.

Failure means H5 is a binary correct-versus-bad control effect, not evidence of
a graded metric mechanism.  Passing remains synthetic and does not upgrade the
practical evidence tier.

