# Native Feature Geometry — reviewer attack audit

Status: **FINAL AFTER H1–H6**

## 1. “The synthetic target was designed to be smooth.”

Correct.  The experiment is a mechanism test: if a declared metric truly
governs a feature, can it transport a learned lookup chart?  It cannot establish
that real tabular features usually have such metrics.  The nominal/random
control limits but does not remove this concern.

## 2. “A full 15-dimensional embedding of 16 values is just a change of basis.”

Correct.  The complete centered spectral basis can represent any category
table.  Proposition 6 shows that the actual method is kernel interpolation:
the Gram relations choose how observed rows extend to held values.  No compact
representation, dimensionality reduction, or literal recovery of spectral axes
is claimed.  Reduced-rank tests remain necessary.

## 3. “H3’s impressive correlation is an interface confound.”

Largely correct.  The frozen pooled gate passes (`rho=-.822` for held MSE), but
within the unconstrained learned interface it falls to `-.121`; within native-
tuned it is `-.450` and inconsistent by domain.  H3 is therefore changed, not
retained as a broad CKA law.  Prospective H6 supplies the relevant within-model
dose response without changing interface identity.

## 4. “The fixed native encoder failed its own negative control.”

Correct.  Although H4 wins all 9 structured cells with 66.6% median reduction,
it also improves nominal/random held MSE by 25.8%, exceeding the frozen <5%
boundary.  H4 is falsified.  Chance extrapolation from a fixed full-rank table
can look beneficial with only three nominal seeds; fixed-interface accuracy is
not the surviving claim.

## 5. “Schema invariance is built in.”

Yes.  Native tables are indexed by semantic identity, so their zero chart
variance is a construction property.  It is useful engineering behavior but
not empirical evidence or novelty.  Only the performance consequences and
metric corruption response are empirical.

## 6. “This does not test the most interesting schema choices.”

Correct.  Charts are category-code permutations.  The pilot does not compare a
flat leaf field against multi-column hierarchical paths, alternative binning,
unit changes, redundant decompositions, or database normalization choices.
Those are required next.

## 7. “The metric could leak the target.”

The synthetic metric is generated before targets and the corruption is
outcome-independent, but the analytic target is intentionally smooth in that
metric.  On real data, metadata provenance must be frozen without target access
and evaluated under temporal or source holdout.

## 8. “Kernel and zero-shot transfer already exist.”

Correct.  Concept kernels, periodic encoders, cold-start recommendation,
zero-shot semantic-kernel transfer, and unseen-category tabular encoding are
close prior art.  The residual distinction is narrow: transport task-trained
neural embedding rows using a typed value metric, coupled to schema-risk and a
causal corruption-dose test.  No “first” claim is supportable.

## 9. “H6 used known endpoints.”

Correct and disclosed in its protocol.  The new H6 evidence is the five-dose
shape: all 18 structured interface × cell curves are perfectly monotone.  The
known alpha-0/alpha-1 ordering is only a replay sanity check.

## 10. “Rows and charts are pseudoreplicates.”

They are treated as repeated measurements.  Domain × seed is the pilot summary
unit; there are only nine structured cells.  No row-level p-values are used.

## 11. “Evidence is too narrow.”

Yes: one synthetic regression generator, one MLP, 16-value domains, three
structured types, three seeds, one training budget, and correct complete metric
metadata.  The direction remains a mechanism candidate rather than a paper
lead.

