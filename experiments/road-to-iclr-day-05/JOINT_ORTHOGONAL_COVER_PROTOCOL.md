# Day 5 adaptive follow-up: joint orthogonal nuisance cover

Status: frozen after the Cartesian factor-marginalization gate failed and
before any orthogonal-cover outcome was computed.

Freeze time: 2026-08-27 (Asia/Seoul).

## Motivation

The frozen Tier-1 action gate failed because exact marginalization pays the
full cardinality product of selected factors. At budget four it is worthwhile
only when one factor dominates strongly; the observed product fANOVA often
contains substantial interactions. This adaptive follow-up changes the action
class, not the observed data or primary loss.

## Estimator

Treat model seed as a fourth nuisance factor alongside feature order, opaque
category IDs, and target IDs. All four seeds/feature levels/category levels
have cardinality four and target IDs have cardinality two.

At budget four, use each seed exactly once, use every feature level exactly
once, use every category level exactly once, and use each target-ID level
twice. With seed positions fixed to `0,1,2,3`, enumerate all `4!` feature
permutations, all `4!` category permutations, and all six balanced binary
target assignments: 3,456 orthogonal-cover designs. Each of the 128 joint
schema--seed cells has equal inclusion probability, so every design-family
average is an unbiased estimator of the full quotient predictor.

Every strength-1 fANOVA main effect is annihilated within every design. Higher
order effects can alias and are averaged over the complete randomized design
family. No validation labels, attribution estimates, or dataset-specific
choice selects the construction.

## Comparators and endpoints

At the same four-fit budget:

- four iid draws from the full joint schema--seed product, whose exact
  expected residual is joint risk divided by four;
- all four ordinary seeds at one uniformly random schema, whose exact
  expected residual is persistent schema risk;
- the four-seed identity representation and the four-seed deterministic
  canonical representation as proper-loss baselines (they target particular
  representatives rather than the raw full quotient).

For Brier/MSE, expected proper loss of any unbiased quotient estimator equals
full-quotient proper loss plus estimator residual risk. The primary endpoint
is exact expected test residual to the full quotient; expected proper loss is
secondary but follows by the same identity.

The prospective adaptive gate passes if the joint orthogonal cover beats both
iid joint sampling and seed-only averaging on a strict majority of material
cells and in panel mean. A cell is material when joint schema--seed risk is at
least 0.5% of mean member loss. Dataset is the replication unit; model cells
within a dataset are repeated strata.

This result, if positive, is about compute-efficient quotient estimation. It
does not make orthogonal arrays new, does not guarantee worst-case behavior,
and remains conditional on the declared finite product and coupling.

