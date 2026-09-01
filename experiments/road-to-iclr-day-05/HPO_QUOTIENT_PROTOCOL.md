# Day 5 prospective quotient-HPO protocol

Status: frozen before this panel was run and before any HPO outcome on these
four datasets was inspected.

Freeze time: 2026-08-27 (Asia/Seoul).

## Question and estimand

Does ordinary validation selection amplify prediction changes over an exact
schema orbit, and can a schema-quotient validation rule remove that selection
path without sacrificing held-out proper loss?

For a candidate `h`, schema representative `z`, validation labels `y`, and
aligned prediction `p[h,z]`, the quotient rule minimizes

`mean_z Brier(y, p[h,z])`.

For squared Hilbert losses this is exactly

`Brier(y, mean_z p[h,z]) + mean_z ||p[h,z] - mean_z p[h,z]||^2`.

Thus this is not an arbitrary robustness penalty: it is empirical risk under
the declared nuisance distribution, decomposed into quotient accuracy and
schema ambiguity. Ordinary per-schema tuning instead chooses a different
candidate at each `z` and creates an additional, data-dependent nuisance path.

## Prospective external panel

Use the checksum-audited official train/validation/test splits for Australian
Credit, Bank Marketing, German Credit, and LendingClub. These are external to
the earlier Adult/Churn/Otto selection pilot. Cap train/validation/test at
20,000/3,000/3,000 rows with the fixed Day-5 subsample.

Families are ordinal random forest and native CatBoost. Each has four frozen
capacity/regularization candidates. Model seeds are 511 and 733. Every
candidate is fit on the same 4 feature permutations x 4 opaque category-ID
maps x 2 target-ID maps. Validation and test predictions are aligned to
semantic class IDs before scoring.

## Policies

- `per_schema`: choose the minimum-validation-Brier candidate independently
  at every representative (ordinary rerun HPO).
- `identity_frozen`: tune on the identity representative and freeze that
  candidate across the orbit.
- `full_quotient`: minimize mean validation Brier over the full orbit; this is
  diagnostic because every nuisance level informs the rule.
- `development_quotient`: minimize mean validation Brier on feature levels
  0--1 x category levels 0--1 x both target maps, then freeze and evaluate on
  disjoint feature levels 2--3 x category levels 2--3 x both target maps.
- `development_minimax`: minimum worst validation Brier on the same
  development representatives; included as a prespecified distributionally
  robust comparator.

Ties use the first candidate. Test labels never affect selection.

## Endpoints and gate

Primary endpoints are aligned test schema risk and Brier of the quotient
centroid. Candidate-switch entropy and exact switch-dispersion/covariance
decomposition are mechanism endpoints. Dataset--family is the primary
replication unit; the two seeds are paired repeated measurements.

The prospective gate passes if `development_quotient` has lower mean schema
risk than `per_schema` on a strict majority of the eight dataset--family
cells, its panel-mean schema risk is lower, and its panel-mean Brier is no more
than 0.1% of reference Brier worse. Results versus `identity_frozen` and
`development_minimax` are secondary and are reported regardless of sign.

Any claim must remain conditional on this finite candidate set and declared
uniform nuisance distribution.

