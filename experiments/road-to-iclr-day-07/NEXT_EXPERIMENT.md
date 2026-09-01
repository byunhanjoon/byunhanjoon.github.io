# Next decisive work — OrbitCover submission synthesis

## Goal

Turn the authoritative Day-5 closure into a compact, reviewer-resistant ICLR
paper without broadening claims beyond what the final common panel supports.

## Primary paper claim

Exact schema representations and stochastic training choices define a finite
nuisance distribution for a complete learning pipeline. The aligned
prediction barycenter is a quotient estimand. Interaction-balanced finite
designs and independent cross-scores estimate that quotient prediction and its
quadratic risk efficiently when the nuisance spectrum matches design strength.

## Required synthesis

1. Treat `../final_closure/results.md` as the authoritative empirical verdict:
   **PARTIALLY SUPPORTED**. Earlier Day-5 artifacts supply supporting history
   only.
2. Use only the exact theorem chain needed for the claim: ambiguity identity,
   product fANOVA filter, finite-population baseline, cross-score unbiasedness,
   and packing covariance.
3. Lead with the 144-cell independent-seed closure and 12-source inference;
   use earlier panels only as mechanism or prospective support.
4. Put the convergence, exact-function-matching, SRS, validation/test-shift,
   high-dimensional, and antithetic-CV boundaries in the main paper.
5. Compare directly with group averaging, randomized orthogonal arrays,
   antithetic CV, U-statistics, test-time augmentation, schema invariance, and
   ordinary ensembling. Claim none of those ingredients as new.
6. Frame the Day-7 conditional-value and learned-PFN studies as motivating
   failures/extensions, not evidence for OrbitCover's primary theorem.

## Submission figures

- nuisance product → aligned prediction tensor → quotient barycenter;
- fANOVA spectrum and exact design multipliers;
- 144-cell coupled versus independent residual comparison;
- convergence and exact-function-matching boundary;
- cross-score/packing budget frontier;
- validation fidelity versus held-out test-regret limitation.

## Final audit gates

- every headline number resolves to an authoritative JSON/CSV artifact;
- sources, not rows/seeds/fits, are the inferential unit;
- no result from an older report overrides `../final_closure/results.md`;
- no universal invariance, SOTA accuracy, antithetic optimality, or
  convergence-efficiency claim remains;
- abstract, theorem statements, tables, captions, and limitations use the same
  estimand and scope;
- the paper draft, reproducibility map, and claim-evidence matrix live in Day 7.

## Current state

The three required synthesis artifacts now exist:

- `ORBITCOVER_PAPER_DRAFT.md`;
- `CLAIM_EVIDENCE_MATRIX.md`;
- `REPRODUCIBILITY_MAP.md`.

Remaining release work is formatting the Markdown draft in the official ICLR
template and obtaining expert novelty review. No new empirical result is
needed to repair a failed claim.

## Stop condition

If the finite complete-pipeline quotient composition does not survive expert
novelty review against the classical components, stop the submission rather
than replacing it with the collided generic learned-kernel claim.
