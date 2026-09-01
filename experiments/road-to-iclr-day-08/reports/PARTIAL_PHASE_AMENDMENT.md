# Partial-phase exploratory amendment

Amended: 2026-08-31, after Phase II was stopped at 8,338 of 13,440 complete jobs and before inspecting aggregate Phase II or Phase III outcomes.

## Authorization and scope

The user authorized moving to later phases with the results available so far. The completed, checksum-valid Phase II jobs may therefore be analyzed and used for exploratory Phase III descriptor screening.

This amendment does not alter the frozen confirmatory protocol or the prospective Gate G2 thresholds. Partial outputs must be stored separately, marked `exploratory_partial`, and must not create a Phase II `DONE.json` marker.

## Interpretation constraints

- Coverage is model-dependent and incomplete, so naive cross-model rankings may reflect selection and coverage bias.
- Within-model summaries describe observed completed cells only.
- Phase III descriptor results are hypothesis-generating and cannot pass Gate G2 by themselves.
- Synthetic, remedy, pretraining, and final-benchmark work may be scoped from these results, but claims requiring a passed G2 remain unresolved until confirmatory evidence exists.
- No missing or failed result may be imputed as a successful or null outcome.

## Reproducibility

The analysis commands, exact source/config digests, job coverage, and missing-cell list must be retained with the partial snapshot.
