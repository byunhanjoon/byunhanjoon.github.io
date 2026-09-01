# Strength-2 cover: changed-subsample repeat

Status: frozen before outcomes.

This repeat isolates the capped-row sample as a robustness axis. Select one
task from each of the six confirmation source groups, with no outcome-based
replacement: Compustat direction, credit-card default, credit-card fraud,
HELOC, KDD17 return, and Polish bankruptcy three-year.

Keep the original Day-5 schema generators, model seeds `[101,202,303,404]`,
five models, training budgets, and 20,000/3,000/3,000 caps. Change only the
subsample RNG seed from `20260827` to `2026082813`. Enumerate the full
`4 feature x 4 category x 2 class x 4 model-seed` product and retain aligned
validation/test predictions.

Apply the already-fixed strength-2 OA-16 analysis. Screen material cells on
validation (`joint risk / mean member loss >= 0.005`) and evaluate their test
residual against iid-16, four independent strength-1 blocks, and four
seed-only blocks. Report cell and source-group results, with source group as
the external replication unit.

