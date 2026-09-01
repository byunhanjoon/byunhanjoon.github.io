# External OpenML model-selection protocol

Status: frozen before external cover outcomes on 2026-08-28.

Using the exact OpenML tensors declared in
`OPENML_EXTERNAL_COVER_PROTOCOL.md`, repeat the budget-16 four-candidate
validation selection experiment with 1,024 randomized actions per dataset.
Use shared nuisance coordinates across candidates and compare strength-2,
IID-16, four strength-1 blocks, and four seed blocks. All outcomes are held-out
Brier scores.

The external selection gate requires strength-2 to have lower equal-dataset
mean realized test Brier than all three controls, to beat IID on at least 6/8
datasets, and to agree with the full-quotient validation winner more often than
IID. Dataset bootstrap intervals are descriptive for this fixed panel.

## Post-failure stronger-control diagnostic

After observing failure, additionally compare SRSWOR-16, scrambled Sobol-16,
and Latin-hypercube-16 under identical candidate coupling. These comparisons
are explanatory only and cannot change the prospective gate above.
