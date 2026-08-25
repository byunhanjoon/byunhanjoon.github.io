# Rejected exploratory identity fallback

These files preserve the first diagnostic run in which numeric values unseen in
training mapped to an all-zero identity fallback. The training spans matched,
but validation/test were therefore not an exact global reparameterization.

They are excluded from `all_results.csv`, every headline analysis, and
`REPORT_DAY3.md`. The accepted experiment uses the exact affine extension and
is saved as `../ple_identity_whitening_exact.csv` with equivalence diagnostics
in `../ple_identity_equivalence_exact.json`.
