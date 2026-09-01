# Theory checks

The checks are executable in `test_geometry_transfer.py` and the raw Monte
Carlo results are produced by `run_synthetic.py`.

## Symbolic expansion

For scalar `u`, transferred mean `g`, and zero-mean estimation error `e` with
variance `v`:

```text
u^2 - E[(u-g-e)^2]
= u^2 - (u^2 - 2ug + g^2 + v)
= 2ug - g^2 - v.
```

The multivariate trace follows from
`E[e^T A^T Q A e]=tr(A^T Q A Sigma)=tr(Q A Sigma A^T)`.
The cross term is `2 mu_U^T Q A mu_T`; no transpose or extra factor is omitted.

## Numerical checks

The frozen seed is `20260829`. The identity suite spans 972 combinations of
state count, operator, signal, state-estimation noise, and rows per state, with
600 training/test replications per cell. `raw/synthetic/summary.json` records
the resulting correlation, calibration, discrepancy, and sign accuracy.

Additional checks cover:

- equality of the norm and expanded forms for dense non-diagonal covariance;
- cancellation of arbitrary fresh test-noise variance;
- the spectral sum against the matrix identity;
- exact MPE/direct-similarity factorization;
- identical geometry statistics with aligned and anti-aligned targets;
- the `GTR=1` phase boundary;
- lower noise cost as per-state sample size increases.

## Important oracle arithmetic check

For a realized training estimate and observed test residual mean, the row-level
state-balanced MSE gain is exactly

```text
2 mu_U_sample^T Q A mu_hat_T - ||A mu_hat_T||_Q^2.
```

This is deliberately reported separately from the population expected-risk
decomposition. Treating their near equality as independent prediction would be
circular.
