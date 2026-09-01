# Closest-baseline result: strict fail, targeted continuation justified

> Follow-up: rank-4 covariance was rejected, while validation-only calibration succeeded. The calibrated diagonal model is within 2% of TACTiS on JenaWeather, beats it on Traffic, and has better average coverage. Electricity remains unresolved. See `FOLLOWUP_RESULTS.md`.

The frozen predictive gate **failed**. The failure is informative rather than terminal: the projective mixture is much better than official MOSES under this compute-limited protocol, is essentially tied with TACTiS-2 on JenaWeather, loses clearly on Electricity, and wins on Traffic. It also retains a very large query-time advantage.

## Primary result

Mean ensemble CRPS across four linear-query families (lower is better; three seeds):

| Dataset | Direct mixture | MOSES | Projective mixture | TACTiS-2 | Projective vs. TACTiS |
|---|---:|---:|---:|---:|---:|
| JenaWeather | 0.2280 | 0.3260 | **0.1794** | **0.1758** | +2.024% |
| Electricity | 0.2752 | 0.3600 | **0.2134** | **0.1870** | +14.083% |
| Traffic | 0.4070 | 0.5248 | **0.3148** | **0.3281** | -4.054% |

The frozen rule required the projective mixture to be within 2% of TACTiS on two datasets. It passed only Traffic. JenaWeather missed the boundary by roughly **0.00004 absolute CRPS** (2.024% rather than 2.000%), whereas Electricity is a substantive miss. It was within 2% of MOSES on all three datasets.

## What survived

- The projective mixture beats the capacity-matched direct scalar mixture on every dataset by 21–23% mean CRPS. This supports learning one joint law rather than unrelated query conditionals.
- It beats MOSES by 40–45% here, though this must not be presented as a definitive MOSES comparison: the official baselines saw fewer examples and received no validation tuning.
- It beats TACTiS in 5/9 seed cells for dense queries and 5/9 for scaled dense queries. It wins only 1/9 for point queries and 1/9 for differences. The evidence therefore aligns better with the **arbitrary linear-query** thesis than with a claim of universally superior forecasting.
- The average coverage-error gap against the better joint baseline is 2.82 percentage points, inside the frozen 3-point guardrail.
- Producing 256 samples and four query projections is about **256x faster** per context than the fastest official joint baseline in this implementation. The projective model took about 0.0046 ms/context on average, MOSES 1.17 ms, and TACTiS-2 5.10 ms.

## What did not survive

- The precommitted overall predictive gate did not pass.
- Electricity is not a threshold artifact: TACTiS is better across all four query families.
- Traffic calibration is weak for the projective mixture: its coverage error is 14.0%, versus 5.0% for TACTiS and 1.7% for MOSES. Its good Traffic CRPS comes with under-dispersed intervals.
- Raw projective consistency is not itself a novelty claim. [MOSES](https://proceedings.iclr.cc/paper_files/paper/2026/hash/899c6a43b9976e1077522fe5a39cafa3-Abstract-Conference.html) already guarantees marginalization consistency, and [TACTiS](https://proceedings.mlr.press/v162/drouin22a.html) is already a flexible joint copula forecaster. The remaining differentiator is exact, analytic closure for arbitrary linear queries, together with speed.

## Decision

This is **not yet ICLR-oral evidence**, and it should not be submitted in its current form. It is strong enough for one targeted iteration:

1. add capacity-matched low-rank covariance inside each mixture component while retaining exact linear projection;
2. add a calibration-aware objective or held-out variance calibration, aimed specifically at Traffic under-dispersion;
3. rerun this exact frozen evaluation first, with Electricity as the decisive expressivity test;
4. only if that version reaches TACTiS quality on Electricity without losing the dense-query and speed advantages, invest in equal-example training, validation tuning, more datasets/horizons, and statistical testing.

The most defensible paper thesis after this pilot is: **an analytically projective mixture can answer arbitrary linear temporal queries at near-copula quality and orders-of-magnitude lower query cost**. The current evidence supports the speed and dense-query halves, but not yet “near-copula quality” across datasets.

## Reproducibility

- Frozen protocol SHA-256: `204e6745f71ba719776cf8ce0bbb829a3d9c2f897571e4c3aa1af488c57d35df`
- MOSES official commit: `302aa7dd6a017ebb8390dcbcd2649264b92930e9`
- TACTiS official commit: `19df68b20b574f662fb1b2e1bf022f4116027f90`
- 18 official-baseline checkpoints, 36 evaluation cells, no failed or non-finite cells.
- Total wall time for the complete run: 1,326.7 seconds.
- The ensemble CRPS implementation was checked against `properscoring.crps_ensemble` to machine precision (maximum discrepancy `8.88e-16`).

Machine-readable outputs are in `outputs/audit.json`, `outputs/evaluation_cells.csv`, `outputs/evaluation_summary.csv`, and `outputs/training_cells.csv`.
