# Real Temporal Pilot — UCI Bike Sharing

Status: **NO-GO for current real-data TabALU and season router.**

## Source and protocol

The hourly UCI Bike Sharing dataset is downloaded from the official UCI
archive, pinned by SHA-256
`b70182d0d0508e9abbb79306ce5c0cec34869000f8220175ac83d11dbe845401`, and
identified by DOI `10.24432/C5W894` under CC BY 4.0. The loader verifies 17,379
rows. `casual` and `registered` are excluded because they sum to the target;
`cnt`, `instant`, and the explicit year flag are also excluded from predictors.

For each of five seeds, 2011 is randomly divided 70/15/15 into train,
validation, and IID test. All 8,734 hourly rows from 2012 form the untouched
future test. This is one dataset, so variation across seeds is not independent
dataset-level evidence.

## Result

| Model | IID NRMSE | 2012 future NRMSE | Future R² |
|---|---:|---:|---:|
| Typed-feature MLP | 0.247 | 1.289 | -0.676 |
| CatBoost | 0.263 | 0.605 | 0.634 |
| XGBoost | 0.267 | 0.602 | 0.637 |
| Random Forest | 0.322 | 0.626 | 0.609 |
| TabALU global typed program | 0.626 | 1.734 | -2.024 |
| TabALU season router | 0.590 | 9.779 | -108.6 |

The season router fails every prespecified check: it is 5.64× worse than the
global typed program, 16.2× worse than the best tree, and degrades 16.6× from
IID to future. All 60 planned records are finite; the experiment audit fails
because the scientific gate fails.

## Failure analysis

The sparse programs select quadratic elapsed-time terms on 2011. Separate
season experts estimate these terms from narrow time windows, producing large,
inconsistent coefficients that explode when evaluated one year later. The
global model is less extreme but still extrapolates the empirical 2011 trend
poorly. Exact execution faithfully executes a bad temporal program; it does not
make that program causally valid.

This invalidates any real temporal-stability claim from the synthetic regime
pilots. Season routing and the current open-loop time basis are excluded from a
real-data architecture. A bounded-time follow-up may diagnose whether explicit
temporal regularization prevents explosion, but it is post hoc and cannot turn
this test into confirmatory evidence. Multiple source-pinned datasets remain
required.

## Post-hoc bounded-time diagnosis

Removing unbounded elapsed and elapsed-squared terms while retaining hour,
weekday, and annual periodic operators reduces routed future NRMSE from 9.779
to 0.782, a 92% reduction. The bounded router is slightly better than the
bounded global program (0.802) but remains 30% worse than XGBoost (0.602).
Because this restriction was selected after inspecting the failure, it is
recorded under `results/real_bike_bounded_diagnostic/` as causal diagnosis only,
not a passed real-data gate.
