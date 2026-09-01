# Rank-4 and calibration follow-up

> Subsequent Electricity 2x2 screen: neither K8 nor a capacity-matched temporal Transformer improved the calibrated K4 MLP. No configuration advanced. See `REPRESENTATION_RESULTS.md`.

## Outcome

**Keep validation calibration; reject low-rank covariance.**

The frozen rank-4 repair protocol failed because rank-4 covariance did not improve Electricity. Six of its seven gates passed, including closeness to TACTiS on two datasets, dense-query quality, calibration, capacity, speed, and numerical stability. The sole failed gate was the one the change was designed to address.

The factorial addendum then showed that calibration alone provides the useful gain. Low rank adds no value beyond it.

## CRPS

Mean ensemble CRPS over four query families and three seeds:

| Dataset | Original diagonal | Calibrated diagonal | Calibrated rank-4 | TACTiS-2 |
|---|---:|---:|---:|---:|
| JenaWeather | 0.17937 | 0.17923 | **0.17825** | 0.17581 |
| Electricity | **0.21337** | **0.21312** | 0.21628 | 0.18703 |
| Traffic | 0.31477 | **0.31113** | 0.31390 | 0.32807 |

Relative to TACTiS, calibrated diagonal is:

- 1.95% worse on JenaWeather, inside the frozen 2% tolerance;
- 13.95% worse on Electricity, a substantive failure;
- 5.16% better on Traffic.

Thus the calibrated diagonal model satisfies the earlier “within 2% on two of three datasets” quality condition, but Electricity remains unresolved.

## Calibration

| Dataset | Original coverage error | Calibrated diagonal | Calibrated rank-4 | TACTiS-2 |
|---|---:|---:|---:|---:|
| JenaWeather | 1.93% | **1.55%** | 1.92% | 2.73% |
| Electricity | 4.27% | **1.43%** | 2.20% | 4.06% |
| Traffic | 14.01% | 6.89% | **5.81%** | 4.96% |

Validation-only diagonal calibration cuts Traffic coverage error by 7.12 percentage points and improves mean CRPS by 0.57%. Its mean coverage error across datasets is 3.29%, better than TACTiS-2's 3.92% in this pilot.

The fitted temperature behaves sensibly by dataset: approximately 0.94–0.99 for JenaWeather, 1.34–1.43 for Electricity, and 1.52–1.84 for Traffic. One global scale therefore corrects dataset-level dispersion without query-specific tuning or loss of projective consistency.

## Component decisions

### Validation covariance calibration: keep

- Passed its frozen component gate.
- Improves calibration substantially without degrading CRPS.
- Preserves the exact joint law and analytic projection formulas.
- Adds only one scalar per trained model.

### Rank-4 component covariance: reject for now

- Capacity was matched: 136,236 versus 136,580 parameters, a 0.25% difference.
- It made calibrated Electricity CRPS 1.48% worse than calibrated diagonal.
- Its mean calibrated CRPS was 0.70% worse than calibrated diagonal.
- It did not address the failure it was introduced to fix.

## Updated scientific diagnosis

Electricity is not primarily limited by second-order covariance rank or global dispersion. TACTiS is better there on point, difference, dense, and scaled-dense queries. The remaining candidates are:

1. **conditional distribution shape:** four Gaussian components may be insufficient;
2. **history representation:** the flattened MLP may miss temporal/channel structure that TACTiS's attention encoder captures;
3. **training objective:** random scalar-query NLL may not provide enough signal for the full future law.

The clean next screen is an Electricity-first, capacity-matched 2x2 ablation: 4 versus 8 mixture components, crossed with flattened MLP versus temporal/channel attention backbone. All four heads remain analytically projective. Only configurations that close at least half the 13.95% Electricity gap should advance to the three-dataset benchmark.

## ICLR assessment

This strengthens the method story but is still below oral-level evidence. The credible result is now:

> A calibrated analytic projective mixture matches a flexible copula model on two of three datasets, has better average calibration, and answers four query distributions roughly 868x faster in this implementation.

That is promising, but three datasets, unequal optimization exposure, and a clear Electricity failure are insufficient for a top-tier claim. The low-rank negative result is useful because it narrows the next work to richer conditional shape or a better temporal encoder.

## Reproducibility

- Rank-4 protocol SHA-256: `798cfff340c2fae989c6631dd06c4292fe8428e7fc6b5d09196f648a2b20c5ea`
- Calibration-control SHA-256: `410e5c0bb77e95599bac918ddcf93f95d026837d02645135dc39cb6962e62fa4`
- Rank-4: 9 training cells and 18 evaluation cells, all finite; wall time 137.2 seconds.
- Calibration control: 9 evaluation cells, all finite.
- Machine-readable audits: `lowrank_outputs/audit.json` and `calibration_control_outputs/audit.json`.
