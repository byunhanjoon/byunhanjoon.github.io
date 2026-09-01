# PHASECOVER PUBLISHED-BACKBONE CONFIRMATION RESULTS

## Decision

- Phase sensitivity transfers across backbones: **PASS**.
- Random-phase training is a remedy: **FAIL**.
- PhaseCover is an efficient quotient design: **PASS**.
- PhaseCover improves forecasting over exact IID4: **PASS**.
- Forecast advantage survives anchor/coset controls: **FAIL**.

## Dataset means over two seeds

| Dataset | Backbone | Training | materiality | canonical | exact IID4 | PhaseCover4 | full8 | quotient ratio | cover−IID |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Electricity | moment | canonical_train | 116.4% | 0.36447 | 0.73348 | 0.66582 | 0.72494 | 0.235 | -0.06766 |
| Electricity | moment | phase_aug_train | 61.5% | 0.55624 | 0.57731 | 0.54384 | 0.56515 | 0.238 | -0.03347 |
| Electricity | patchtst | canonical_train | 129.9% | 0.33263 | 0.72701 | 0.65750 | 0.71760 | 0.239 | -0.06951 |
| Electricity | patchtst | phase_aug_train | 41.4% | 0.65811 | 0.61955 | 0.60357 | 0.61122 | 0.255 | -0.01598 |
| JenaWeather | moment | canonical_train | 25.1% | 0.44226 | 0.46241 | 0.45618 | 0.46059 | 0.301 | -0.00623 |
| JenaWeather | moment | phase_aug_train | 22.6% | 0.45202 | 0.46070 | 0.45675 | 0.45912 | 0.336 | -0.00395 |
| JenaWeather | patchtst | canonical_train | 50.9% | 0.42167 | 0.52663 | 0.52423 | 0.52184 | 0.149 | -0.00240 |
| JenaWeather | patchtst | phase_aug_train | 25.8% | 0.46688 | 0.44506 | 0.44138 | 0.44267 | 0.424 | -0.00369 |
| Traffic | moment | canonical_train | 99.0% | 0.65766 | 1.13018 | 1.04537 | 1.11243 | 0.174 | -0.08481 |
| Traffic | moment | phase_aug_train | 51.6% | 1.00443 | 1.06832 | 1.02258 | 1.05297 | 0.220 | -0.04574 |
| Traffic | patchtst | canonical_train | 103.2% | 0.64167 | 1.09632 | 1.01515 | 1.07624 | 0.164 | -0.08118 |
| Traffic | patchtst | phase_aug_train | 35.0% | 1.01834 | 1.00434 | 0.98663 | 0.99572 | 0.238 | -0.01771 |

`cover−IID` is PhaseCover4 RMSE minus the exact mean over all 70 IID4 designs; negative is better.

## Frozen-gate details

### Phase sensitivity

- patchtst: 3/3 material datasets — PASS.
- moment: 3/3 material datasets — PASS.

### Phase augmentation

- patchtst: materiality reduction 64.0%; canonical RMSE change +53.5% — FAIL.
- moment: materiality reduction 43.5%; canonical RMSE change +37.4% — FAIL.

### Quotient and forecast groups

- patchtst/canonical_train: quotient 3/3, ratio 0.184 (PASS); forecast 3/3, mean delta -0.05103 (PASS).
- patchtst/phase_aug_train: quotient 3/3, ratio 0.305 (PASS); forecast 3/3, mean delta -0.01246 (PASS).
- moment/canonical_train: quotient 3/3, ratio 0.237 (PASS); forecast 3/3, mean delta -0.05290 (PASS).
- moment/phase_aug_train: quotient 3/3, ratio 0.265 (PASS); forecast 3/3, mean delta -0.02772 (PASS).

## Post-hoc anchor and complementary-coset audit

The frozen IID4 comparison includes phase 0 in only half of its designs, while PhaseCover `{0,2,4,6}`
always includes it. The table therefore adds an exact 35-design IID4 comparator conditioned on phase 0
and evaluates the equally spaced complementary coset `{1,3,5,7}`.

| Dataset | Backbone | Training | cover−anchored IID | complement−IID | complement−cover |
|---|---|---|---:|---:|---:|
| Electricity | moment | canonical_train | +0.00445 | +0.05383 | +0.12150 |
| Electricity | moment | phase_aug_train | +0.00002 | +0.01507 | +0.04854 |
| Electricity | patchtst | canonical_train | +0.00649 | +0.05424 | +0.12375 |
| Electricity | patchtst | phase_aug_train | -0.00909 | +0.00360 | +0.01958 |
| JenaWeather | moment | canonical_train | -0.00043 | +0.00367 | +0.00990 |
| JenaWeather | moment | phase_aug_train | -0.00004 | +0.00187 | +0.00582 |
| JenaWeather | patchtst | canonical_train | +0.02091 | -0.00535 | -0.00295 |
| JenaWeather | patchtst | phase_aug_train | -0.00166 | +0.00098 | +0.00467 |
| Traffic | moment | canonical_train | +0.01194 | +0.05440 | +0.13922 |
| Traffic | moment | phase_aug_train | -0.00296 | +0.02167 | +0.06741 |
| Traffic | patchtst | canonical_train | +0.01564 | +0.04674 | +0.12792 |
| Traffic | patchtst | phase_aug_train | -0.00534 | +0.00467 | +0.02238 |

Only 2/4 groups pass against phase-0-matched IID4; 0/4 pass for the complementary coset. The two cosets have identical
quotient MSE up to 9.31e-10, but sharply different forecast
accuracy. Thus spacing explains quotient efficiency, while the apparent accuracy win depends on the
arbitrary choice of the phase-0-anchored coset.

## Integrity and scope

- Protocol SHA-256 `020f883bda2a4cbe02d6406eb199255b58da34b7999a28775bb4ac40fa582826` matched: True.
- Exact reconstruction maximum error: 0.0.
- Complete cells 24/24; method rows 96; exact design rows 1,680; phase rows 192.
- Summed fit time: 378.0 seconds.
- Published implementations were used, but the experiment remains an eight-channel, mean-boundary-fill screen.
- MOMENT is a frozen-encoder linear probe; this is not a full TSFM fine-tuning comparison.
- Jena's finite `-9999` missing-value sentinel required the documented data-cleaning deviation in
  `PROTOCOL_DEVIATION.md`; the original invalid artifacts were retained in quarantine.

## Paper decision

The representation/quotient phenomenon transfers, but deterministic phase coverage still lacks a reliable
forecasting advantage. Continue only if the paper is reframed around invariance measurement or certified
quotient approximation; do not sell PhaseCover as an accuracy method.
