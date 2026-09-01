# H6 development-only note

These three bundles were observed before H6 was frozen and are excluded from
all gates.  This note documents the motivating pattern; it is not evidence for
generalization.

The fixed score fits `log10` mean FP32 orbit MSE at epochs 5, 10, and 20 and
extrapolates the line to epoch 200.

| Excluded bundle | Early slope / epoch | Predicted log10 MSE at 200 | Actual log10 MSE at 200 | Actual material (`>1e-5`) |
|---|---:|---:|---:|---|
| Bank / MLP / 8101 | 0.01586 | -12.19 | -11.21 | no |
| Bank / ResNet / 8101 | 0.16192 | 16.58 | -1.64 | yes |
| Credit / MLP / 8101 | 0.00777 | -13.62 | -11.17 | no |

The development rule separates these three cases, but its numeric forecast is
not calibrated: Bank/ResNet extrapolates to an impossible unsaturated scale.
H6 consequently gates discrimination (AUROC, sensitivity, specificity, and
rank correlation), not absolute forecast error.  The important prospective
test is whether the slope beats the raw epoch-20 level by at least 0.10 AUROC
on the 33 untouched bundles.
