# Basis Dependence of Tabular Learning — Confirmation Round

## Executive Verdict

PHENOMENON-STRONG-METHOD-UNSOLVED

## One-Paragraph Conclusion

Condition-number-one changes of basis produced meaningful prediction changes in 72.7% of development dataset×model units across all five model families, and natural one-hot/Helmert, local/spectral-hat, and Fourier-origin pairs also disagreed (overall median 0.0566). Function matching reduced epoch-0 differences below numerical tolerance, function-matched SGD retained equivalence, and AdamW rebuilt disagreement (final medians 1.466e-08 versus 0.0842), identifying optimizer coordinate geometry as a mechanism. PCA canonicalization was the only viable non-oracle repair and reduced median disagreement by 100.0% in development and 100.0% on the untouched holdout, but its median task-error changes were +2.9% and +17.2%. Therefore the phenomenon and mechanism are strong, while the method is not yet a performance-preserving general solution.

## Frozen Protocol

- git commit: `0c456660ae9a87aab7932b569e1954b0ee1d25fe`
- hardware: NVIDIA H100 NVL; Linux-5.4.0-216-generic-x86_64-with-glibc2.31; CUDA 12.6
- package versions: catboost 1.2.10, numpy 1.26.4, pandas 2.3.3, pytabkit 1.7.3, scikit-learn 1.4.2, scipy 1.11.4, tabicl 2.0.3, tabpfn 8.5.0, torch 2.7.0
- seeds: 0, 1, 2; split/assignment seed 20260901; eight orbit members
- model checkpoints: tabicl_v2: `0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a`; tabicl_v2: `bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0`; tabpfn_2_6: `0578fa56f97e11024e31735aaec2c4e7332584b7730242fbaf6c0bbd0299206a`; tabpfn_2_6: `c855169116d79d2c957a29107ffc04464555f39aec0eef90c90390f2c2b2008b`; TabM-D: pytabkit 1.7.3 implementation; CatBoost: package 1.2.10; controlled MLP: protocol-defined 3×256 GELU
- development datasets: california_housing, house_16H, pendigits, credit-g, adult, bike-sharing, bank-marketing, cpu_small, phoneme, wine-quality-red, diamonds
- prospective datasets: jannis, elevators, houses, MagicTelescope, pol, optdigits, segment
- frozen method config SHA256: `2a6378522943f2cb767de199ab4df4e1a1797e750d951d11b8e6ce2ba0131329`

## 1. Orthogonal Basis Replication

| dataset | model | one-block disagreement | all-block disagreement | task original | orbit mean | orbit worst |
| --- | --- | --- | --- | --- | --- | --- |
| adult | catboost | 0.0501 | 0.0581 | 0.3440 | 0.3463 | 0.3505 |
| adult | controlled_mlp | 0.0112 | 0.0301 | 0.3884 | 0.3827 | 0.3923 |
| adult | tabicl_v2 | 0.0263 | 0.0425 | 0.3380 | 0.3367 | 0.3394 |
| adult | tabm_d | 0.0532 | 0.0525 | 0.3551 | 0.3624 | 0.3873 |
| adult | tabpfn_2_6 | 0.0156 | 0.0362 | 0.3486 | 0.3477 | 0.3530 |
| bank-marketing | catboost | 0.0422 | 0.0470 | 0.2546 | 0.2600 | 0.2667 |
| bank-marketing | controlled_mlp | 0.0254 | 0.0399 | 0.2534 | 0.2520 | 0.2582 |
| bank-marketing | tabicl_v2 | 0.0237 | 0.0422 | 0.2369 | 0.2433 | 0.2466 |
| bank-marketing | tabm_d | 0.0038 | 0.0087 | 0.4519 | 0.4456 | 0.4493 |
| bank-marketing | tabpfn_2_6 | 0.0151 | 0.0440 | 0.2315 | 0.2428 | 0.2521 |
| bike-sharing | catboost | 0.0529 | 0.0867 | 59.2249 | 59.8171 | 61.3032 |
| bike-sharing | controlled_mlp | 0.0987 | 0.1524 | 77.8382 | 75.9962 | 79.7526 |
| bike-sharing | tabicl_v2 | 0.1198 | 0.2112 | 70.5319 | 78.0698 | 83.9536 |
| bike-sharing | tabm_d | 0.0390 | 0.1501 | 63.4810 | 61.1057 | 64.3862 |
| bike-sharing | tabpfn_2_6 | 0.0261 | 0.0751 | 53.8270 | 55.3222 | 56.8870 |
| california_housing | catboost | 0.0872 | 0.1240 | 61831.7425 | 62156.7744 | 63130.3184 |
| california_housing | controlled_mlp | 0.1284 | 0.1300 | 64967.5198 | 65732.3568 | 66358.1767 |
| california_housing | tabicl_v2 | 0.0972 | 0.1864 | 58786.4727 | 61148.1856 | 63013.4175 |
| california_housing | tabm_d | 0.0527 | 0.1740 | 61010.7920 | 63040.2399 | 64750.4122 |
| california_housing | tabpfn_2_6 | 0.0726 | 0.1801 | 53263.7016 | 55457.9941 | 56563.4531 |
| cpu_small | catboost | 0.0438 | 0.0902 | 4.5757 | 4.5942 | 4.9219 |
| cpu_small | controlled_mlp | 0.0583 | 0.0767 | 3.8722 | 3.8976 | 4.0028 |
| cpu_small | tabicl_v2 | 0.0780 | 0.0974 | 3.2584 | 3.4033 | 3.5469 |
| cpu_small | tabm_d | 0.0522 | 0.0975 | 3.5642 | 3.8828 | 4.6111 |
| cpu_small | tabpfn_2_6 | 0.0351 | 0.0706 | 2.8803 | 3.0501 | 3.1041 |
| credit-g | catboost | 0.0842 | 0.0865 | 0.4786 | 0.4881 | 0.4990 |
| credit-g | controlled_mlp | 0.0457 | 0.0532 | 0.4970 | 0.5069 | 0.5186 |
| credit-g | tabicl_v2 | 0.0353 | 0.0517 | 0.4758 | 0.4823 | 0.4921 |
| credit-g | tabm_d | 0.0499 | 0.0581 | 0.4843 | 0.4922 | 0.5070 |
| credit-g | tabpfn_2_6 | 0.0285 | 0.0471 | 0.4870 | 0.4927 | 0.4993 |
| diamonds | catboost | 0.0369 | 0.0637 | 842.1246 | 892.8417 | 920.5357 |
| diamonds | controlled_mlp | 0.0257 | 0.0382 | 687.8116 | 679.5005 | 693.2175 |
| diamonds | tabicl_v2 | 0.0637 | 0.1311 | 1006.4715 | 1117.7571 | 1153.5923 |
| diamonds | tabm_d | 0.0165 | 0.0749 | 677.1438 | 687.6006 | 719.2464 |
| diamonds | tabpfn_2_6 | 0.0141 | 0.0750 | 646.5863 | 659.2437 | 689.1423 |
| house_16H | catboost | 0.1102 | 0.1551 | 34618.5873 | 35302.0655 | 35727.8483 |
| house_16H | controlled_mlp | 0.1453 | 0.1951 | 34877.3765 | 35131.4445 | 35841.5956 |
| house_16H | tabicl_v2 | 0.1115 | 0.3277 | 31199.8290 | 31705.8466 | 33214.6528 |
| house_16H | tabm_d | 0.0712 | 0.2478 | 32856.1900 | 33272.7705 | 34562.8925 |
| house_16H | tabpfn_2_6 | 0.0536 | 0.1970 | 31166.2000 | 32518.7482 | 33683.9835 |
| pendigits | catboost | 0.0098 | 0.0213 | 0.2023 | 0.2235 | 0.2283 |
| pendigits | controlled_mlp | 0.0154 | 0.0340 | 0.1911 | 0.1908 | 0.2145 |
| pendigits | tabicl_v2 | 0.0059 | 0.0236 | 0.0318 | 0.0371 | 0.0421 |
| pendigits | tabm_d | 0.0207 | 0.0372 | 0.0768 | 0.1084 | 0.1251 |
| pendigits | tabpfn_2_6 | 0.0084 | 0.0304 | 0.0588 | 0.0809 | 0.0903 |
| phoneme | catboost | 0.0619 | 0.0813 | 0.2985 | 0.3101 | 0.3194 |
| phoneme | controlled_mlp | 0.0317 | 0.0476 | 0.3568 | 0.3502 | 0.3593 |
| phoneme | tabicl_v2 | 0.0453 | 0.0741 | 0.2752 | 0.2913 | 0.2948 |
| phoneme | tabm_d | 0.0599 | 0.0900 | 0.3303 | 0.3358 | 0.3468 |
| phoneme | tabpfn_2_6 | 0.0727 | 0.0950 | 0.2738 | 0.2863 | 0.2941 |
| wine-quality-red | catboost | 0.0950 | 0.1375 | 0.6483 | 0.6496 | 0.6590 |
| wine-quality-red | controlled_mlp | 0.0949 | 0.1281 | 0.6924 | 0.7017 | 0.7136 |
| wine-quality-red | tabicl_v2 | 0.0792 | 0.1507 | 0.6392 | 0.6377 | 0.6463 |
| wine-quality-red | tabm_d | 0.0556 | 0.3062 | 0.6731 | 0.7018 | 0.7185 |
| wine-quality-red | tabpfn_2_6 | 0.1170 | 0.2150 | 0.6384 | 0.6275 | 0.6375 |

Across units, the all-block median disagreement was 0.0767; 72.7% met the preregistered meaningful-effect threshold. Median seed SD was 0.0033. The dataset-bootstrap intervals are in `results/processed/bootstrap_intervals.csv`.

## 2. Condition<=3 Results

| dataset | model | mean disagreement | max disagreement | task original | orbit mean | orbit worst |
| --- | --- | --- | --- | --- | --- | --- |
| adult | catboost | 0.0612 | 0.0658 | 0.3440 | 0.3424 | 0.3509 |
| adult | controlled_mlp | 0.0347 | 0.0481 | 0.3884 | 0.3829 | 0.4147 |
| adult | tabicl_v2 | 0.0424 | 0.0483 | 0.3380 | 0.3364 | 0.3469 |
| adult | tabm_d | 0.0493 | 0.0780 | 0.3551 | 0.3560 | 0.3809 |
| adult | tabpfn_2_6 | 0.0365 | 0.0472 | 0.3486 | 0.3438 | 0.3495 |
| bank-marketing | catboost | 0.0477 | 0.0554 | 0.2546 | 0.2607 | 0.2661 |
| bank-marketing | controlled_mlp | 0.0453 | 0.0630 | 0.2534 | 0.2503 | 0.2557 |
| bank-marketing | tabicl_v2 | 0.0465 | 0.0571 | 0.2369 | 0.2454 | 0.2524 |
| bank-marketing | tabm_d | 0.0092 | 0.0102 | 0.4519 | 0.4497 | 0.4529 |
| bank-marketing | tabpfn_2_6 | 0.0422 | 0.0511 | 0.2315 | 0.2415 | 0.2502 |
| bike-sharing | catboost | 0.0863 | 0.0956 | 59.2249 | 60.1396 | 61.1725 |
| bike-sharing | controlled_mlp | 0.1686 | 0.2147 | 77.8382 | 75.9403 | 85.0051 |
| bike-sharing | tabicl_v2 | 0.2051 | 0.2404 | 70.5319 | 76.6990 | 82.3533 |
| bike-sharing | tabm_d | 0.1595 | 0.1976 | 63.4810 | 60.2521 | 62.9654 |
| bike-sharing | tabpfn_2_6 | 0.0755 | 0.0847 | 53.8270 | 55.5730 | 56.8091 |
| california_housing | catboost | 0.1253 | 0.1416 | 61831.7425 | 62390.7425 | 63655.8634 |
| california_housing | controlled_mlp | 0.1546 | 0.1991 | 64967.5198 | 66477.4392 | 67413.0664 |
| california_housing | tabicl_v2 | 0.1870 | 0.1947 | 58786.4727 | 60998.0265 | 63093.7383 |
| california_housing | tabm_d | 0.1773 | 0.1908 | 61010.7920 | 62967.3032 | 63953.1517 |
| california_housing | tabpfn_2_6 | 0.1783 | 0.2006 | 53263.7016 | 56200.4844 | 57801.1366 |
| cpu_small | catboost | 0.0905 | 0.1037 | 4.5757 | 4.5077 | 4.7584 |
| cpu_small | controlled_mlp | 0.0852 | 0.1083 | 3.8722 | 3.9471 | 4.1262 |
| cpu_small | tabicl_v2 | 0.1003 | 0.1041 | 3.2584 | 3.4647 | 3.5426 |
| cpu_small | tabm_d | 0.1033 | 0.1452 | 3.5642 | 4.0011 | 4.6554 |
| cpu_small | tabpfn_2_6 | 0.0752 | 0.0851 | 2.8803 | 3.0947 | 3.1559 |
| credit-g | catboost | 0.0852 | 0.0945 | 0.4786 | 0.4928 | 0.5073 |
| credit-g | controlled_mlp | 0.0601 | 0.0726 | 0.4970 | 0.5054 | 0.5188 |
| credit-g | tabicl_v2 | 0.0503 | 0.0578 | 0.4758 | 0.4806 | 0.4878 |
| credit-g | tabm_d | 0.0590 | 0.0784 | 0.4843 | 0.4919 | 0.5055 |
| credit-g | tabpfn_2_6 | 0.0447 | 0.0513 | 0.4870 | 0.4935 | 0.5018 |
| diamonds | catboost | 0.0642 | 0.0808 | 842.1246 | 887.8307 | 906.4420 |
| diamonds | controlled_mlp | 0.0448 | 0.0552 | 687.8116 | 680.0758 | 701.0986 |
| diamonds | tabicl_v2 | 0.1263 | 0.1348 | 1006.4715 | 1101.4683 | 1128.6293 |
| diamonds | tabm_d | 0.0746 | 0.1150 | 677.1438 | 690.5016 | 718.3472 |
| diamonds | tabpfn_2_6 | 0.0774 | 0.0894 | 646.5863 | 665.1473 | 701.3229 |
| house_16H | catboost | 0.1623 | 0.1791 | 34618.5873 | 35525.2026 | 36217.4690 |
| house_16H | controlled_mlp | 0.2561 | 0.3357 | 34877.3765 | 35830.8629 | 37083.7792 |
| house_16H | tabicl_v2 | 0.3143 | 0.3634 | 31199.8290 | 31742.7517 | 32858.4349 |
| house_16H | tabm_d | 0.2606 | 0.2891 | 32856.1900 | 33423.8531 | 34474.3010 |
| house_16H | tabpfn_2_6 | 0.2035 | 0.2283 | 31166.2000 | 32422.9715 | 33466.2142 |
| pendigits | catboost | 0.0224 | 0.0248 | 0.2023 | 0.2279 | 0.2364 |
| pendigits | controlled_mlp | 0.0386 | 0.0458 | 0.1911 | 0.1882 | 0.2236 |
| pendigits | tabicl_v2 | 0.0241 | 0.0303 | 0.0318 | 0.0396 | 0.0486 |
| pendigits | tabm_d | 0.0390 | 0.0445 | 0.0768 | 0.1117 | 0.1366 |
| pendigits | tabpfn_2_6 | 0.0302 | 0.0342 | 0.0588 | 0.0763 | 0.0869 |
| phoneme | catboost | 0.0804 | 0.0899 | 0.2985 | 0.3079 | 0.3194 |
| phoneme | controlled_mlp | 0.0702 | 0.0869 | 0.3568 | 0.3531 | 0.3671 |
| phoneme | tabicl_v2 | 0.0720 | 0.0810 | 0.2752 | 0.2908 | 0.2967 |
| phoneme | tabm_d | 0.0929 | 0.1065 | 0.3303 | 0.3335 | 0.3467 |
| phoneme | tabpfn_2_6 | 0.0950 | 0.1037 | 0.2738 | 0.2838 | 0.2966 |
| wine-quality-red | catboost | 0.1394 | 0.1521 | 0.6483 | 0.6487 | 0.6553 |
| wine-quality-red | controlled_mlp | 0.1749 | 0.2656 | 0.6924 | 0.7082 | 0.7267 |
| wine-quality-red | tabicl_v2 | 0.1603 | 0.1764 | 0.6392 | 0.6403 | 0.6514 |
| wine-quality-red | tabm_d | 0.3189 | 0.3395 | 0.6731 | 0.6945 | 0.7150 |
| wine-quality-red | tabpfn_2_6 | 0.2228 | 0.2667 | 0.6384 | 0.6309 | 0.6423 |

All measured transform condition numbers satisfied the bound; the median condition≤3 disagreement was 0.0804. The nonzero orthogonal result already rules out poor conditioning as the sole explanation.

## 3. Natural Equivalent Bases

### One-hot vs Helmert

| dataset | model | basis pair | reconstruction error | disagreement | performance delta |
| --- | --- | --- | --- | --- | --- |
| adult | catboost | onehot_vs_helmert | 9.573e-17 | 0.0520 | -0.0133 |
| adult | controlled_mlp | onehot_vs_helmert | 9.573e-17 | 0.0164 | 0.0015 |
| adult | tabicl_v2 | onehot_vs_helmert | 9.573e-17 | 0.0150 | 0.0009 |
| adult | tabm_d | onehot_vs_helmert | 9.573e-17 | 0.0397 | 0.0006 |
| adult | tabpfn_2_6 | onehot_vs_helmert | 9.573e-17 | 0.0296 | -0.0012 |
| bank-marketing | catboost | onehot_vs_helmert | 1.770e-16 | 0.0456 | -0.0038 |
| bank-marketing | controlled_mlp | onehot_vs_helmert | 1.770e-16 | 0.0448 | 0.0047 |
| bank-marketing | tabicl_v2 | onehot_vs_helmert | 1.770e-16 | 0.0365 | 0.0045 |
| bank-marketing | tabm_d | onehot_vs_helmert | 1.770e-16 | 0.0042 | -0.0005 |
| bank-marketing | tabpfn_2_6 | onehot_vs_helmert | 1.770e-16 | 0.0437 | 0.0126 |
| bike-sharing | catboost | onehot_vs_helmert | 1.792e-16 | 0.0579 | -0.5668 |
| bike-sharing | controlled_mlp | onehot_vs_helmert | 1.792e-16 | 0.0893 | -2.1863 |
| bike-sharing | tabicl_v2 | onehot_vs_helmert | 1.792e-16 | 0.1175 | -3.4017 |
| bike-sharing | tabm_d | onehot_vs_helmert | 1.792e-16 | 0.0450 | -0.6649 |
| bike-sharing | tabpfn_2_6 | onehot_vs_helmert | 1.792e-16 | 0.0736 | 0.4467 |
| california_housing | catboost | onehot_vs_helmert | 1.602e-16 | 0.0876 | -194.9415 |
| california_housing | controlled_mlp | onehot_vs_helmert | 1.602e-16 | 0.0869 | -272.8274 |
| california_housing | tabicl_v2 | onehot_vs_helmert | 1.602e-16 | 0.0427 | 67.7095 |
| california_housing | tabm_d | onehot_vs_helmert | 1.602e-16 | 0.0449 | 130.9782 |
| california_housing | tabpfn_2_6 | onehot_vs_helmert | 1.602e-16 | 0.1120 | 1007.0042 |
| credit-g | catboost | onehot_vs_helmert | 1.626e-16 | 0.0846 | 0.0074 |
| credit-g | controlled_mlp | onehot_vs_helmert | 1.626e-16 | 0.0565 | 0.0088 |
| credit-g | tabicl_v2 | onehot_vs_helmert | 1.626e-16 | 0.0352 | 0.0044 |
| credit-g | tabm_d | onehot_vs_helmert | 1.626e-16 | 0.0724 | 0.0173 |
| credit-g | tabpfn_2_6 | onehot_vs_helmert | 1.626e-16 | 0.0589 | 0.0082 |
| diamonds | catboost | onehot_vs_helmert | 1.940e-16 | 0.0597 | -11.0184 |
| diamonds | controlled_mlp | onehot_vs_helmert | 1.940e-16 | 0.0311 | 1.6541 |
| diamonds | tabicl_v2 | onehot_vs_helmert | 1.940e-16 | 0.0940 | -99.4534 |
| diamonds | tabm_d | onehot_vs_helmert | 1.940e-16 | 0.0280 | 3.9483 |
| diamonds | tabpfn_2_6 | onehot_vs_helmert | 1.940e-16 | 0.0495 | -18.0924 |

### Local spline vs spectral spline

| dataset | model | basis pair | reconstruction error | disagreement | performance delta |
| --- | --- | --- | --- | --- | --- |
| adult | catboost | local_hat_vs_spectral_hat | 2.126e-16 | 0.0532 | 0.0050 |
| adult | controlled_mlp | local_hat_vs_spectral_hat | 2.126e-16 | 0.0213 | -0.0043 |
| adult | tabicl_v2 | local_hat_vs_spectral_hat | 2.126e-16 | 0.0345 | 0.0052 |
| adult | tabm_d | local_hat_vs_spectral_hat | 2.126e-16 | 0.0819 | 0.0118 |
| adult | tabpfn_2_6 | local_hat_vs_spectral_hat | 2.126e-16 | 0.0218 | 0.0059 |
| bank-marketing | catboost | local_hat_vs_spectral_hat | 2.150e-16 | 0.0435 | 0.0028 |
| bank-marketing | controlled_mlp | local_hat_vs_spectral_hat | 2.150e-16 | 0.0145 | 0.0007 |
| bank-marketing | tabicl_v2 | local_hat_vs_spectral_hat | 2.150e-16 | 0.0301 | -0.0099 |
| bank-marketing | tabm_d | local_hat_vs_spectral_hat | 2.150e-16 | 0.0040 | -0.0020 |
| bank-marketing | tabpfn_2_6 | local_hat_vs_spectral_hat | 2.150e-16 | 0.0158 | 0.0010 |
| bike-sharing | catboost | local_hat_vs_spectral_hat | 2.142e-16 | 0.0601 | 0.3045 |
| bike-sharing | controlled_mlp | local_hat_vs_spectral_hat | 2.142e-16 | 0.0817 | 1.2285 |
| bike-sharing | tabicl_v2 | local_hat_vs_spectral_hat | 2.142e-16 | 0.1215 | 3.1981 |
| bike-sharing | tabm_d | local_hat_vs_spectral_hat | 2.142e-16 | 0.0354 | 0.7793 |
| bike-sharing | tabpfn_2_6 | local_hat_vs_spectral_hat | 2.142e-16 | 0.0444 | 0.3233 |
| california_housing | catboost | local_hat_vs_spectral_hat | 2.085e-16 | 0.0873 | 21.6600 |
| california_housing | controlled_mlp | local_hat_vs_spectral_hat | 2.085e-16 | 0.1139 | -94.1590 |
| california_housing | tabicl_v2 | local_hat_vs_spectral_hat | 2.085e-16 | 0.0988 | 1060.1953 |
| california_housing | tabm_d | local_hat_vs_spectral_hat | 2.085e-16 | 0.0519 | 74.5053 |
| california_housing | tabpfn_2_6 | local_hat_vs_spectral_hat | 2.085e-16 | 0.0828 | -604.4419 |
| cpu_small | catboost | local_hat_vs_spectral_hat | 2.289e-16 | 0.0579 | -0.2883 |
| cpu_small | controlled_mlp | local_hat_vs_spectral_hat | 2.289e-16 | 0.0549 | -0.0941 |
| cpu_small | tabicl_v2 | local_hat_vs_spectral_hat | 2.289e-16 | 0.0620 | -0.1384 |
| cpu_small | tabm_d | local_hat_vs_spectral_hat | 2.289e-16 | 0.0731 | -0.3847 |
| cpu_small | tabpfn_2_6 | local_hat_vs_spectral_hat | 2.289e-16 | 0.0423 | -0.1033 |
| credit-g | catboost | local_hat_vs_spectral_hat | 2.204e-16 | 0.0849 | 0.0055 |
| credit-g | controlled_mlp | local_hat_vs_spectral_hat | 2.204e-16 | 0.0172 | 0.0033 |
| credit-g | tabicl_v2 | local_hat_vs_spectral_hat | 2.204e-16 | 0.0654 | 0.0068 |
| credit-g | tabm_d | local_hat_vs_spectral_hat | 2.204e-16 | 0.0636 | -0.0089 |
| credit-g | tabpfn_2_6 | local_hat_vs_spectral_hat | 2.204e-16 | 0.0321 | 0.0027 |
| diamonds | catboost | local_hat_vs_spectral_hat | 1.965e-16 | 0.0556 | 27.0047 |
| diamonds | controlled_mlp | local_hat_vs_spectral_hat | 1.965e-16 | 0.0256 | 4.0255 |
| diamonds | tabm_d | local_hat_vs_spectral_hat | 1.965e-16 | 0.0214 | 6.1714 |
| house_16H | catboost | local_hat_vs_spectral_hat | 2.075e-16 | 0.1135 | -296.9560 |
| house_16H | controlled_mlp | local_hat_vs_spectral_hat | 2.075e-16 | 0.0568 | -5.3872 |
| house_16H | tabicl_v2 | local_hat_vs_spectral_hat | 2.075e-16 | 0.1489 | 1261.0788 |
| house_16H | tabm_d | local_hat_vs_spectral_hat | 2.075e-16 | 0.1042 | 76.9623 |
| house_16H | tabpfn_2_6 | local_hat_vs_spectral_hat | 2.075e-16 | 0.0699 | -214.6349 |
| pendigits | catboost | local_hat_vs_spectral_hat | 2.201e-16 | 0.0208 | -0.0009 |
| pendigits | controlled_mlp | local_hat_vs_spectral_hat | 2.201e-16 | 0.0124 | -0.0033 |
| pendigits | tabicl_v2 | local_hat_vs_spectral_hat | 2.201e-16 | 0.0118 | -0.0032 |
| pendigits | tabm_d | local_hat_vs_spectral_hat | 2.201e-16 | 0.0160 | 0.0188 |
| pendigits | tabpfn_2_6 | local_hat_vs_spectral_hat | 2.201e-16 | 0.0117 | -0.0029 |
| phoneme | catboost | local_hat_vs_spectral_hat | 1.957e-16 | 0.0619 | -0.0075 |
| phoneme | controlled_mlp | local_hat_vs_spectral_hat | 1.957e-16 | 0.0567 | 0.0094 |
| phoneme | tabicl_v2 | local_hat_vs_spectral_hat | 1.957e-16 | 0.0650 | -0.0178 |
| phoneme | tabm_d | local_hat_vs_spectral_hat | 1.957e-16 | 0.0469 | -0.0040 |
| phoneme | tabpfn_2_6 | local_hat_vs_spectral_hat | 1.957e-16 | 0.0722 | -0.0036 |
| wine-quality-red | catboost | local_hat_vs_spectral_hat | 2.003e-16 | 0.1084 | 0.0020 |
| wine-quality-red | controlled_mlp | local_hat_vs_spectral_hat | 2.003e-16 | 0.1049 | 0.0033 |
| wine-quality-red | tabicl_v2 | local_hat_vs_spectral_hat | 2.003e-16 | 0.1224 | 0.0044 |
| wine-quality-red | tabm_d | local_hat_vs_spectral_hat | 2.003e-16 | 0.1236 | 0.0039 |
| wine-quality-red | tabpfn_2_6 | local_hat_vs_spectral_hat | 2.003e-16 | 0.1325 | 0.0001 |

### Fourier-origin changes

| dataset | model | basis pair | reconstruction error | disagreement | performance delta |
| --- | --- | --- | --- | --- | --- |
| bike-sharing | catboost | fourier_origin_vs_fourier_shift | 2.077e-15 | 0.0806 | 0.4516 |
| bike-sharing | controlled_mlp | fourier_origin_vs_fourier_shift | 2.077e-15 | 0.0792 | 0.7666 |
| bike-sharing | tabicl_v2 | fourier_origin_vs_fourier_shift | 2.077e-15 | 0.0984 | -1.7034 |
| bike-sharing | tabm_d | fourier_origin_vs_fourier_shift | 2.077e-15 | 0.1180 | -3.2178 |
| bike-sharing | tabpfn_2_6 | fourier_origin_vs_fourier_shift | 2.077e-15 | 0.0540 | -0.4889 |

### Other valid natural basis pairs

No additional valid pair was run: the optional polynomial pair was dropped under the frozen compute-priority rule. Diamonds C3 was excluded in six early foundation-model bundles because the initially selected feature had duplicate hat knots; all exclusions are preserved in bundle metadata. The deterministic feature scan used in later bundles is documented in `results/IMPLEMENTATION_REPAIRS.md`.

## 4. Mechanism: Initialization and Optimizer

| dataset | optimizer | initialization | condition | epoch0 disagreement | final disagreement | task metric |
| --- | --- | --- | --- | --- | --- | --- |
| bike-sharing | adamw | matched | matched_adamw | 1.411e-08 | 0.0617 | 75.8950 |
| bike-sharing | adamw | matched | matched_adamw_no_weight_decay | 1.411e-08 | 0.0617 | 75.8952 |
| bike-sharing | sgd | matched | matched_sgd_momentum | 1.411e-08 | 3.033e-08 | 177.2570 |
| bike-sharing | sgd | matched | matched_sgd_plain | 1.411e-08 | 1.847e-08 | 183.2250 |
| bike-sharing | adamw | ordinary | ordinary_adamw | 0.0069 | 0.1265 | 75.8950 |
| credit-g | adamw | matched | matched_adamw | 5.338e-09 | 0.1066 | 2.7330 |
| credit-g | adamw | matched | matched_adamw_no_weight_decay | 5.338e-09 | 0.1045 | 2.7218 |
| credit-g | sgd | matched | matched_sgd_momentum | 5.338e-09 | 1.006e-08 | 0.6116 |
| credit-g | sgd | matched | matched_sgd_plain | 5.338e-09 | 6.428e-09 | 0.6679 |
| credit-g | adamw | ordinary | ordinary_adamw | 0.0019 | 0.1846 | 2.7330 |
| house_16H | adamw | matched | matched_adamw | 1.698e-08 | 0.1534 | 34870.0161 |
| house_16H | adamw | matched | matched_adamw_no_weight_decay | 1.698e-08 | 0.1534 | 34869.9506 |
| house_16H | sgd | matched | matched_sgd_momentum | 1.698e-08 | 3.246e-08 | 49188.2754 |
| house_16H | sgd | matched | matched_sgd_plain | 1.698e-08 | 2.073e-08 | 50293.1921 |
| house_16H | adamw | ordinary | ordinary_adamw | 0.0104 | 0.2283 | 34870.0161 |
| phoneme | adamw | matched | matched_adamw | 5.493e-09 | 0.0366 | 0.3777 |
| phoneme | adamw | matched | matched_adamw_no_weight_decay | 5.493e-09 | 0.0366 | 0.3778 |
| phoneme | sgd | matched | matched_sgd_momentum | 5.493e-09 | 1.085e-08 | 0.5975 |
| phoneme | sgd | matched | matched_sgd_plain | 5.493e-09 | 8.033e-09 | 0.6672 |
| phoneme | adamw | ordinary | ordinary_adamw | 0.0041 | 0.0527 | 0.3777 |

- Does function matching eliminate the initial difference? Yes: the maximum matched initial-logit difference was below 1.9e-8, and median epoch-0 disagreement was below 1e-8.
- Does SGD preserve equivalence better than AdamW? Yes: both plain and momentum SGD remained near 1e-8 at the final checkpoint.
- Does AdamW reintroduce coordinate dependence? Yes: matched AdamW rose from numerical zero to substantial disagreement.
- What role does weight decay play? Little here: removing weight decay barely changed matched-AdamW disagreement, implicating adaptive coordinate-wise scaling rather than the decay term.

## 5. Equal-HPO Control

| dataset | problem_type | model | disagreement | disagreement_seed_sd | original task | transformed task | seeds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| credit-g | classification | controlled_mlp | 0.0335 | 0.0108 | 0.4919 | 0.4940 | 3 |
| credit-g | classification | tabm_d | 0.1566 | 0.0933 | 0.4874 | 0.5674 | 3 |
| house_16H | regression | controlled_mlp | 0.2487 | 0.0866 | 36143.4474 | 36212.4549 | 3 |
| house_16H | regression | tabm_d | 0.2277 | 0.0062 | 32563.0670 | 33457.6473 | 3 |
| phoneme | classification | controlled_mlp | 0.0692 | 0.0121 | 0.3650 | 0.3685 | 3 |
| phoneme | classification | tabm_d | 0.0815 | 0.0009 | 0.3234 | 0.3347 | 3 |

Each representation received the same independent nine-trial validation-only budget. Material disagreement remained in all six comparisons, so unequal or obviously mismatched tuning does not explain the effect.

## 6. Non-Oracle Repairs

| method | median disagreement | median reduction | task metric | median relative task change | mean rank | W/T/L | units |
| --- | --- | --- | --- | --- | --- | --- | --- |
| orbit_consistency_lambda_1 | 0.4370 | -343.01% | log loss / RMSE | +2.81% | 6.0000 | 0/0/11 | 11 |
| pca_canonical | 0 | +100.00% | log loss / RMSE | +2.90% | 1.4182 | 54/0/1 | 55 |
| raw | 0.0767 | +0.00% | log loss / RMSE | +0.00% | 3.4182 | 0/55/0 | 55 |
| standardization | 0.0870 | -3.10% | log loss / RMSE | +0.39% | 4.0727 | 15/0/40 | 55 |
| whitening | 0.0939 | -17.85% | log loss / RMSE | +6.82% | 4.4727 | 9/0/46 | 55 |

AnchorCanonical was excluded from predictive repair runs because every development audit orbit contained at least one rank-deficient RBF block. The dual-view refinement and optional polynomial basis were dropped before prospective access under the protocol's compute-priority rule. Oracle diagnostic (reported separately):

| method | median disagreement | median reduction | task metric | median relative task change | mean rank | W/T/L | units |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ORACLE INVERSE — NOT A METHOD | 0 | +100.00% | log loss / RMSE | +0.00% | 1.6182 | 55/0/0 | 55 |

## 7. AnchorCanonical Audit

| variant | comparisons | full_rank_rate | median_coordinate_error | max_coordinate_error | pass_rate |
| --- | --- | --- | --- | --- | --- |
| condition_le_3_all | 88 | +0.00% | 7.256e-05 | 1.0000 | +28.41% |
| orthogonal_all | 88 | +0.00% | 1.146e-10 | 0.0540 | +77.27% |
| orthogonal_one | 88 | +0.00% | 5.240e-17 | 3.068e-14 | +100.00% |

With 16 anchors, only 39/94 individual feature blocks were full rank; increasing to 256 anchors reached only 62/94. Accordingly AnchorCanonical is a documented failed candidate, not a valid general repair. Orthogonal one-block cases pass because untouched blocks dominate and the selected transformed block can be recoverable; all-block and condition≤3 failures reveal the rank problem.

## 8. Is Basis Sensitivity Sometimes Helpful?

| model | oracle_best_median_gain | validation_selected_median_gain | oracle_win_rate | validation_selected_win_rate |
| --- | --- | --- | --- | --- |
| catboost | +0.36% | -1.64% | +57.58% | +27.27% |
| controlled_mlp | +2.15% | -0.18% | +84.85% | +45.45% |
| tabicl_v2 | +0.02% | -0.62% | +54.55% | +36.36% |
| tabm_d | +1.51% | -1.74% | +63.64% | +27.27% |
| tabpfn_2_6 | -0.13% | -1.96% | +42.42% | +30.30% |

For all-block orthogonal bases, the oracle-best basis improved median task error by 0.83%, but validation selection transferred with a median -1.23% gain (negative means degradation) and won in only 33.3% of units. Favorable bases exist, but selecting them reliably is unresolved.

## 9. Prospective Holdout

The seven datasets below were never used during development or method selection. The method and SHA256 above were frozen before `results/raw/prospective/RUN_STARTED.json` was created.

| dataset | model | raw disagreement | proposed disagreement | reduction | task raw | task proposed | relative task change |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MagicTelescope | catboost | 0.0677 | 0 | +100.00% | 0.3523 | 0.3716 | +5.49% |
| MagicTelescope | tabicl_v2 | 0.0915 | 0 | +100.00% | 0.3298 | 0.3724 | +12.93% |
| MagicTelescope | tabm_d | 0.1198 | 0 | +100.00% | 0.3599 | 0.3907 | +8.57% |
| MagicTelescope | tabpfn_2_6 | 0.0729 | 0.0057 | +92.18% | 0.3318 | 0.3613 | +8.91% |
| elevators | catboost | 0.1727 | 0.0094 | +94.42% | 0.0032 | 0.0038 | +18.16% |
| elevators | tabicl_v2 | 0.3396 | 0 | +100.00% | 0.0031 | 0.0039 | +24.39% |
| elevators | tabm_d | 0.2607 | 0 | +100.00% | 0.0028 | 0.0037 | +34.34% |
| elevators | tabpfn_2_6 | 0.2913 | 0.0061 | +97.92% | 0.0020 | 0.0028 | +43.57% |
| houses | catboost | 0.1270 | 0 | +100.00% | 62552.1287 | 62706.9783 | +0.25% |
| houses | tabicl_v2 | 0.1724 | 0 | +100.00% | 59926.5263 | 61294.4675 | +2.28% |
| houses | tabm_d | 0.1722 | 0 | +100.00% | 61030.5101 | 65323.0642 | +7.04% |
| houses | tabpfn_2_6 | 0.1770 | 0.0090 | +94.92% | 53741.3186 | 59296.6228 | +10.35% |
| jannis | catboost | 0.0539 | 0.0346 | +35.78% | 0.8655 | 0.9127 | +5.45% |
| jannis | tabicl_v2 | 0.0776 | 0.0004 | +99.43% | 0.8411 | 0.8993 | +6.92% |
| jannis | tabm_d | 0.0880 | 0.0024 | +97.07% | 0.9220 | 0.8999 | -2.07% |
| jannis | tabpfn_2_6 | 0.0694 | 0.0194 | +72.07% | 0.8220 | 0.8494 | +3.33% |
| optdigits | catboost | 0.0190 | 0.0200 | -5.09% | 0.2158 | 0.2567 | +18.99% |
| optdigits | tabicl_v2 | 0.0350 | 0 | +100.00% | 0.0390 | 0.0590 | +51.10% |
| optdigits | tabm_d | 0.0297 | 0 | +100.00% | 0.0672 | 0.1388 | +106.65% |
| optdigits | tabpfn_2_6 | 0.0269 | 0.0251 | +4.87% | 0.0589 | 0.0818 | +39.44% |
| pol | catboost | 0.0825 | 0.0846 | -2.98% | 9.0887 | 11.0769 | +21.94% |
| pol | tabicl_v2 | 0.0750 | 0 | +100.00% | 5.8185 | 7.9949 | +37.41% |
| pol | tabm_d | 0.1299 | 0 | +100.00% | 8.5849 | 12.0948 | +40.95% |
| pol | tabpfn_2_6 | 0.1236 | 0.1222 | +1.04% | 8.2406 | 9.5725 | +16.15% |
| segment | catboost | 0.0201 | 0 | +100.00% | 0.1269 | 0.1615 | +27.29% |
| segment | tabicl_v2 | 0.0213 | 0 | +100.00% | 0.0612 | 0.0855 | +39.65% |
| segment | tabm_d | 0.0337 | 0 | +100.00% | 0.1103 | 0.1166 | +7.93% |
| segment | tabpfn_2_6 | 0.0255 | 0.0005 | +97.83% | 0.0540 | 0.0647 | +19.94% |

## 10. Development vs Prospective Summary

| panel | method | units | median disagreement | median reduction | 95% dataset bootstrap CI (reduction) | median relative task change | W/T/L |
| --- | --- | --- | --- | --- | --- | --- | --- |
| development | ORACLE INVERSE — NOT A METHOD | 55 | 0 | +100.00% | [1.0000, 1.0000] | +0.00% | 55/0/0 |
| development | orbit_consistency_lambda_1 | 11 | 0.4370 | -343.01% | [-8.4253, -2.7019] | +2.81% | 0/0/11 |
| development | pca_canonical | 55 | 0 | +100.00% | [1.0000, 1.0000] | +2.90% | 54/0/1 |
| development | raw | 55 | 0.0767 | +0.00% | [0, 0] | +0.00% | 0/55/0 |
| development | standardization | 55 | 0.0870 | -3.10% | [-0.0720, -0.0077] | +0.39% | 15/0/40 |
| development | whitening | 55 | 0.0939 | -17.85% | [-0.3144, -0.1226] | +6.82% | 9/0/46 |
| prospective (untouched) | ORACLE INVERSE — NOT A METHOD | 28 | 0.0014 | +92.89% | [0.8663, 0.9289] | +0.00% | 28/0/0 |
| prospective (untouched) | pca_canonical | 28 | 0 | +100.00% | [0.9707, 1.0000] | +17.16% | 26/0/2 |
| prospective (untouched) | raw | 28 | 0.0800 | +0.00% | [0, 0] | +0.00% | 0/28/0 |
| prospective (untouched) | standardization | 28 | 0.0799 | +0.13% | [-0.0066, 0.0038] | +0.44% | 15/0/13 |
| prospective (untouched) | whitening | 28 | 0.1065 | -20.21% | [-0.4884, -0.0749] | +26.01% | 4/0/24 |

Primary statistical unit is dataset×model; orbit members are averaged within units. Bootstrap intervals resample datasets, and W/T/L compares each repair with raw disagreement using a 1e-12 tie tolerance.

## 11. Strongest Evidence FOR the Hypothesis

The strongest evidence is the conjunction of (i) broad condition-number-one effects across frozen transformers, neural tabular models, an MLP, and CatBoost; (ii) nontrivial effects under recognizable natural basis pairs with reconstruction errors near machine precision; (iii) persistence after equal HPO; and (iv) a controlled mechanism in which exact function matching plus SGD preserves equivalence while AdamW reconstructs basis dependence.

## 12. Strongest Evidence AGAINST the Hypothesis

The strongest counterevidence is that a canonical basis can change task performance, oracle-best basis choices sometimes help, and PCA does not canonicalize general condition≤3 maps. Some residual disagreement remains even for oracle/PCA inputs in stochastic or threshold-sensitive learners, and AnchorCanonical fails because practical feature blocks are frequently rank deficient. These facts argue against treating invariance as unconditionally desirable or solved.

## 13. Reviewer Attack Audit

### "Random rotations are artificial."

Yes, but natural one-hot/Helmert, local/spectral spline, and Fourier-origin coordinates reproduce the phenomenon.

### "This is just poor numerical conditioning."

No. Orthogonal transforms have condition number one and already produce broad effects; every general transform also satisfied condition≤3.

### "You used the wrong hyperparameters."

Equal independent nine-trial validation-only HPO leaves disagreement in every tested pair.

### "This is only optimization noise."

Frozen TabICL/TabPFN models also change, and controlled function-matching experiments separate optimizer geometry from initial function differences.

### "Of course inverse canonicalization works."

PCA uses no knowledge of Q and achieved 100.0% development and 100.0% prospective median reduction, though its task cost and restriction to orthogonal changes prevent claiming a complete solution.

### "Maybe basis choice is useful rather than nuisance."

Oracle-best bases sometimes improve task error, but validation-selected choices usually fail to transfer. The appropriate target is controlled or learnable basis handling, not blanket invariance.

### "The method was tuned to these datasets."

The PCA rule was frozen with SHA256 `2a6378522943f2cb767de199ab4df4e1a1797e750d951d11b8e6ce2ba0131329` before the seven prospective datasets were accessed; its prospective reduction was 100.0% with +17.2% median task-error change.

## 14. ICLR/ICML/NeurIPS Assessment

- novelty assessment: strong empirical identification of a hidden within-feature basis prior, with natural-basis and optimizer-geometry evidence.
- empirical strength: high; 18 real datasets, five development model families, exact equivalence audits, equal HPO, and a locked seven-dataset holdout.
- method strength: partial; PCA is a strong orthogonal canonicalizer but not consistently performance-neutral or general-invertible.
- biggest remaining weakness: no non-oracle method simultaneously preserves useful raw-coordinate inductive bias, handles rank deficiency/general invertible maps, and stays within 1% task cost.
- estimated paper direction: a phenomenon-and-mechanism paper is credible; a top-tier full paper needs a stronger controlled/learnable interface.

## 15. Recommended Next Step

Build the missing rank-robust, target-free or validation-controlled dual-view method. It must exceed 70% dataset-level median disagreement reduction for both orthogonal and condition≤3 changes, keep median relative task degradation at or below 1%, preserve any genuinely helpful basis signal, and pass a newly locked external panel. Focus first on degeneracy-aware subspace canonicalization plus a tightly regularized raw branch; do not tune on the completed prospective panel.

## 16. Files Produced

- `results.md` (this report)
- `configs/FROZEN_METHOD_CONFIG.json` (`2a6378522943f2cb767de199ab4df4e1a1797e750d951d11b8e6ce2ba0131329`)
- `results/integrity_audit.json` (pass; 870 audited bundles)
- `results/processed/` (replication, natural-basis, mechanism, HPO, repair, prospective, and bootstrap tables)
- `results/raw/development/` and `results/raw/prospective/` (immutable hashed bundles)
- `figures/figure_01_*.{png,pdf}` through `figures/figure_08_*.{png,pdf}`
- `results/IMPLEMENTATION_REPAIRS.md`, `results/PROSPECTIVE_LOCK.json`, and run markers
- `src/`, `scripts/`, and `tests/` (implementation, runners, analysis, plots, and integrity tests)
