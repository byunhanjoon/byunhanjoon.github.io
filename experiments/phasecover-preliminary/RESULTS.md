# PHASECOVER PRELIMINARY RESULTS

## Verdict

**MECHANISM PRELIMINARILY SUPPORTED; FORECAST-GAIN CLAIM NOT SUPPORTED.** The frozen 64-draw screen passed all gates, but exact post-hoc enumeration
shows that its marginal forecast gate was sampling-sensitive. The result applies only to one compact
phase-augmented patch Transformer and is not a forecasting SOTA claim.

## Frozen gates

| Gate | Result | Requirement | Verdict |
|---|---:|---:|---:|
| phase materiality | 2/3 datasets | >=2 | PASS |
| full16 vs canonical | 3/3; mean gain +0.02354 | >=2 and positive | PASS |
| PhaseCover quotient vs IID4 | 3/3; mean ratio 0.435 | >=2 and <=0.80 | PASS |
| PhaseCover forecast vs IID4 | 2/3; mean gain -0.00109 | >=2 | PASS |

## Dataset means over three model seeds

| Dataset | phase materiality | canonical | IID4 | PhaseCover4 | full16 | cover/IID quotient | cover−IID forecast |
|---|---:|---:|---:|---:|---:|---:|---:|
| ETTh1 | 22.9% | 0.32666 | 0.29017 | 0.29412 | 0.28876 | 0.463 | +0.00395 |
| Exchange | 2.0% | 1.10761 | 1.10388 | 1.10378 | 1.10335 | 0.557 | -0.00010 |
| Solar | 29.2% | 0.39205 | 0.36762 | 0.36703 | 0.36359 | 0.284 | -0.00059 |

`cover−IID forecast` is PhaseCover4 RMSE minus expected IID4 RMSE, so negative is better.

## Exhaustive post-hoc robustness check

This diagnostic enumerates all `C(16,4)=1,820` four-phase subsets. It does not alter the frozen gates.

| Dataset | exact IID4 | PhaseCover4 | cover−exact IID | quotient ratio | quotient percentile |
|---|---:|---:|---:|---:|---:|
| ETTh1 | 0.29062 | 0.29412 | +0.00350 | 0.484 | 88.8% |
| Exchange | 1.10340 | 1.10378 | +0.00038 | 0.554 | 69.2% |
| Solar | 0.36719 | 0.36703 | -0.00016 | 0.301 | 92.9% |

The quotient advantage survives on 3/3 datasets (dataset-balanced ratio 0.446). The forecast comparison wins only 1/3 and is +0.00124 RMSE worse on average.

## Interpretation

- ETTh1 and Solar exhibit material patch-origin sensitivity; Exchange is a useful near-null control.
- Full phase averaging improves the canonical representation on the frozen dataset-level criterion.
- Four equally spaced phases estimate the all-phase quotient substantially better than expected IID4.
- Exact enumeration rejects a reliable forecast-gain claim. The clean result is efficient quotient
  estimation, not demonstrated universal accuracy improvement.

## Integrity and compute

- Protocol SHA-256: `ba5bade9069dbb9a02e00450044a26d69efd719d9dd3ee3bafd96f09526d1f44` (matched: True).
- Exact context reconstruction maximum error: 0.0.
- Cells: 9/9; method rows: 36; IID design rows: 576; phase rows: 144.
- Summed fit time: 72.1 seconds; mean epochs: 13.8.

## What would falsify the paper direction next

The next study must use frozen published implementations (at least PatchTST and one pretrained TSFM),
include canonical-trained and phase-augmented training controls, and repeat on untouched datasets.
Kill the direction if materiality or quotient efficiency does not transfer. Do not tune offsets per dataset.
Channel permutation, adaptive patch size, and TabPFN-on-lags are not claimed as new.

## Readiness

Novelty potential: **3.5/5**. Empirical readiness: **1.5/5**. Status: **promising preliminary
mechanism, failed forecast-gain robustness check; not yet an ICLR result.**
