# PROSPECTIVE RESULT — COMPATIBILITY × CANDIDATE RELIABILITY

## Verdict

**KILL the candidate-wise reliability reranker as the next ICLR method.**  The intervention
reliably changes the diagnostic in the intended direction, but that change does not
reliably improve prediction and is not stronger than the permutation control.
The Retrieval Risk Law remains exact; the failed step was treating the mean of
one-neighbor risks as a sufficient diagnostic for a multi-neighbor aggregate.

## Frozen panel integrity

- Real: 12 datasets, 216 trained cells, 648 method rows.
- Synthetic: 4 tasks, 8 fresh seeds, 2 models, 32 aggregate rows.
- Every real cell has distance, true OOF-reliability, and permuted-reliability results.
- All lambda choices used validation loss only; no test label entered retrieval scoring.

## Real primary gates

| Model | Method | score gain | W/L/T | proxy-risk change | risk improves | nonzero lambda |
|---|---|---:|---:|---:|---:|---:|
| TabR | oof_reliability | 0.00033 | 7/4/1 | -0.01292 | 11/12 | 60.2% |
| TabR | permuted_reliability | 0.00059 | 6/6/0 | 0.01016 | 0/12 | 66.7% |
| ModernNCA | oof_reliability | 0.00033 | 4/6/2 | -0.02261 | 9/12 | 43.5% |
| ModernNCA | permuted_reliability | 0.00052 | 7/3/2 | 0.00439 | 1/12 | 52.8% |

True OOF reliability lowered proxy risk on 11/12 TabR and 9/12 ModernNCA
datasets, yet the prediction gates were only 7/12 and 4/12.  Its dataset-balanced
score gain was smaller than the permutation control for both models.  Risk reduction
versus score gain had Spearman rho 0.329 for TabR and 0.018 for ModernNCA.

## Per-dataset OOF-reliability result

| Dataset | TabR score gain | TabR risk change | ModernNCA score gain | ModernNCA risk change |
|---|---:|---:|---:|---:|
| Bike_Sharing_Demand | 0.00275 | -0.00456 | -0.00013 | -0.00186 |
| MagicTelescope | -0.00005 | -0.00479 | 0.00011 | -0.02176 |
| abalone | 0.00103 | -0.02914 | 0.00167 | -0.04393 |
| bank-marketing | 0.00060 | -0.04017 | 0.00000 | 0.00000 |
| covertype | 0.00000 | 0.00000 | -0.00022 | -0.04337 |
| cpu_act | 0.00082 | -0.00062 | 0.00057 | -0.00103 |
| credit-g | 0.00278 | -0.03523 | 0.00278 | -0.07756 |
| electricity | 0.00016 | -0.00422 | -0.00016 | -0.05309 |
| elevators | 0.00006 | -0.01903 | 0.00000 | 0.00000 |
| jannis | -0.00315 | -0.01143 | -0.00054 | -0.03099 |
| sulfur | -0.00053 | -0.00439 | -0.00007 | -0.00378 |
| superconduct | -0.00049 | -0.00142 | -0.00006 | 0.00602 |

## Synthetic S3 gate

| Model | Method | RMSE change | wins/8 | exact proxy-risk change | mean lambda |
|---|---|---:|---:|---:|---:|
| TabR | exact_reliability | -0.00071 | 2/8 | -0.06602 | 0.163 |
| TabR | estimated_reliability | -0.00130 | 3/8 | -0.06626 | 0.164 |
| TabR | permuted_exact_reliability | -0.00142 | 4/8 | 0.00402 | 0.092 |
| ModernNCA | exact_reliability | -0.00065 | 5/8 | -0.31549 | 0.591 |
| ModernNCA | estimated_reliability | 0.00007 | 5/8 | -0.29759 | 0.028 |
| ModernNCA | permuted_exact_reliability | 0.00017 | 2/8 | 0.00506 | 0.091 |

On S3, exact candidate variance sharply reduced the mean top-16 one-neighbor risk
but changed neural test RMSE by less than 0.001 on average.  Estimated variance did
not recover a clean ModernNCA gain, and TabR's permutation control was comparable.
The frozen synthetic gate therefore fails.

## Why the original diagnostic can fail

For normalized aggregation weights `w`, signed conditional-mean discrepancies `d`,
and candidate variances `sigma2`, the exact aggregate risk is

```text
R_aggregate = (sum_i w_i d_i)^2 + sum_i w_i^2 sigma2_i.
```

The weighted mean of one-neighbor risks used by the diagnostic is

```text
R_one_mean = sum_i w_i (d_i^2 + sigma2_i).
```

Their exact nonnegative gap is

```text
R_one_mean - R_aggregate
  = Var_w(d) + sum_i w_i(1-w_i)sigma2_i >= 0.
```

Thus a neighborhood can look much better under average candidate risk while losing
useful signed-bias cancellation or receiving little benefit because averaging already
dilutes candidate noise by squared weights.  This explains the observed diagnostic–
prediction decoupling and invalidates top-k mean one-neighbor risk as a standalone
mechanism certificate for TabR/ModernNCA.

## Epoch and compute diagnostic

| Model | cells | mean epochs | median | max | summed fit seconds |
|---|---:|---:|---:|---:|---:|
| ModernNCA | 108 | 25.48 | 24.5 | 48 | 220.3 |
| TabR | 108 | 18.04 | 18.0 | 44 | 246.5 |

Most compact models early-stopped well before 48 epochs, though some reached the cap.
This does not establish large-scale convergence behavior; it does show that the failed
reranking mechanism is not contingent on one extremely short training run.

## Novelty/readiness decision

- Arithmetic helices are occupied by 2025 work with causal interventions, and 2026
  work already analyzes carry fibers, layer transitions, and convergence-dependent
  sharpening. A reproduction/atlas alone is not a credible ICLR novelty claim.
- Generic uncertainty-aware or reliability-weighted neighbor retrieval is crowded.
- The exact aggregation gap above is useful and empirically exposed here, but it is
  bias-variance algebra rather than a sufficiently new theorem by itself.
- No new Day-8 embedding/retrieval method currently has both defensible novelty and
  strong prospective results. Status: **INTERESTING NEGATIVE MECHANISM, NOT ICLR-READY.**

## Post-hoc corrective outcome

The one allowed post-hoc experiment optimized the *aggregate* plug-in risk directly
over a frozen shortlist. It passed both real-data subgates, but mismatch-only weighting
matched the full estimator and ModernNCA failed the frozen S3 transfer gate. The joint
stop rule is therefore met; see `POSTHOC_AGGREGATION_RESULTS.md`.
