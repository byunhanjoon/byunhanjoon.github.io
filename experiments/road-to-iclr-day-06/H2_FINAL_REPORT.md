# H2 final report — Precision-Delay Law

## Verdict

**FALSIFIED AS WRITTEN**

All 18 frozen bundles completed (3 datasets × 3 models × 2 seeds; 216 trained
paths and 144 nonidentity comparisons).  Two boundary gates pass:

- float64 interface accumulation is exactly closed in 9/9 cells;
- fp32 MLP/ResNet remains below the stable threshold in 6/6 cells.

The quantitative law fails both primary gates.  No FT dataset has the frozen
ordering `bfloat16 <= float16 <= float32 < float64`.  Pooled median hitting
epochs are:

| Interface format | Median epoch to MSE 1e-5 |
|---|---:|
| bfloat16 | 10.0 |
| float16 | 8.5 |
| float32 | 12.0 |
| float64 | >30 (censored at 31) |

The resulting four-point Spearman coefficient is numerically 0.80 but does not
rescue the predeclared exact ordering gate.  In every FT dataset, float16
separates before the nominally coarser bfloat16.

## What changed scientifically

Nominal unit roundoff is not the effective perturbation amplitude.  Operand
quantization, accumulator behavior, output casting, and quantization-cell
coalescence jointly determine whether two conjugate dot products land on the
same representable value.  In the first H2 bundle, bfloat16 schema paths are
identical at epoch zero whereas float16 paths already differ, directly
contradicting the simple `C u exp(lambda t)` initialization assumed by H2.

A post-outcome diagnostic using discrepancy after the first update is not
strong enough to justify immediate confirmation: across FT paths its Spearman
association with hitting time is only -0.34 at epoch 1 (rising in magnitude
only when measured close to the hitting event).  Therefore no H2b confirmation
is launched.  H1 remains confirmed; the precision-scaling embellishment is
discarded.

## Retained idea

The bfloat16/float16 reversal suggests a separate future theory of
**quantization coalescence**—coarser cells can sometimes erase semantic
reduction-order differences even while injecting more ordinary approximation
error.  Current evidence is a mechanism clue, not a validated performance
method or paper claim.
