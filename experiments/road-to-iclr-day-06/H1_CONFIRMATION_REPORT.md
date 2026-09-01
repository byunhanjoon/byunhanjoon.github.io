# H1 confirmation report — Semantic Arithmetic Amplification

## Verdict

**CONFIRMED, WITH ARCHITECTURE-SPECIFIC PRACTICAL MAGNITUDE**

The prospectively fixed six-seed confirmation completed all 54
dataset×model×seed bundles after the two-seed pilot.  Across all eight seeds,
the study contains 72 bundles and 576 trained paths.

Interface Exact Accumulation (IEA64) makes all three nonidentity schema paths'
aligned predictions bitwise identical to their canonical path at every
recorded checkpoint in all 9/9 dataset×model cells.  A separate deterministic
one-update audit verifies exact conjugacy of parameters and AdamW moments for
all three architectures.  The confirmation criterion passes: 9/9 cell-level
seed means improve, and the dataset-block bootstrap interval for the log risk
ratio is strictly below zero.  Because every numerator is exactly zero, the
finite log interval depends on the declared numerical floor; the exact-zero
count is the more informative evidence.

## Magnitude at epoch 20

| Dataset | Model | FP32 schema-orbit MSE | IEA64 MSE | Seeds |
|---|---|---:|---:|---:|
| Bank marketing | FT-Transformer | 4.096e-3 | 0 | 8 |
| Credit default | FT-Transformer | 9.600e-4 | 0 | 8 |
| FreMTPL claims | FT-Transformer | 7.837e-2 | 0 | 8 |
| Bank marketing | MLP | 4.496e-16 | 0 | 8 |
| Credit default | MLP | 7.275e-16 | 0 | 8 |
| FreMTPL claims | MLP | 4.400e-16 | 0 | 8 |
| Bank marketing | ResNet | 3.006e-14 | 0 | 8 |
| Credit default | ResNet | 4.816e-14 | 0 | 8 |
| FreMTPL claims | ResNet | 8.969e-13 | 0 | 8 |

The fp32 maximum amplification factors are `6.54e12`, `2.41e12`, and
`3.63e13` in the three FT cells.  MLP and ResNet stay near their injected
roundoff scale, establishing the required stable boundary.

## Interpretation

For this dense-stem FT-Transformer, a schema permutation and conjugate weight
permutation are identical in real arithmetic, use the same data order and
dropout tape, and differ at epoch zero by only float32 reduction order.  That
perturbation alone becomes a large prediction difference.  Performing only the
first affine accumulation in float64 removes the perturbation after its float32
cast.  The state-level update audit and exact checkpoint predictions support
pathwise commutation under the tested implementation; every intermediate state
of the long GPU runs was not stored.

This does **not** establish a predictive-accuracy improvement.  On canonical
FT paths, changing interface arithmetic selects a different chaotic trajectory:
across seeds the relative test-loss change can be positive or negative, while
its pooled mean is close to zero.  H1 supports exact semantic reproducibility
and a causal mechanism, not SOTA accuracy.

## Next decision

Advance to H2, the frozen precision-delay law.  H2 tests a stronger quantitative
prediction: if interface roundoff is an injected perturbation subsequently
amplified by common dynamics, the epoch at which schema paths separate should
shift approximately linearly with log unit roundoff across bfloat16, float16,
float32, and float64 interface arithmetic.  Failure of that ordered delay would
narrow H1 to an implementation-specific exactness trick rather than a dynamical
law.
