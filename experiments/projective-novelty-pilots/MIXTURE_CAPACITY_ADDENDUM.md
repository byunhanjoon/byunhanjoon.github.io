# MIXTURE PILOT: CAPACITY-MATCHED CONTROL ADDENDUM

Status: frozen after the primary mixture gate passed but before these control
outcomes are inspected. This addendum does not alter the primary result.

The four-component projective mixture has 136,580 parameters, compared with
98,945 for the single projective Gaussian and 94,860 for the direct mixture.
Retrain two widened controls with the same data, broad query batches, seeds,
optimizer, and 3,000 steps:

1. `joint_gaussian_matched`, widened to approximately 136,000 parameters.
2. `direct_mixture4_matched`, widened to approximately 136,000 parameters.

The non-Gaussian result survives capacity matching if the original projective
mixture (i) beats the matched Gaussian in at least six of nine cells while
retaining NLL improvements of at least `0.05` on two datasets, and (ii) beats
the matched direct mixture in at least six of nine cells.
