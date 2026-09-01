# Frozen protocol: four-pack cross-score variance identity

Estimate the 128-cell weight covariance of the fixed graph four-pack sampler
from 65,536 fresh packs in eight independent chunks. For each of the 23 stored
full-product candidates, use its complete validation prediction tensor to
evaluate the exact independent-cross formula

`Var(score) = 2 <r,C r> + tr(C^2)`

with the validation-example normalization included. Compare this operator
prediction with the empirical variance of the 512 frozen pack-cross128 score
draws. Operator uncertainty is the standard error across eight chunks;
empirical variance uncertainty uses the normal-reference
`sqrt(2/(511))*variance` scale. Also report panel geometric predicted/observed
ratios and the relative sizes of the linear and quadratic terms.

The calibration gate passes if at least 20/23 candidates differ by at most
2.58 combined standard errors and every represented panel geometric ratio lies
in `[0.85,1.15]`. This diagnoses the variance mechanism; it is not new source
evidence.
