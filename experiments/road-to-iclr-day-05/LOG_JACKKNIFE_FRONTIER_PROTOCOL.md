# Four-block log-jackknife frontier protocol

Status: protocol frozen after the 32-fit log-jackknife result and before
inspecting 64-fit outcomes; analysis complete.

## Design

On the same six classification panels, draw four independent 16-fit
strength-2 blocks or four independent IID-16 blocks. For `K=4`, score each
candidate by

`L_J,4 = [4 L(mean_k Q_hat_k) - mean_k L(Q_hat_k)] / 3`.

Use 512 deterministic actions (64 fits per candidate). Compare cover versus
IID RMSE and exact log-quotient validation regret, and compare cover RMSE to
the earlier 32-fit two-block jackknife. Report mean absolute bias separately.

## Frozen interpretation

- **Frontier pass:** cover beats IID in RMSE and has no higher regret on at
  least 5/6 panels, and cover RMSE is no higher from 32 to 64 fits on at least
  5/6.
- **Qualified:** cover beats IID but the 32-to-64 frontier clause fails.
- **Fail:** a cover-versus-IID clause fails.

No finite-sample unbiasedness is claimed.

## Outcome

The frozen interpretation is **frontier pass**, with every clause passing on
all 6/6 panels.

- Cover-64 RMSE is 29--32% below cover-32 across panels and lower than IID-64
  everywhere.
- Every source has lower cover-64 RMSE than IID-64 and lower cover-64 than
  cover-32. All twelve equal-source RMSE intervals exclude zero favorably.
- Validation regret is lower or tied on every panel, but source intervals
  usually touch zero because most datasets are at exact-selection ceiling.
- Mean absolute bias is not monotone from 32 to 64 fits in every panel, as the
  approximate theory allows. Multiclass bias is higher at 64 fits despite its
  substantially lower RMSE.

The result supports a compute/RMSE frontier for the approximate nonlinear
extension, not exact unbiasedness or monotone bias.
