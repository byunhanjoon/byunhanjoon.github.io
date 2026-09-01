# Frozen protocol: 32/64/128-fit unbiased packed-score frontier

Restrict all comparisons to the same 23 candidates with the full
`4 x 4 x 2 x 4` product. Assemble candidate RMSE for:

1. two independent 16-fit strength-2 covers crossed at 32 fits;
2. two independent disjoint-pair averages crossed at 64 fits; and
3. two independent mutually-disjoint four-pack averages crossed at 128 fits.

Every checkpoint is exactly unbiased for quotient Brier/MSE; only the
within-half dependence changes. Report candidate and panel RMSE ratios for
each doubling and the effective exponent `-log2(RMSE_2B/RMSE_B)`.

The frozen gate passes if both doublings strictly lower panel-mean RMSE in
every represented panel, each transition improves at least 20/23 candidates,
and the 128-fit checkpoint is best on at least 22/23. No exact-closure
candidate is included.
