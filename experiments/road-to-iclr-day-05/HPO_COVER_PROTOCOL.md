# Orthogonal cover after ordinary per-schema HPO

Status: adaptive post-HPO extension, frozen before cover outcomes on the HPO
tensor were computed.

For each dataset, family, schema representative, and model seed in the frozen
prospective HPO panel, select the minimum-validation-Brier candidate exactly
as ordinary per-schema HPO does. This yields a test prediction tensor over
`feature x category x class x seed` that includes the selection path.

Without refitting or changing candidates, apply the frozen budget-four
strength-1 and budget-16 strength-2 covers. Comparators and materiality are
identical to the fixed-recipe cover experiments. This is an end-to-end stress
test, not a new independent dataset confirmation.

