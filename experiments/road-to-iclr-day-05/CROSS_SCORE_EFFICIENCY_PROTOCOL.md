# Cross-score estimation-efficiency diagnostic

Status: frozen before candidate-level score-RMSE comparisons (the primary
selection gate and score means were already observed).

For every candidate cell, estimate selection-score RMSE around the exact full
quotient validation loss as

`sqrt(sample_variance + squared_sample_bias)`,

where `sample_variance = draws * MC_standard_error^2` from the 1,024 randomized
actions.  Compare the unbiased strength-2 cross-score with the 32-member IID
U-statistic, panel by panel and candidate by candidate.  Also show the ordinary
strength-2 mean score to expose the bias/variance tradeoff rather than assuming
unbiasedness is automatically preferable.

The diagnostic gate requires lower mean RMSE than IID-U in at least four of
five panels and in more than 60% of all candidate cells.  Source-bootstrap the
dataset-averaged paired RMSE differences within each panel.  This is a
post-primary mechanism diagnostic, not independent confirmation.
