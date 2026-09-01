# Synthetic-to-real shrinkage transfer

Regression preserves the synthetic routing-strength curve; classification does not.

On the 9,600-episode synthetic test, classification gain peaks at lambda=0.7
(+0.006447), while all 20 synthetic regression cells prefer lambda=1 and the gain rises
monotonically to +0.254185. Across the initial real panels, regression also improves
monotonically to lambda=1 (+0.217965 across 16 datasets).

Real classification instead has a shallow inverted-U curve: its nine-dataset point
optimum is lambda=0.5 (+0.001165), then gain falls to -0.000266 at lambda=1. The smallest
steps, lambda=0.1 and 0.2, have descriptive dataset-bootstrap intervals barely above
zero. Unit optima are highly heterogeneous, so this retrospective curve does not itself
validate a lambda.

The transfer failure is therefore one of adaptation strength, not just the existence of
predictive competence information. Full regression movement transfers; classification
requires substantial shrinkage after leaving PriorDial. This motivated the separately
frozen independent lambda=0.1 confirmation.

Artifacts: `SHRINKAGE_TRANSFER_PROTOCOL.md`, processed unit/summary CSVs and audit under
`results/processed/shrinkage_transfer_*`, and `figures/shrinkage_transfer_v1.png`.
