# Direction 2 artifacts

- `schema_transfer_metrics.csv`: same-schema, held-schema, shuffled-semantics, and paired role-reversal metrics.
- `few_shot_metrics.csv`: 0/10/50/100-label target-schema results.
- `role_reversal_invariance.csv`: paired prediction-change audit.
- `robustness_classification.csv`: five seeds over three training sizes and three signal strengths.
- `robustness_regression_alternative.csv`: an alternative regression mechanism over three noise levels.

All target schemas and field names are held out. The semantic representation is the protocol's final-fallback manual source/destination role vector, not a learned text embedding; results therefore estimate the value of correct role semantics, not the ability to recover them from language.
