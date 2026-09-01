# Direction 1 artifacts

- `observational_equivalence_pairs.csv`: explicit linear-Gaussian SCM parameters and paired causal effects sharing the exact same observed sample.
- `ate_prediction_models.csv`: held-pair ATE prediction metrics for a constant, ridge, random forest, and small MLP.
- `identifiability_models.csv`: observational-only versus assumption-aware classification.
- `robustness_continuous.csv`: five seeds across dataset size, effect gap, and summary capacity.
- `robustness_binary_alternative.csv`: separate binary potential-outcome/confounding equivalence family with three noise levels.
- `robustness_assumption_shuffle.csv`: real versus permuted assumption metadata.

Pairs and observational-equivalence blocks are kept together in train/test splitting. The latent variable is never exposed to a learned model.
