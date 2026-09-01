# Model-selection tail-risk analysis

Status: frozen before quantile computation on 2026-08-28.

Mean held-out loss can conceal unlucky nuisance-design realizations. Reuse the
1,024 stored draws from the frozen model-selection experiment and compute,
within every dataset/method, the standard deviation, 90th percentile, 95th
percentile, and conditional mean above the 95th percentile of realized
held-out proper loss. No rows or representatives are treated as independent.

The tail gate requires strength-2 to have lower equal-dataset mean 95th-
percentile loss than all three controls, and lower dataset-level 95th
percentile than IID in at least 60% of datasets, in every one of the three
panels. This is a conditional post-training analysis.

