# Table 10 — Efficiency and scalability

| Scope | Method | Dimension | Parameters | Representation bytes | Precompute/fit seconds | Peak GPU bytes |
|---|---|---|---|---|---|---|
| 10,000 states | full_similarity | 6000.0000 | — | 2.400e+08 | 1.5708 | 0.0000 |
| 10,000 states | lookup_unknown | 6001.0000 | — | 1.200e+05 | 0.0000 | 0.0000 |
| 10,000 states | mpe_dense | 32.0000 | — | 1.280e+06 | 0.3231 | 0.0000 |
| 10,000 states | mpe_sparse | 32.0000 | — | 1.201e+05 | 0.3396 | 0.0000 |
| 10,000 states | nystrom | 32.0000 | — | 2.560e+06 | 0.3243 | 0.0000 |
| real ridge mean | mpe | 32.0000 | 32.0000 | — | 6.7966 | 0.0000 |
| real ridge mean | nystrom | 32.0000 | 32.0000 | — | 6.5515 | 0.0000 |
| real ridge mean | similarity_unnormalized | 32.0000 | 32.0000 | — | 7.2174 | 0.0000 |
| real ridge mean | unknown_embedding | 204.1111 | 204.1111 | — | 1.0679 | 0.0000 |
| neural mean | ft_transformer / character_3gram_hash | 128.0000 | 2.740e+05 | — | 7.5841 | 2.035e+08 |
| neural mean | ft_transformer / mpe | 32.0000 | 2.398e+05 | — | 470.5212 | 6.822e+08 |
| neural mean | ft_transformer / mpe_corrupt_0 | 32.0000 | 2.563e+05 | — | 17.3048 | 1.817e+08 |
| neural mean | ft_transformer / mpe_corrupt_1 | 32.0000 | 2.347e+05 | — | 12.2492 | 1.956e+08 |
| neural mean | ft_transformer / mpe_corrupt_2 | 32.0000 | 2.128e+05 | — | 9.7116 | 2.510e+08 |
| neural mean | ft_transformer / mpe_corrupt_3 | 32.0000 | 2.198e+05 | — | 9.0590 | 2.037e+08 |
| neural mean | ft_transformer / mpe_corrupt_4 | 32.0000 | 1.868e+05 | — | 11.4396 | 2.400e+08 |
| neural mean | ft_transformer / mpe_corrupt_5 | 32.0000 | 2.689e+05 | — | 9.9785 | 2.489e+08 |
| neural mean | ft_transformer / mpe_corrupt_6 | 32.0000 | 2.779e+05 | — | 13.1776 | 2.066e+08 |
| neural mean | ft_transformer / mpe_corrupt_7 | 32.0000 | 2.368e+05 | — | 6.4649 | 3.011e+08 |
| neural mean | ft_transformer / mpe_corrupt_8 | 32.0000 | 3.121e+05 | — | 16.2340 | 2.125e+08 |
| neural mean | ft_transformer / mpe_corrupt_9 | 32.0000 | 2.598e+05 | — | 11.4828 | 1.947e+08 |
| neural mean | ft_transformer / mpe_equality | 32.0000 | 2.210e+05 | — | 22.6202 | 1.559e+08 |
| neural mean | ft_transformer / nystrom | 32.0000 | 2.015e+05 | — | 43.5523 | 2.398e+08 |
| neural mean | ft_transformer / q_ple | 32.0000 | 2.060e+05 | — | 34.4996 | 2.544e+08 |
| neural mean | ft_transformer / similarity_same_metric | 32.0000 | 2.064e+05 | — | 114.3567 | 3.309e+08 |
| neural mean | ft_transformer / similarity_unnormalized | 32.0000 | 1.997e+05 | — | 115.9991 | 2.479e+08 |
| neural mean | ft_transformer / uniform_ple | 32.0000 | 2.295e+05 | — | 13.7825 | 1.752e+08 |
| neural mean | ft_transformer / unknown_embedding | 57.0000 | 2.418e+05 | — | 37.1179 | 2.355e+08 |
| neural mean | mlp / character_3gram_hash | 128.0000 | 1.747e+05 | — | 13.3702 | 1.088e+08 |
| neural mean | mlp / mpe | 32.0000 | 1.197e+05 | — | 344.9325 | 4.260e+08 |
| neural mean | mlp / mpe_corrupt_0 | 32.0000 | 1.385e+05 | — | 19.9589 | 9.810e+07 |
| neural mean | mlp / mpe_corrupt_1 | 32.0000 | 1.316e+05 | — | 16.6364 | 9.875e+07 |
| neural mean | mlp / mpe_corrupt_2 | 32.0000 | 1.167e+05 | — | 19.0763 | 9.977e+07 |
| neural mean | mlp / mpe_corrupt_3 | 32.0000 | 1.005e+05 | — | 18.1798 | 1.006e+08 |
| neural mean | mlp / mpe_corrupt_4 | 32.0000 | 1.397e+05 | — | 17.6533 | 1.016e+08 |
| neural mean | mlp / mpe_corrupt_5 | 32.0000 | 1.378e+05 | — | 12.8828 | 1.018e+08 |
| neural mean | mlp / mpe_corrupt_6 | 32.0000 | 1.326e+05 | — | 12.6461 | 9.781e+07 |
| neural mean | mlp / mpe_corrupt_7 | 32.0000 | 1.709e+05 | — | 12.5224 | 1.034e+08 |
| neural mean | mlp / mpe_corrupt_8 | 32.0000 | 1.955e+05 | — | 28.2598 | 9.633e+07 |
| neural mean | mlp / mpe_corrupt_9 | 32.0000 | 1.823e+05 | — | 9.2457 | 9.574e+07 |
| neural mean | mlp / mpe_equality | 32.0000 | 1.576e+05 | — | 31.9758 | 1.642e+08 |
| neural mean | mlp / nystrom | 32.0000 | 1.376e+05 | — | 59.4433 | 3.465e+08 |
| neural mean | mlp / q_ple | 32.0000 | 1.189e+05 | — | 37.3182 | 1.983e+08 |
| neural mean | mlp / raw_coordinates | 3.0000 | 1.531e+05 | — | 72.5056 | 4.992e+08 |
| neural mean | mlp / similarity_same_metric | 32.0000 | 1.364e+05 | — | 136.6116 | 3.023e+08 |
| neural mean | mlp / similarity_unnormalized | 32.0000 | 1.378e+05 | — | 82.2185 | 3.399e+08 |
| neural mean | mlp / uniform_ple | 32.0000 | 1.493e+05 | — | 36.5580 | 1.836e+08 |
| neural mean | mlp / unknown_embedding | 124.5556 | 1.504e+05 | — | 40.7310 | 4.916e+08 |
| neural mean | resnet / character_3gram_hash | 128.0000 | 3.790e+05 | — | 15.6821 | 1.089e+08 |
| neural mean | resnet / mpe | 32.0000 | 3.173e+05 | — | 527.0050 | 4.307e+08 |
| neural mean | resnet / mpe_corrupt_0 | 32.0000 | 2.603e+05 | — | 7.6503 | 1.180e+08 |
| neural mean | resnet / mpe_corrupt_1 | 32.0000 | 1.975e+05 | — | 6.7725 | 1.124e+08 |
| neural mean | resnet / mpe_corrupt_2 | 32.0000 | 3.573e+05 | — | 6.3947 | 1.216e+08 |
| neural mean | resnet / mpe_corrupt_3 | 32.0000 | 3.617e+05 | — | 8.5622 | 1.142e+08 |
| neural mean | resnet / mpe_corrupt_4 | 32.0000 | 3.104e+05 | — | 12.4040 | 1.141e+08 |
| neural mean | resnet / mpe_corrupt_5 | 32.0000 | 2.278e+05 | — | 8.0225 | 1.029e+08 |
| neural mean | resnet / mpe_corrupt_6 | 32.0000 | 3.016e+05 | — | 8.7470 | 1.154e+08 |
| neural mean | resnet / mpe_corrupt_7 | 32.0000 | 1.920e+05 | — | 6.3244 | 1.070e+08 |
| neural mean | resnet / mpe_corrupt_8 | 32.0000 | 2.478e+05 | — | 10.2297 | 1.044e+08 |
| neural mean | resnet / mpe_corrupt_9 | 32.0000 | 2.635e+05 | — | 7.3672 | 1.000e+08 |
| neural mean | resnet / mpe_equality | 32.0000 | 2.537e+05 | — | 10.6449 | 1.100e+08 |
| neural mean | resnet / nystrom | 32.0000 | 2.964e+05 | — | 45.8994 | 1.340e+08 |
| neural mean | resnet / q_ple | 32.0000 | 3.055e+05 | — | 10.5621 | 1.014e+08 |
| neural mean | resnet / similarity_same_metric | 32.0000 | 3.419e+05 | — | 118.0543 | 1.862e+08 |
| neural mean | resnet / similarity_unnormalized | 32.0000 | 2.722e+05 | — | 78.6164 | 1.740e+08 |
| neural mean | resnet / uniform_ple | 32.0000 | 2.962e+05 | — | 7.9655 | 9.896e+07 |
| neural mean | resnet / unknown_embedding | 70.9333 | 2.040e+05 | — | 46.6423 | 1.311e+08 |
| neural mean | tabm / character_3gram_hash | 128.0000 | 1.076e+05 | — | 10.1670 | 1.845e+08 |
| neural mean | tabm / mpe | 32.0000 | 1.143e+05 | — | 380.2651 | 5.085e+08 |
| neural mean | tabm / mpe_corrupt_0 | 32.0000 | 85560.0000 | — | 14.0055 | 1.511e+08 |
| neural mean | tabm / mpe_corrupt_1 | 32.0000 | 1.012e+05 | — | 13.3890 | 2.357e+08 |
| neural mean | tabm / mpe_corrupt_2 | 32.0000 | 1.343e+05 | — | 9.0644 | 2.169e+08 |
| neural mean | tabm / mpe_corrupt_3 | 32.0000 | 1.201e+05 | — | 14.6144 | 2.170e+08 |
| neural mean | tabm / mpe_corrupt_4 | 32.0000 | 1.425e+05 | — | 11.1120 | 1.690e+08 |
| neural mean | tabm / mpe_corrupt_5 | 32.0000 | 1.246e+05 | — | 5.1238 | 2.263e+08 |
| neural mean | tabm / mpe_corrupt_6 | 32.0000 | 1.471e+05 | — | 5.6923 | 2.140e+08 |
| neural mean | tabm / mpe_corrupt_7 | 32.0000 | 82781.3333 | — | 6.1110 | 1.520e+08 |
| neural mean | tabm / mpe_corrupt_8 | 32.0000 | 1.514e+05 | — | 10.1409 | 1.312e+08 |
| neural mean | tabm / mpe_corrupt_9 | 32.0000 | 92264.0000 | — | 7.9295 | 1.509e+08 |
| neural mean | tabm / mpe_equality | 32.0000 | 1.302e+05 | — | 29.9346 | 2.299e+08 |
| neural mean | tabm / nystrom | 32.0000 | 1.183e+05 | — | 39.9761 | 2.974e+08 |
| neural mean | tabm / q_ple | 32.0000 | 1.323e+05 | — | 19.0944 | 1.905e+08 |
| neural mean | tabm / similarity_same_metric | 32.0000 | 1.243e+05 | — | 132.1420 | 3.307e+08 |
| neural mean | tabm / similarity_unnormalized | 32.0000 | 1.379e+05 | — | 81.1247 | 3.348e+08 |
| neural mean | tabm / uniform_ple | 32.0000 | 82925.3333 | — | 14.5270 | 1.594e+08 |
| neural mean | tabm / unknown_embedding | 68.2143 | 1.164e+05 | — | 14.0872 | 2.970e+08 |
