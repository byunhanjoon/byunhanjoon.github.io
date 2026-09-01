# Table 3 — Main real unseen-state result

Primary loss is state-balanced standardized MSE; values average tasks, settings, splits, seeds, and available frozen backbones equally.

| Source | Status | MPE | Best metric baseline | Baseline name(s) | Similarity | PLE | UNK | Mean corrupt MPE | Relative gain (%) | Winner |
|---|---|---|---|---|---|---|---|---|---|---|
| ACS | RUN | 0.6275 | 0.6153 | knn_metric, laplacian, node2vec, path_to_root | 0.6254 | 0.6202 | 0.6790 | 0.6760 | -1.9789 | baseline |
| NYC_TLC | RUN | 0.9510 | 0.9495 | knn_metric, nystrom, raw_coordinates, rbf_normalized, rbf_unnormalized, spatial_rbf | 0.9465 | 1.0957 | 1.0901 | 1.1693 | -0.1630 | baseline |
| CITI_BIKE | RUN | 1.0109 | 1.0047 | knn_metric, rbf_normalized | 1.0109 | 1.0141 | 1.0212 | 1.0212 | -0.6111 | baseline |
| BTS | RUN | 1.1196 | 1.1195 | raw_coordinates, raw_latlon, rbf_normalized, rbf_unnormalized, spatial_rbf | 1.1195 | 1.1218 | 1.1206 | 1.1202 | -0.0094 | baseline |
| AMAZON_2023 | NOT RUN | — | — | — | — | — | — | — | — | — |
