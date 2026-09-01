# Baseline freeze

Frozen: 2026-08-31 before Day-09 model outcomes.

The official living leaderboard CSV was read programmatically from the
[TabArena leaderboard space](https://huggingface.co/spaces/TabArena/leaderboard/blob/main/README.md),
using the all-splits/all-tasks/all-datasets, imputation-enabled model table. The leading
verified open model rows at freeze were:

| Model | Elo | Type |
|---|---:|---|
| TabFM default | 1765 | foundation model |
| EXAONE-Tabular default | 1755 | foundation model |
| TabPFN-3 default | 1642 | foundation model |
| TabPFN-2.6 default | 1592 | foundation model |
| RealTabPFN-2.5 tuned + ensembled | 1572 | foundation model |
| TabICLv2 default | 1569 | foundation model |
| RealMLP tuned + ensembled | 1482 | neural network |
| TabM tuned + ensembled | 1426 | neural network |
| LightGBM tuned + ensembled | 1410 | tree-based |
| CatBoost tuned + ensembled | 1398 | tree-based |

Mandatory scientific hosts remain TabPFN-3 default/OOD, TabICLv2 default/minimal,
and Mitra. Strong controls remain TabM, RealMLP, XGBoost, CatBoost and LightGBM. TabFM
and EXAONE-Tabular are added as frozen current-frontier context if their public interfaces
and resource requirements are compatible. No later baseline may influence method choice;
additions must be labeled post-hoc.

