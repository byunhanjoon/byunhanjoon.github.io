# Phase II development-suite freeze

Frozen: 2026-08-31, after the Phase I gate and before any Phase II development-suite model outcomes were run.

## Dataset selection

The development suite is the 20-task TabArena/OpenML slice in `configs/audit/main.yaml`: eight regression, seven binary-classification, and five multiclass-classification tasks. The first 12 are the already frozen Phase I panel. Eight additions were selected from the official TabArena curation metadata using task type and metadata/schema considerations only: `airfoil_self_noise`, `concrete_compressive_strength`, `physiochemical_protein`, `superconductivity`, `APSFailure`, `bank-marketing`, `diabetes`, and `MIC`. No Phase II outcome was available during selection.

The additions increase task and sample-size diversity while keeping a tractable numerical or mixed-feature audit. Very high-dimensional and all-categorical multiclass tasks were not selected because the primary intervention is numerical reparameterization. This development suite is explicitly not the later confirmatory benchmark.

| Problem type | Count | Datasets |
|---|---:|---|
| Regression | 8 | wine_quality, miami_housing, Food_Delivery_Time, diamonds, airfoil_self_noise, concrete_compressive_strength, physiochemical_protein, superconductivity |
| Binary | 7 | seismic-bumps, churn, heloc, credit_card_clients_default, APSFailure, bank-marketing, diabetes |
| Multiclass | 5 | anneal, maternal_health_risk, students_dropout_and_academic_success, SDSS17, MIC |

## Frozen evaluation protocol

- Official OpenML repeat/fold `(0, 0)` is used for all tasks.
- Two independently sampled context/query memberships use split seeds `20260831` and `20260841`.
- Context/query caps are 2,048/1,024 rows.
- Model/warp seed is `20260831`; Phase I already established the effect across three seeds, while Phase II allocates replication to independently sampled data splits for hierarchical inference.
- Twelve model configurations include single/default TabPFN v2.5 and TabICLv2, Mitra default, XGBoost, CatBoost, LightGBM, Random Forest, linear/logistic, RealMLP, and TabM.
- The 28 transform settings cover identity, increasing/decreasing affine, signed power, asinh, random monotone PWL, held-out monotone spline, atomic spacing, compositions, empirical CDF, quantile Gaussian, and categorical identity bijections.
- Data-dependent transforms and all model preprocessing are fit only on context rows.
- Raw predictions, split membership, transform state/audits, preprocessing telemetry, timing, and metrics are retained per job.

The frozen grid contains 13,440 jobs (`20 datasets × 12 models × 28 settings × 2 splits × 1 seed`). Any change to membership, caps, seeds, model settings, or transformations requires a new protocol version and phase; existing outputs remain immutable.
