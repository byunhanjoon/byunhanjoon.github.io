# Phase I pilot panel freeze

Frozen: 2026-08-31, before any reparameterization model outcome was viewed.

The panel is the 12-task, task-balanced TabArena/OpenML slice recorded in `configs/audit/pilot.yaml`: four regression, four binary-classification, and four multiclass-classification tasks. It spans 898–78,053 source rows, 7–39 source features, predominantly numerical and heavily mixed schemas, and both continuous and strongly tied/integer-like numerical columns. The fixed Phase I context/query caps are 2,048/1,024 rows; these caps and row sampling seed were fixed before outcomes.

The selected tasks are:

| Problem type | Datasets |
|---|---|
| Regression | wine_quality, miami_housing, Food_Delivery_Time, diamonds |
| Binary | seismic-bumps, churn, heloc, credit_card_clients_default |
| Multiclass | anneal, maternal_health_risk, students_dropout_and_academic_success, SDSS17 |

The OpenML official repeat/fold is `(0, 0)` for all tasks. Inner validation membership and context/query subsampling use seed `20260831`. Model/warp seeds `[20260831, 20260832, 20260833]` were frozen before any OpenML Phase I outcomes were produced so that matched sensitivity can be compared with ordinary stochastic variation. Data-dependent transforms are fitted only on the final context rows. Pilot tasks are development data and are not the later frozen confirmatory benchmark.

This freeze is invalidated if dataset membership, split/fold, context/query cap, transformation values, or Phase I model settings are altered after results are inspected. A changed protocol must receive a new config and immutable run phase; old outputs remain in place.
