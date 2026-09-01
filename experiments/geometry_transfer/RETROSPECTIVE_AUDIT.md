# Retrospective audit

Status: **PASS — 405/405 declared cells complete**

- Nine runnable tasks, five frozen state splits, and nine operators are present.
- Train/validation/test semantic states are disjoint in all 45 task-splits.
- The CatBoost base excludes `field_state`; training residuals are genuine
  three-fold row-OOF predictions within observed states.
- State effects and diagonal finite-sample Sigma use observed-state residuals.
- Operator bandwidths and matrices use the target-independent metric only.
- All 79 harmful cells are retained; no task/operator was filtered by outcome.
- The realized-oracle gain equals the row-level MSE difference to numerical
  precision (MAE `1.10e-17`), as the algebra predicts.
- This exact realized relation is explicitly labeled an oracle arithmetic
  identity. The population plug-in `Delta_theory` subtracts estimated repeated-
  training noise and is not presented as independently observable.
- Primary retrospective Spearman is `0.9909`, sign accuracy `0.9630`; source-
  level Spearman is positive in all five source families.
- Gate R1–R5 passes, authorizing the separately frozen prospective stage.

Known limitation: the primary base family is CatBoost. Weak/medium/strong
CatBoost conditioning is tested on three tasks, but the planned neural-family
robustness comparison was not computationally justified after the earlier MPE
matrix had been stopped and is not claimed.
