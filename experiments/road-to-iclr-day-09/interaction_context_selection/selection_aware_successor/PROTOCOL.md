# Selection-aware context search: prospective protocol

Frozen before generating any outcomes on this context-selection panel.

## Hypothesis

The previous contextual optimizer suffered selection-induced validation overfitting: methods with larger gains on a fixed selector-query set transferred a smaller fraction to untouched test queries. A rotating scout/judge rule can reduce that winner's curse without additional TFM calls because every context evaluation already produces predictions for every selector query.

## Untouched context-selection panel

Classification:

- `breast-w` (OpenML 15)
- `credit-approval` (OpenML 29)
- `blood-transfusion` (OpenML 1464)
- `sick` (OpenML 38)

Regression:

- `kin8nm` (OpenML 189)
- `puma32h` (OpenML 308)
- `cpu-act` (OpenML 573)
- `elevators` (OpenML 216)

These datasets may exist elsewhere in the repository, but none was used to design or evaluate the parent context-selection experiments. This experiment creates a fresh deterministic 256-candidate/128-selector/256-test partition using split seed 1729.

## Frozen design

- Backbone: official frozen TabICLv2, one deterministic estimator.
- Context size: 32.
- Three stratified random starts per dataset: seeds 5201, 5202, and 5203.
- Four search rounds, 32 uniform exploration swaps and 32 guided swaps per round.
- Logical selector budget: 257 context evaluations per method and start.
- The 128 selector queries are deterministically stratified into four folds.
- In round `r`, fold `r mod 4` is the judge and the other three are scouts.
- Active acquisition is trained only on the mean gain over the current scout folds.
- A proposed move receives score `min(mean scout gain, judge gain)` and is accepted only when this score is positive.
- Guided acquisition uses the already frozen ExtraTrees/UCB configuration: 200 trees, minimum leaf 2, 70% features, mean plus 0.5 standard deviations, and at most two proposals per outgoing row.

## Methods

- `contextual_rotating`: selection-aware successor with contextual action features.
- `row_rotating`: identical selection-aware method without current-context features.
- `random_rotating`: random proposals with the same rotating conservative move rule.
- `contextual_mean`: previous contextual method optimizing mean selector utility.
- `row_mean`: previous row-only method optimizing mean selector utility.
- `random_mean`: equal-budget random search optimizing mean selector utility.

## Gates

Primary unit: dataset × start, 24 paired confirmation trajectories per method.

Proceed only if all hold:

1. `contextual_rotating` has positive untouched-test improvement in at least 70% of trajectories.
2. It beats `contextual_mean` in at least 60% of paired trajectories and has positive mean paired test difference.
3. It beats `row_rotating` in at least 60% of paired trajectories and has positive mean paired test difference.
4. Its mean paired test difference versus `random_mean` is positive, with a bootstrap interval whose lower endpoint is not materially below -0.005.
5. The conclusions are not supported only by classification or only by regression.

Passing these gates would justify an official VIP-COP comparison and a larger dataset panel. Failure kills this selection-aware implementation.

## Integrity

- Test labels are evaluated only after a trajectory's final context is fixed.
- No test result may alter methods, folds, starts, hyperparameters, stopping, or gates.
- Selector fold utilities, logical calls, contexts, and post-selection test outcomes are persisted.
- A cached duplicate counts against the logical method budget even if it saves a physical TFM call.
