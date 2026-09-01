# Prospective conditional-swap rescue protocol

Frozen before generating any outcomes for the independent-start trajectories.

## Question

Does a context-conditioned swap-gain learner improve TabICLv2 contexts more reliably than a row-only learner, random local search, and fixed additive/pairwise scores under a common selector-evaluation budget?

## Data roles

- Development datasets: `credit-g`, `diamonds`. Their earlier oracle trajectories informed the choice of ExtraTrees and the feature families.
- Prospective confirmation datasets: `adult`, `bank-marketing`, `electricity`, `california_housing`, `churn`, `house_16H`.
- Candidate, selector, and untouched test partitions are exactly those frozen in the parent experiment.
- Test labels are evaluated only after a final context has been selected.

## Fixed design

- Backbone: official frozen TabICLv2 checkpoint, one deterministic estimator.
- Candidate pool: 256 rows.
- Context size: 32 rows.
- Starts: three independently seeded, target-stratified random contexts per dataset.
- Local-search budget: at most 257 selector-context evaluations per trajectory: one initial context and four rounds of 64 swaps.
- Active rounds: 32 uniform exploration swaps followed by 32 model-guided swaps.
- Guided acquisition: ExtraTrees mean plus 0.5 times across-tree standard deviation, with at most two acquisitions sharing an outgoing row in a batch.
- Optimization target: selector utility difference from the current context.
- Models:
  - `contextual_active`: outgoing/incoming row features plus current-set, row-to-set, selector-query, and target-coverage features.
  - `row_active`: the identical learner using outgoing/incoming row features only.
  - `random_search`: 64 uniformly sampled unseen swaps per round.
  - `additive_static` and `fm_static`: deterministic best predicted swaps from the previously fitted static models.
  - `vip_style`: fixed-size membership regression with value-guided global subset sampling. This is a diagnostic baseline, not an official VIP-COP reproduction.

No method or hyperparameter will be changed after inspecting confirmation outcomes. Any subsequent method change starts a new protocol and a new confirmation panel.

## Primary metrics and gates

The unit of analysis is dataset × start.

Primary comparison: `contextual_active` versus `row_active` on the six confirmation datasets.

Proceed to paper-scale only if all hold:

1. Contextual search has at least 25% lower mean selector regret than row-only search on held-out one-step decisions. This component requires separate oracle neighborhoods and is not fully tested by the present end-to-end run.
2. Contextual search produces positive untouched-test improvement in at least 70% of confirmation trajectories.
3. Contextual search beats row-only search in untouched-test improvement on at least 70% of confirmation trajectories.
4. Contextual search beats random search in mean untouched-test improvement, with a positive paired bootstrap 95% interval once enough trajectories are available.
5. Results are not driven solely by the two development datasets.

Failure of gates 2 or 3 kills the present algorithm. Passing the end-to-end gates licenses the more expensive oracle-neighborhood and official VIP-COP comparison; it does not by itself establish an ICLR-level result.

## Integrity constraints

- Final-test outcomes cannot alter contexts, models, stopping, datasets, seeds, or hyperparameters.
- All selector calls, chosen swaps, final contexts, and test outcomes are persisted.
- Cached duplicate TFM evaluations may reduce physical runtime but still count against each method's logical call budget.
