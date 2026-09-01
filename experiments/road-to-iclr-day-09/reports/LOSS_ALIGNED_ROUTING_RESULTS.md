# Loss-aligned routing: untouched-test result

Status: **the preregistered performance-opportunity gate passes; this is a successful
fallback experiment, not a reopening of M6 or E4–E10**.

## Question and frozen design

The earlier fallback routed the same six frozen experts by inferred generator family.
That objective was highly identifiable but predictively misaligned for classification.
This experiment instead estimates each expert's competence using context-only three-fold
cross-validation and converts the six losses to softmax weights. It never observes query
labels, generator family, warp identity, rho, or coupling state at routing time.

The protocol, expert panel, temperature/shrinkage grid, development/test seeds, primary
contrast, bootstrap, and stop gate were fixed in
`LOSS_ALIGNED_ROUTING_PROTOCOL.md` before the test run. Development contained 4,800
episodes; the untouched test contained 9,600 episodes covering both task types, four
`(context rows, features)` regimes, and five rho values. Every cell has 240 test episodes
and 128 query points per episode.

## Primary untouched-test result

Positive gain means lower loss for competence routing.

| Task | Fixed loss | Competence loss | Gain vs fixed (95% paired task-bootstrap CI) | Fixed-to-best-individual headroom | Headroom captured (95% CI) | Episode win rate |
|---|---:|---:|---:|---:|---:|---:|
| Classification | 0.625964 | 0.620500 | **0.005465 [0.004797, 0.006129]** | 0.018520 | **29.51% [26.34%, 32.65%]** | 60.75% |
| Regression | 0.461034 | 0.206850 | **0.254185 [0.246516, 0.262313]** | 0.255046 | **99.66% [98.88%, 100.43%]** | 95.19% |

Both task types clear the frozen requirement of a positive interval and at least 20%
oracle-headroom capture. Neither task is materially harmed, so the aggregate
performance-opportunity gate passes.

The high-dimensional/high-coupling classification slice also passes its separate claim
gate: at 12 features and rho at least 0.75, competence routing improves over fixed by
0.004811 [0.003364, 0.006253]. Across the complete 40-cell test grid, only two cells have
negative point estimates, both at 64 context rows, 12 features, and rho 0 or 0.25.

## Comparator and alignment diagnostics

| Task | Gain vs uniform | Gain vs shape-family router | Context-CV argmin equals query-best expert |
|---|---:|---:|---:|
| Classification | 0.009122 [0.008447, 0.009781] | 0.008479 [0.007656, 0.009301] | 42.02% |
| Regression | 0.261008 [0.252859, 0.269654] | 0.192718 [0.185728, 0.199956] | 83.08% |

The classification result is the crucial reversal. Family-based routing was harmful in
the independent fallback replication even with 99.2% mechanism identification, whereas
loss-aligned routing is beneficial despite matching the query-best individual expert on
only 42.0% of episodes. Exact expert identity is therefore neither necessary nor a safe
surrogate for calibrated mixture quality in this panel.

Regression competence routing nearly reaches the per-episode best-individual diagnostic
on average and sometimes improves on it because a convex mixture can outperform every
individual expert. The `100.43%` upper confidence limit is consequently possible and is
not an integrity error.

## What is and is not novel

Cross-validated stacking, dynamic ensemble selection, and competence weighting are prior
art; this implementation is not claimed as a new ensemble algorithm. The remaining
candidate contribution is the controlled benchmark finding: a fixed-marginal dial can
make generator metadata almost perfectly identifiable while that metadata remains an
unsafe routing objective, whereas context-only predictive competence recovers usable
signal and held-out performance. This is a target-alignment result, not a general claim
that this simple router is state of the art.

## Integrity and compute

- Development run: `fallback_loss_router_2e46ddf857_development`, 4,800 episodes,
  115,200 expert fits, 582.67 seconds.
- Test run: `fallback_loss_router_2e46ddf857_test`, 9,600 episodes, 230,400 expert fits,
  1,165.01 seconds.
- Test bundle SHA-256:
  `004e5acd3ee4d8024bba543ef1bc0ad30e0f12a36ddc7f07e9548b3f25bb6b51`.
- All numeric arrays are finite and have the declared 4,800/9,600 episode dimensions.
- At the time of the final continuation audit, the 149-record manifest parses and every
  referenced raw/processed artifact exists.
- Exact test contrasts are in
  `results/processed/fallback_loss_router_contrasts_v1.csv`; the gate record is in
  `results/processed/fallback_loss_router_audit_v1.json`; the frozen figure is
  `figures/fallback_loss_router_v1.png`.

## Decision

Keep the result as a successful, scoped theory/benchmark fallback. Do not relabel it as
the killed M6 program and do not promote the generic routing rule as standalone novelty.
Subsequent frozen controls completed that falsification: soft weighting beat hard
selection, cyclic loss-to-expert assignment nulls failed, real regression transfer was
independently confirmed, and full classification transfer failed through its high-NLL
tail. See `ALIGNMENT_CONTROLS.md`, `OPENML_COMPETENCE_RESULTS.md`, and the tail-risk
reports for the completed chain.
