# Prospective conditional-swap rescue: results

## Verdict

**The current contextual active-search algorithm fails the prospective ICLR gate.**

It reliably finds contexts that improve over their random starts, but it does not outperform an otherwise identical row-only learner or equal-budget random local search on untouched test data. The strongest new signal is selection-induced validation overfitting, not a successful contextual optimizer.

## Design

- Eight datasets, with `credit-g` and `diamonds` designated as development and the other six as prospective confirmation.
- Three independently generated, stratified starting contexts per dataset.
- Context size 32 from 256 candidates.
- Frozen TabICLv2 backbone.
- `contextual_active`, `row_active`, and `random_search` each received 257 logical selector calls per trajectory.
- Static additive/FM methods used five calls. The diagnostic `vip_style` baseline used 254 calls and is not an official VIP-COP reproduction.
- Final test labels were not used for fitting, acquisition, stopping, or method changes.
- The complete design was frozen in `PROTOCOL.md` before any independent-start test outcomes were generated.

## Primary confirmation result

Mean across 18 confirmation trajectories:

| Method | Selector gain | Untouched-test gain | Test/selector transfer |
|---|---:|---:|---:|
| Random local search | 0.1699 | **0.1059** | 62% |
| Row-only active | 0.1845 | 0.0999 | 54% |
| Contextual active | **0.1904** | 0.0894 | 47% |
| VIP-style fixed attribution | 0.1489 | 0.0407 | — |
| Static additive | 0.0827 | 0.0302 | — |
| Static FM | 0.0569 | 0.0243 | — |

The contextual learner wins on selector utility but loses after transferring to unseen test queries:

- Positive untouched-test improvement: 18/18 trajectories.
- Beats row-only on test: 5/18, below the frozen 70% gate.
- Mean contextual minus row-only test gain: -0.0105; paired bootstrap 95% interval `[-0.0241, 0.0039]`.
- Mean contextual minus random test gain: -0.0165; paired bootstrap 95% interval `[-0.0383, 0.0029]`.
- Mean contextual minus row-only selector gain: +0.0058; paired bootstrap interval `[-0.0027, 0.0138]`.

Thus the extra context features provide no reliable selector advantage and no test advantage.

## Post-hoc round audit

The round audit uses test outcomes diagnostically after all methods and hyperparameters were frozen. It must not be interpreted as a prospective stopping-rule comparison.

| Round | Contextual test gain | Row-only test gain | Random test gain |
|---:|---:|---:|---:|
| 1 | 0.0622 | 0.0613 | 0.0535 |
| 2 | 0.0719 | 0.0829 | 0.0779 |
| 3 | 0.0871 | 0.0914 | 0.0970 |
| 4 | 0.0894 | 0.0999 | 0.1059 |

At round 1, contextual minus row-only is only +0.0009 and contextual minus random is +0.0087; both bootstrap intervals cross zero. Random search overtakes by later rounds. There is no defensible early-round win to rescue post hoc.

## Interpretation

The retrospective result remains valid: swap values change with the context, and a conditional learner can predict some of that variation along the two oracle trajectories. But that did not yield a better prospective optimizer from new random starts.

The end-to-end experiment instead exposes a monotone winner's-curse pattern:

1. Random search has the lowest selector gain among the three budget-matched local methods but the highest test gain.
2. Row-only modeling increases selector gain and reduces transfer.
3. Contextual modeling increases selector gain again and reduces transfer further.

The contextual learner is optimizing increasingly fine distinctions on a fixed 128-query selector set. Those distinctions do not generalize well enough to the untouched query distribution.

## Decision

The proposed paper claim—conditional swap modeling improves black-box context optimization—does not survive the prospective test. Do not scale this algorithm to 30 datasets or market it against VIP-COP.

A different project may remain viable around **selection-aware generalization for context optimization**: cross-fitted query utilities, bootstrap lower-confidence acquisition, or reusable validation predictions could target the observed winner's curse without additional TFM calls. That is a new hypothesis and requires a new frozen protocol and genuinely untouched datasets; it cannot be established by retuning on this panel.

## Artifacts

- `PROTOCOL.md`: prospective freeze.
- `run_prospective.py`: end-to-end implementation and reporting.
- `results.csv`, `summary.csv`, and `audit.json`: primary outcomes and gate audit.
- `posthoc_round_audit.py`, `posthoc_round_results.csv`, and `posthoc_round_summary.csv`: labeled post-hoc trajectory analysis.
- `raw/*_calls.csv`: complete logical call ledger.
- `processed/*.csv`: one record per dataset/method/start plus post-hoc round records.
