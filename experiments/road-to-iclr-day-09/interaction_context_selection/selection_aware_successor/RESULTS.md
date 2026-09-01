# Selection-aware context search: prospective results

## Verdict

**The rotating scout/judge successor fails.** It is safe in the weak sense that it improves untouched-test utility in 22/24 trajectories, but it does not beat the original mean-selector contextual method, rotating row-only search, or equal-budget random search.

This is a second independent prospective failure of learned contextual acquisition. The branch should stop here rather than add another adaptive patch.

## Frozen experiment

- Eight datasets not used by the parent context-selection experiments: four classification and four regression.
- Fresh deterministic 256-candidate/128-selector/256-test partitions.
- Three independent starts per dataset.
- Six methods and 257 logical selector calls per method/start: 144 trajectories total.
- Four selector folds. At each round, three scout folds trained/proposed swaps and a rotating held-out judge fold confirmed moves.
- The successor accepted a move only when both the mean scout gain and judge gain were positive.
- Test labels were accessed only after final contexts were fixed.

## Primary result

Mean over all 24 dataset/start pairs:

| Method | Selector gain | Untouched-test gain | Selector optimism |
|---|---:|---:|---:|
| Contextual mean | **0.1762** | **0.1381** | 0.0381 |
| Random mean | 0.1707 | 0.1358 | 0.0349 |
| Row rotating | 0.1665 | 0.1339 | **0.0327** |
| Row mean | 0.1742 | 0.1326 | 0.0415 |
| Contextual rotating | 0.1747 | 0.1293 | 0.0454 |
| Random rotating | 0.1559 | 0.1250 | 0.0309 |

The successor's paired comparisons:

| Baseline | Wins | Mean test difference | Bootstrap 95% interval |
|---|---:|---:|---:|
| Contextual mean | 8/24 | -0.0088 | [-0.0301, 0.0140] |
| Row rotating | 8/24 | -0.0046 | [-0.0265, 0.0189] |
| Random mean | 8/24 | -0.0066 | [-0.0254, 0.0135] |

Only gates 1 and 5 pass: the method usually improves the starting context and has positive mean gains in classification and regression. All comparative gates fail.

## Task breakdown

| Task | Contextual mean | Contextual rotating | Random mean | Row rotating |
|---|---:|---:|---:|---:|
| Classification | 0.0676 | 0.0703 | **0.0727** | 0.0678 |
| Regression | **0.2086** | 0.1882 | 0.1990 | 0.2000 |

The rotating contextual rule is roughly neutral for classification and harmful for regression. It helps notably on `puma32h` and `sick`, but loses on six of eight dataset means, including substantial losses on `cpu-act`, `elevators`, and `kin8nm`.

## What happened

The fold safeguard did not reduce winner's curse for the contextual learner. Its mean selector-minus-test gap increased from 0.0381 to 0.0454. Requiring agreement between a 96-query scout and a 32-query judge made acquisition more conservative, but the judge estimate was itself noisy and the rule discarded useful moves.

The control comparisons support that diagnosis:

- Rotating versus mean random search: -0.0108 test utility, winning 7/24.
- Rotating versus mean row-only search: +0.0012, winning 12/24.
- Rotating versus mean contextual search: -0.0088, winning 8/24.

Thus fold rotation is not a generally useful regularizer here. It occasionally helps row-only search, but there is no contextual interaction advantage.

## Decision

Do not proceed to an official VIP-COP comparison or paper-scale study for this method. Across two prospective panels, increasingly structured search has not beaten equal-budget random local search on untouched queries. The stable empirical finding is that direct local search can improve TFM contexts; the unsupported claim is that learned contextual acquisition improves that search.

Any further attempt would need a genuinely different source of information—such as model-internal representations or a much larger independent validation pool—not another selection rule fitted to the same 128 selector queries. That would be a new research branch, not a rescue of this one.

## Artifacts

- `PROTOCOL.md`: pre-outcome freeze.
- `run_successor.py`: dataset construction, fold-aware evaluation, all methods, and gate reporting.
- `results.csv` and `summary.csv`: complete trajectory outcomes.
- `audit.json`: paired comparisons and frozen gates.
- `raw/*_calls.csv`: logical call ledger including all four fold utilities.
- `processed/*.csv`: one row per method/start/dataset.
