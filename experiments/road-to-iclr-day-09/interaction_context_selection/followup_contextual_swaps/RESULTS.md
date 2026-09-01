# Follow-up: context-conditional swap signal

## Bottom line

There is a sharper signal than the failed static pairwise factorization: **the value of a swap changes with the current context, and part of that change is predictable**. This supports a new research direction, but the present two-dataset pilot is not publication evidence and the naive cold-start budgeted learner is too weak.

The promising paper object is not a global context utility model. It is the conditional action value

\[
\Delta(i \to j \mid S,Q)
= U((S\setminus\{i\})\cup\{j\};Q)-U(S;Q),
\]

where `S` is the current context and `Q` is the selector-query set.

## What the exhaustive neighborhoods say

The analysis uses all 35,840 swap/state rows per dataset from the five saved exhaustive one-swap neighborhoods. No new TFM outcomes were generated.

| Diagnostic | credit-g | diamonds |
|---|---:|---:|
| Mean rank correlation for the same swaps across contexts | 0.524 | 0.535 |
| Minimum cross-context rank correlation | 0.270 | 0.276 |
| Mean swap sign-flip rate | 26.1% | 23.5% |
| Initial-to-final-neighborhood sign-flip rate | 45.3% | 35.4% |
| Median within-swap standard deviation | 0.0193 | 0.0076 |
| Median absolute within-swap mean | 0.0183 | 0.0095 |

The useful pattern is temporal. Nearby late-search contexts have fairly stable rankings, but the ranking learned at the initial context becomes stale as local search moves away from it. For example, credit-g's rank correlation falls from 0.457 at rounds 0→1 to 0.270 at rounds 0→4; diamonds falls from 0.623 to 0.276.

This explains the earlier contradiction:

- A static additive or pairwise model is a poor optimizer because it compresses changing marginal gains into fixed coefficients.
- Exhaustive local search works because it re-evaluates the marginal gains at the current state.
- A global DeepSets utility predictor can have decent R² while missing the extreme tail needed for optimization.

## Can the conditional part be predicted?

The stricter test trains only on earlier neighborhoods and predicts the next neighborhood. The contextual model sees the outgoing row, incoming row, current-set summaries, row-to-set relations, query summaries, and target-coverage changes. The row-only model omits current-set information.

Mean across eight dataset/round predictions:

| Method | Spearman | Chosen-swap percentile | Mean regret | Improving choices |
|---|---:|---:|---:|---:|
| Static additive delta | 0.133 | 0.129 | 0.0676 | 0/8 |
| Static FM delta | 0.135 | 0.266 | 0.0463 | 0/8 |
| Row-only ExtraTrees | 0.561 | 0.902 | 0.0139 | 5/8 |
| Contextual ExtraTrees | **0.614** | **0.974** | **0.0090** | **7/8** |

The contextual features reduce next-neighborhood regret by 35% relative to the row-only model. The largest qualitative difference is credit-g round 1: the contextual model selects a +0.0270 swap at the 99.75th percentile, while the row-only model selects a -0.0139 swap near the median.

This result is encouraging but small-sample: there are only two datasets and eight sequential decisions. Leave-one-round-out tests also show that an empirical static lookup of the same swap in other, including future, contexts is highly competitive. That baseline is unrealistic online but confirms that much of the signal remains swap-specific rather than context-specific.

## Sample-efficiency result is not yet good enough

In a cold-start simulation, ExtraTrees was fit to `B` random evaluations from the current neighborhood and allowed to propose one additional swap. It only slightly improved over the best of the random evaluations:

| B | Random regret | Learned-plus-one regret |
|---:|---:|---:|
| 16 | 0.0188 | 0.0185 |
| 32 | 0.0172 | 0.0162 |
| 64 | 0.0105 | 0.0102 |
| 128 | 0.0096 | 0.0088 |
| 256 | 0.0070 | 0.0068 |

Therefore, “fit a generic tree inside each neighborhood” is not the algorithm. A credible method must transfer information across states and actively choose informative swaps.

## ICLR-worthy reformulation

Working title: **Context Value Is Not a Scalar: Conditional Marginal-Gain Optimization for Tabular Foundation Models**.

The core story would be:

1. **Diagnosis.** Fixed sample importance, fixed pairwise interactions, and global set regression lose decision-relevant information because exchange gains drift as the context changes.
2. **Object.** Learn a permutation-equivariant state-action model `q(S, Q, i→j)` of swap gains, trained with a listwise or top-tail ranking loss rather than MSE on whole-set utility.
3. **Search.** Couple it to uncertainty-aware active local search. Reuse observations across consecutive, highly overlapping neighborhoods; refresh when measured exchange instability says the scorer is stale.
4. **Claim.** Under an equal black-box TFM-call budget, conditional exchange modeling reaches better contexts faster than fixed-value attribution and geometry-only selection.

The closest work makes this positioning narrow. [VIP-COP](https://arxiv.org/abs/2605.12904) already performs anytime black-box context optimization using online KernelSHAP regression and value-guided sampling, but ultimately assigns fixed item values and selects the top-valued items. [LUCoS](https://arxiv.org/abs/2605.27254) covers latent-geometry cold-start selection. [CRUMB](https://arxiv.org/abs/2606.11473) covers query-clustered MMD matching. [CASE](https://proceedings.mlr.press/v267/purohit25a.html) covers sample-efficient combinatorial demonstration selection for LLMs using a linear-bandit formulation. The novelty must therefore be the state-dependent exchange value, its failure diagnosis for fixed attribution, and a demonstrably better budget/performance frontier.

## Fast kill gates before scaling

Run a prospective 8–10 dataset rescue panel before committing to a full paper:

1. Generate three independently seeded local-search trajectories per dataset, rather than analyzing one oracle trajectory.
2. Evaluate contextual, row-only, VIP-COP, and random/challenger search under identical validation-call budgets.
3. Require the contextual model to reduce one-step regret by at least 25% over the strongest non-contextual learner and select an improving swap in at least 70% of held-out decisions.
4. Require end-to-end final-context gains on untouched test labels in at least 70% of datasets, with no dataset selected or tuned using test outcomes.
5. Stop if the gain disappears under independent starting contexts, because the present neighborhoods are all from one greedy path per dataset.

If those gates pass, the paper-scale study should use at least 30 classification/regression datasets, TabICLv2 plus a current TabPFN model, multiple `K` and candidate-pool sizes, and call-budget curves against VIP-COP, CRUMB, LUCoS, random search, additive attribution, and static pairwise models. A theory component can formalize **exchange instability**—the variance of `Δ(i→j | S,Q)` over `S`—and give a lower bound for any context-independent scorer when swap preferences reverse.

## Verdict

**Conditional proceed.** The original FM hypothesis is dead. The stronger signal—predictable, state-dependent exchange gains—is worth one tightly budgeted prospective rescue experiment. It becomes ICLR-plausible only if it produces a clear equal-call-budget advantage over VIP-COP; otherwise it is a useful negative/mechanistic result, not a competitive main-track method paper.

## Artifacts

- `analyze_contextual_swaps.py`: deterministic reconstruction and analysis.
- `audit.json`: neighborhood sizes and stability summaries.
- `predictive_results.csv` and `predictive_summary.csv`: all predictive folds and aggregates.
- `budget_simulation.csv` and `budget_summary.csv`: cold-start budget simulations.
- `*_cross_round_stability.csv`: pairwise context-drift diagnostics.
