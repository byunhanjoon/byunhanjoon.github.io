# E1 — PriorDial development validation

Status: **shape-information dial independently replicated; predictive utility replicated
only for regression**. The ordinary raw/rank crossover and G3 gates failed, so held-out
E5 was not authorized.

## Frozen evidence

Primary run: 420 independent tasks per cell, `n=64`, `d=8`, both classification and
regression, all seven declared rho values. The mechanism and warp marginals are exactly
balanced at every rho. All mechanism selection is five-fold cross-fitted; selectors use
context labels but never query labels. Raw predictions and OOF selector probabilities are
immutable in `results/raw/e1_primary_b779842a24_t420_n64.npz`; metadata and config hashes
are adjacent. The processed table is
`results/processed/e1_primary_b779842a24_t420_n64_summary.csv`.

At rho=0, empirical `I(C;W)` was 0.0008 nats. Marginal-only mechanism accuracy was 0.145
(95% bootstrap CI 0.112–0.179) for classification and 0.121 (0.090–0.155) for regression,
both consistent with 1/6 chance. Conditional marginal query utility—loss of an invariant
context selector minus loss after adding label-free marginal shape—was 0.0006
(-0.0000–0.0013) for classification and -0.0052 (-0.0134–0.0031) for regression.

At rho=1, empirical `I(C;W)` reached `log(6)=1.7918` nats. Mechanism accuracy was 0.902
(0.874–0.931) for classification and 0.905 (0.876–0.931) for regression. Conditional
marginal query utility was 0.0073 log-loss units (0.0043–0.0103) and 0.2052 standardized
MSE units (0.1758–0.2370), respectively. Both frozen predictive-utility gates pass.

The mechanism-information curve is monotone across the dial. Query utility is broadly
increasing but need not be pointwise monotone because cross-fitted experts add finite-
sample estimation error; primary classification utility peaks at rho=.9 before remaining
positive at rho=1.

## Context sweep

The complete declared development context grid `{32,64,128,256,512}` was run with 120
tasks per cell and no cell filtering. Both metadata gates pass. Low-rho utility remains
null or negative, while high-rho utility is positive across every context. The sweep is
exploratory and does not replace the 420-task primary intervals. Its immutable bundle is
`results/raw/e1_context_sweep_b779842a24_t120_n32-64-128-256-512.npz`.

## Important negative result

The ordinary standardized raw linear learner does **not** beat the rank learner on
average; `rank_loss - raw_loss` stays negative. E1 therefore supports the narrower and
more relevant claim that an explicit marginal channel contains conditional predictive
information that a quotient-only selector cannot access. It does not support “raw values
are generally better.” E3 must still establish a raw/rank/fixed/gated tradeoff under
matched nuisance transforms before M6 is allowed.

## Post-kill independent benchmark replication

Following fallback branch 16L, a benchmark-only replication used 630 fresh tasks per
rho/task, `n=96`, and `d=12`. Marginal mechanism accuracy again moved from chance at
rho=0 (0.127 classification, 0.144 regression) to near-perfect at rho=1 (0.978, 0.987).
Regression marginal utility replicated at +0.2434 MSE [0.2166, 0.2704], but
classification reversed to -0.00402 log-loss utility [-0.00653, -0.00152]. The latter
failure prevents a task-general predictive-utility claim. Full interpretation is in
`reports/FALLBACK_DIAL_CALIBRATION.md`.

Two post-hoc axis controls then localized the classification weakening: changing context
64→96 at `d=8` had no clear effect, while changing features 8→12 at `n=64` reduced
routing utility by 0.00532 [0.00129, 0.00937]. This is exploratory mechanism evidence,
not a new confirmatory gate.

## Figures

- `figures/e1_priordial_phase_v1_1.png`
- `figures/e1_context_surface_v1_1.png`
