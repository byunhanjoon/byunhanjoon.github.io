# Frozen protocol

Protocol date: 2026-08-31 (Asia/Seoul)  
Repository base commit: `0c456660ae9a87aab7932b569e1954b0ee1d25fe`

The adjacent long-form `AGENT.md — ...` file is the authoritative protocol. This
document records operational choices made before observing Day-09 outcomes.

## Estimand and order of work

The estimand is expected predictive risk under a declared mixture of task priors and
matched featurewise, strictly increasing reparameterizations. Clean performance,
matched-transform disagreement, and compute remain separate outcomes.

Execution follows the mandated gates: E0/E1/theory, then the E2 audit, then M0–M5.
M6/M7 are allowed only if the oracle gate has meaningful headroom and the learned gate
beats the frozen fixed-mixture comparator. Confirmatory synthetic configs are immutable
after method selection. Real-data configs are frozen before their outcomes are read.

## Primary synthetic choices

- Mechanisms: sparse linear, additive smooth, threshold, pairwise interaction,
  shallow partition, and periodic.
- Tasks: binary classification and regression.
- Dial: `rho = [0, .1, .25, .5, .75, .9, 1]`.
- Main cells: `n_context in {64, 128}`, `d in {8, 16}`.
- Development and confirmation use disjoint seeds and transform parameter ranges.
- The held-out confirmation transform is monotone spline; it is not used for model
  selection.
- Independent task seed is the synthetic unit of analysis.

The coupling scheduler balances mechanism and warp marginals exactly within each rho
cell. Thus rho changes dependence, not marginal warp prevalence.

## Primary comparisons

Synthetic: raw, robust-affine, rank, transform augmentation, 50/50 raw+rank,
development-tuned fixed mixture, learned non-oracle gate, oracle gate. Real:
proposed-vs-host, rank, fixed two-view mixture, and equal-view transform augmentation.

## Statistical freeze

- Synthetic main cells target at least 400 independent tasks and report task bootstrap
  95% intervals; pilot smoke cells may be smaller and are labeled exploratory.
- Real data aggregates transforms within split and split within dataset before a paired
  dataset bootstrap with 10,000 draws.
- Query labels are unavailable to preprocessing and gating.
- Failures and protocol changes are appended to `reports/DECISION_LOG.md`.

## Resource rule

Two NVIDIA H100 NVL GPUs are available, but root storage has only about 12 GiB free at
freeze. No 100-GB TabArena raw artifact will be downloaded to root. Large immutable
artifacts require a validated alternate volume; until then, results-only leaderboard
artifacts and locally generated prediction bundles are allowed.

