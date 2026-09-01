# Frozen protocol: Projective Tabular Foundation Processes

Frozen on 2026-08-31 (Asia/Seoul), before running the new CTR23 outcomes.
The machine-readable source of truth is `config.json`. Any necessary change is
recorded in `PROTOCOL_DEVIATIONS.md`; no outcome-dependent replacement of a
dataset, query family, baseline, or primary endpoint is allowed.

## Question and claim boundary

The primary question is whether a frozen state-of-the-art tabular foundation
model (TFM), which supplies strong one-row predictive marginals, can be lifted
to a compatible joint stochastic process over arbitrary query rows. The lift
must preserve every TFM mean and marginal variance. Consequently, an advantage
over the diagonal control can only come from learned cross-row covariance.

The proposed model is **ProjTabICL**. For context table (C), TabICLv2 produces
a marginal mean (m_C(x)), variance (s_C(x)^2), and a query-row representation
(h_C^{(e)}(x)) for ensemble view (e). A rowwise head (g_\theta) maps each
representation to a unit vector. For a finite query set,

\[
 K_C(x_i,x_j)=E^{-1}\sum_e
 g_\theta(h_C^{(e)}(x_i))^\top g_\theta(h_C^{(e)}(x_j)),
\]

\[
 \Sigma_{ij}=s_C(x_i)s_C(x_j)
 \{(1-\rho_{|C|})\mathbf 1[i=j]+\rho_{|C|}K_C(x_i,x_j)\}.
\]

The backbone is frozen. Only the small covariance head and three context-size
correlation strengths are trained. Head training and all hyperparameter choices
use six development datasets outside OpenML-CTR23. CTR23 labels are never used
to train or select the head.

## Data and splits

- Evaluation uses every one of the 35 tasks in active OpenML suite 353
  (OpenML-CTR23), with task IDs pinned in `config.json`.
- We use official repeat-0 folds 0, 1, and 2. Query rows come only from the
  official test fold. Context rows come only from its training fold.
- Context sizes are 16, 64, and 256. Two nested context draws are used per
  fold. Contexts are nested across sizes within a draw.
- Each evaluation episode contains six disjoint groups of eight query rows.
  The smallest CTR23 test fold has enough rows for all 48 rows.
- Development uses the six explicitly pinned non-CTR23 OpenML datasets, three
  random 70/30 splits, and three context draws. Dataset-level cross-validation
  selects head hyperparameters. The final head is refit on all development
  datasets before CTR23 is opened.
- FreMTPL claim counts, KDD17 stock returns, and bike demand are held out from
  head training for semantically grounded case studies. They do not enter the
  primary gate.

All targets are scored after affine normalization by the mean and standard
deviation of the corresponding official training fold. This normalization is
part of the metric only. Each estimator receives unnormalized labels and its
native preprocessing unless its documented implementation requires otherwise.

## Queries and scores

For each eight-row group we deterministically generate: a point; a subset
mean; an L2-normalized subset total; a normalized pair difference; a normalized
two-group contrast; a random signed unit vector; a random nonnegative unit
vector; and a rescaled copy of the signed vector. Coefficients never depend on
test labels. The six non-point, non-duplicate families are primary.

For every scalar (a^TY), we report Gaussian NLL, Gaussian CRPS, RMSE of the
functional mean, 50/80/90/95% interval coverage, and interval width. The main
estimate first averages episodes and families within a dataset and then gives
each dataset equal weight. The independent dataset is the unit of uncertainty:
we use a paired dataset bootstrap, a paired sign/randomization test, and report
all 35 effects rather than treating query groups as IID datasets.

Point RMSE and point CRPS establish backbone quality. ProjTabICL and its
diagonal TabICLv2 control are required to agree exactly on point means and
marginal variances; point performance therefore cannot be traded for aggregate
performance.

## Baselines

The full matrix contains:

1. TabICLv2 diagonal marginals (the exact controlled backbone);
2. ProjTabICL (the proposed learned projective lift);
3. an untrained hidden-cosine kernel and a raw-feature RBF lift;
4. a PSD row-shuffled-correlation mechanism control;
5. TabPFN-2.5 diagonal marginals;
6. exact RBF/Matérn Gaussian processes;
7. Bayesian linear regression;
8. a bootstrapped CatBoost process (member covariance plus diagonal noise);
9. the earlier 53k-parameter projective/direct models at 16 shots, clearly
   labeled as legacy evidence rather than modern SOTA.

TabICLv2 and TabPFN use eight inference ensemble members. Their marginal scale
temperatures are selected globally per context size on development data and
then frozen. GP kernel choice and CatBoost configuration are likewise selected
on the development panel under the grids in `config.json`; there is no
per-test-dataset HPO. TabDPT-Turbo/TabPFN-3 may be added only as point-prediction
baselines if their official local weights and licenses are accessible; absence
is reported, not silently replaced.

## Primary gates and interpretation

ProjTabICL passes the predeclared covariance claim only if, relative to the
identical diagonal control:

- dataset-balanced mean aggregate NLL and CRPS both improve;
- at least 60% of the 35 datasets improve in NLL;
- the paired dataset-level randomization test for NLL is below 0.05;
- point means and marginal diagonals match to (10^{-10}); and
- restriction/permutation/scaling identity error is below (10^{-5}).

Failure of a gate produces a negative paper conclusion; results are not
filtered. Comparisons with GP, CatBoost, and TabPFN are secondary and are
reported with confidence intervals, not converted into a SOTA claim solely by
pooled averages.

## Integrity and compute

Raw episode predictions, representations, split hashes, timing, package
versions, checkpoint hashes, and exceptions are cached under the explicit
`cache_root`. Each shard is written atomically and is resumable. Audits check
official split disjointness, unique query rows, finite targets, PSD covariance,
exact marginal preservation, deterministic coefficient hashes, complete cell
counts, and absence of test-label access during prediction. Two H100 GPUs may
be used, with models sharded by dataset; statistical analysis is deterministic
on CPU.

## Prior observations, not prospective outcomes

Earlier pilots already observed that a small projective network beat a matched
direct query network on 12 natural datasets, while its post-hoc TabPFN
ensemble-view covariance failed a breadth gate. Several CTR23 dataset
identities overlap those pilots. We therefore do not call CTR23 identities
"unseen to the researchers"; the narrower auditable fact is that CTR23 labels
are excluded from the new head training and HPO. The new protocol tests a
different, frozen-backbone method and reports all suite tasks.
