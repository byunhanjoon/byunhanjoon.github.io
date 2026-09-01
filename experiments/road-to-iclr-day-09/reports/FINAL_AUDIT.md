# Day-09 execution audit

Status: **the original M6 program stopped at its predeclared Stage-B kill gate; the
authorized theory/benchmark fallback completed, and a separately frozen loss-alignment
opportunity produced a confirmed outer-fold-normalized numeric-regression transfer
result.**

## Original program

- E0 hardened the Day-8 four-way protocol on TabICLv2 and Mitra: 72 completed jobs, 48
  explicit TabPFN-3 unavailable records, and zero silent substitutions.
- E1 froze and validated `prior_dial_v1_1` as a shape-information dial with a 420-task
  primary cell and the declared context sweep. The ordinary raw learner remained worse
  than rank, and that negative was preserved.
- T1–T4 provide the quotient-risk identities, counterexample, and exact PriorDial mutual
  information calibration. T5 proves that perfectly informative metadata can still make
  matched routing worse under miscalibration; T6 records the convex-mixture/Jensen fact.
- E2 completed 252 model episodes / 756 host fits. The scoped TabICLv2-vs-Mitra G2 gate
  passed; TabPFN-3 stayed access-incomplete.
- E3 completed five immutable runs totaling 94,920 episodes and 379,680 expert fits. All
  inspected tests were replaced by fresh seeds in the next failure-branch iteration.

The final E3 test had clear oracle headroom but captured only 0.27% in classification and
16.96% in regression. Classification learned-vs-fixed included zero and the required
raw/rank crossover never appeared, so G3 failed. No M6 model was frozen and E4–E10 were
not launched. Their reports explicitly record `not authorized`/`not launched`.

## Fallback benchmark result

A fresh 8,820-episode run independently replicated the fixed-marginal information dial.
At rho=1, shape routing reduced regression MSE by 0.24339 [0.21664, 0.27044] but increased
classification log loss by 0.00402 [0.00152, 0.00653]. Classification mechanism selection
was nevertheless 99.2% accurate, exposing the central objective-alignment failure:
identifying the data generator is not sufficient for loss-optimal expert routing.

The post-hoc 2x2 axis control found no clear context-size effect in classification, while
8 to 12 features reduced routing utility by 0.00532 [0.00129, 0.00937]. Regression stayed
strongly positive in all four cells. These diagnostics do not reopen the killed method.

## Separate loss-alignment opportunity

A context-only three-fold loss router reused the same six frozen experts. Hyperparameters
were chosen on 4,800 development episodes, then evaluated once on 9,600 untouched
episodes. Relative to the development-tuned fixed mixture, it improved classification
log loss by 0.005465 [0.004797, 0.006129] and regression MSE by 0.254185
[0.246516, 0.262313], capturing 29.51% and 99.66% of best-individual headroom.

Mechanism controls distinguish four targets rather than crediting a generic selector:

- soft weighting beat hard context-CV selection by 0.023797 classification log loss and
  0.024655 regression MSE; hard selection itself harmed classification;
- cyclically assigning the same CV-loss spectra to the wrong experts made performance
  worse on synthetic classification/regression and on all real task panels;
- the benefit was largest when hard-selection margins were small or the hard expert was
  wrong, consistent with calibrated aggregation rather than reliable expert recovery.

Competence estimation, cross-validated stacking, dynamic selection, and tabular mixtures
are established prior art. The remaining novelty candidate is the controlled separation
between metadata identification, individual-expert choice, correct loss-to-expert
assignment, and calibrated mixture prediction under an exact fixed-marginal information
dial—not a new ensemble algorithm.

## Real-data transfer

The frozen synthetic-tuned rule was evaluated without real-data tuning:

- the initial 3-classification/4-regression numeric panel was inconclusive in both tasks;
- on 13 unseen OpenML identities, regression improved by 0.29103
  [0.02525, 0.94647] across seven datasets, while classification worsened by 0.00213
  [0.00026, 0.00465] across six;
- a deterministic five-dataset regression panel independently confirmed +0.10045
  [0.00173, 0.25055], with four positives and Auction Verification preserved as negative;
- a retrospective all-panel synthesis over 16 regression identities gave +0.217965
  [0.041682, 0.459988], with 14/16 positive, positive median and trimmed mean, and every
  leave-one-dataset-out mean positive. The nine-dataset classification synthesis was
  -0.000266 [-0.002976, 0.003167], with only 3/9 positive.

The supported external-performance result is therefore scoped to numeric regression.
It does not cover categorical inputs, multiclass tasks, generic tabular superiority, or
full synthetic-only classification transfer. An immutable-prediction diagnostic localized
the unseen binary failure: AUC was unchanged, the bottom 90% of pointwise NLL was neutral,
but worst-decile
NLL worsened by 0.02512 [0.00850, 0.04525] and the NLL>2 rate increased. This is a
rare-confident-error problem rather than broad ranking collapse. Regression is the
opposite: worst-decile squared error improves by 1.88738 [0.22577, 5.03685], and the
SE>4 rate falls. Larger weight KL consistently predicts worse classification tails
(Spearman -0.3524), but has no monotone regression association; regression benefits from
large shifts only on high-headroom datasets. The post-result real shrinkage path then
suggested lambda=0.1. On a separately frozen deterministic five-dataset CC18 panel, this
real-development-tuned candidate improved log loss by 0.000600 [0.000038, 0.001471] and
Brier by 0.000207 [0.000001, 0.000527], passing its 3/5-dataset confirmation gate. This
is a modest new numeric-binary result, not restoration of synthetic-only classification
transfer. A context-size diagnostic found a positive level effect from 32–192 rows but no
supported scaling slope. Two-fold CV saved 25% of standalone fits and remained better
than fixed, but failed its noninferiority margin; three folds remains the supported default.

A final reviewer-style audit noted that the OpenML protocols fit affine scaling on the
full official training fold before context sampling. This was predeclared and used no
query labels, but is weaker than strict few-shot scaling. On fresh context-rescaled runs,
10% classification shrinkage passed again (+0.000732 [0.000142, 0.001680], 5/5 positive),
while regression retained +0.09290 and 4/5 positives but its interval crossed zero.
Regression confirmation is therefore scoped to outer-fold normalization.

## Reproducibility and integrity

- `results/MANIFEST.jsonl` contains 150 valid JSON records and 49 unique referenced raw
  or processed artifacts.
- `scripts/audit_manifest.py` verifies that every referenced artifact and config exists,
  every current config hash matches its recorded SHA-256, and every numeric NPZ array is
  finite. It also rejects duplicate declared run keys and episode-dimension mismatches;
  the final audit reports zero failures in all categories.
- `scripts/audit_claim_state.py` asserts the documented positive and negative gate state,
  including the failed strong-transfer and context-rescaled regression gates.
- All 27 processed JSON audits parse under strict rejection of NaN/Infinity tokens, and
  every numeric value in the seven new synthesis/tail/confirmation detail tables is finite.
- Recent raw/processed shapes agree: 9,600/57,600 synthetic test episodes/cells; and
  840/4,200, 520/2,600, 300/1,500, 600/3,000, and 200/800 episodes/cells for the five
  real transfer, confirmation, scaling, and CV-budget runs. The independent classification
  shrinkage run adds 400 episodes / 2,000 parent cells / 9,600 expert fits.
- The context-rescaled robustness run adds 500 episodes / 2,500 parent cells / 12,000 fits.
- The new shrinkage analyzer reproduces parent fixed/full log losses within `7.93e-9`;
  its raw bundle SHA-256 is `1a2f79541ec1...e073f`.
- The context-rescaled analyzer reproduces parent losses within `3.24e-7`; its raw
  bundle SHA-256 is `7c51679617ff...9cf7`.
- Because Day 09 is untracked, final executable/config SHA-256 hashes are recorded in
  `reports/CODE_HASHES.md` rather than relying only on the parent Git commit.
- The loss-router development/test SHA-256 hashes are `99d23048...397979` and
  `004e5acd...b6b51`; immutable-prediction diagnostics reproduce parent losses within
  their declared float-storage tolerances.
- Final unit tests: **25 passed**, including exact outer-to-context affine-cancellation
  checks. Every script under `scripts/` and `src/` compiles.
- Day-09 artifacts occupy approximately 358 MiB. The only new external data were the five
  compact CC18 confirmation datasets added to the existing OpenML cache; no broad benchmark
  mirror was downloaded.

## Authorized claims

The evidence supports: an exact conditional-marginal information construction; a scoped
TabICLv2/Mitra stability contrast; replicated regression utility from explicit marginal
shape; a synthetic four-target loss-alignment result; and confirmed, sensitivity-robust
numeric-regression transfer for the frozen loss-aligned soft mixture under outer-fold
normalization. It also supports a small independently confirmed numeric-binary gain from
a real-development-tuned 10% adaptation step, including context-rescaled robustness. It
explicitly does not support the killed M6 method, a novel routing
or shrinkage algorithm, synthetic-only classification transfer, task-general real
transfer, TabPFN-3 coverage, clean TabArena preservation, matched-transform gains, or broad
external validity.
