# Neural-base transfer certificate — results

Status: **COMPLETE — DEVELOPMENT GATE FAILED, SELECTOR SIGNAL SURVIVES**

## Executive verdict

The Geometry Transfer diagnostic is not yet shown to transfer cleanly from a
CatBoost base to a neural base. Source-by-operator predicted-versus-actual
Spearman correlation is `0.574`, below the frozen `0.60` gate, and sign
accuracy is `63.9%`, below `75%`.

The conservative decision rule nevertheless produced a useful narrow result.
It selected geometry in 4/8 task-split cells, had zero harmful cells, no
negative source mean, and a source-balanced gain of `+0.01197` standardized
MSE. The ordinary positive-mean selector selected 5/8 and also had zero harmful
cells, with gain `+0.01266`.

This supports a fresh experiment on *certification/abstention*, but it does not
support claiming calibrated operator-level prediction for neural bases.

## Frozen gates

| Gate | Result |
|---|---:|
| source×operator Spearman at least 0.60 | **FAIL** — 0.574 |
| source×operator sign accuracy at least 75% | **FAIL** — 63.9% |
| pessimistic source-balanced gain positive | PASS — +0.01197 |
| no source below -0.002 | PASS — minimum 0.0 |
| pessimistic no more harmful cells | PASS — 0 versus 0 |
| integrity | PASS |

## What was selected

The pessimistic rule selected kernel-ridge transfer for both ACS occupation
splits and both Medical Charges splits. Their actual gains were `+0.02037`,
`+0.01277`, `+0.04251`, and `+0.02011`. It abstained on both airline splits
and both TLC splits. The mean rule additionally selected TLC split 1, which was
beneficial by `+0.00550`.

The operator-level failure is concentrated in conservative underprediction on
TLC: most neighborhood operators helped on the outer states while their inner
state-fold gains were negative. This is evidence that the *state-split law*
matters as much as the fixed-operator risk identity. Exchangeable-state CV is
not a harmless technical detail.

## Interpretation for the paper search

Three claims must be kept separate:

1. the fixed-base, fixed-operator squared-risk identity is exact;
2. state-held-out validation may estimate future gain only under a justified
   state sampling design;
3. a conservative selector can still be practically safe even when the full
   ranking is imperfect.

The promising ICLR question is therefore whether a neural tabular learner can
expose a *certifiable optional inductive bias*—with explicit abstention and
shift assumptions—not whether one more metric embedding wins a benchmark.

## Evidence label and limitations

This is development evidence on four previously studied sources, two state
splits, one fixed MLP, and fixed training settings. It uses row-OOF residuals
but not fully nested refitting of preprocessing/base learning inside every
state fold. The result must not be presented as prospective, architecture-wide,
or distribution-free safety.

