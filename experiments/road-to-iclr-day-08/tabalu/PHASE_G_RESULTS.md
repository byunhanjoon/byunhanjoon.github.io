# Phase G — Penalized Neural Escape Hatch

Status: **partial: GO for IID flexibility, NO-GO for unguarded magnitude
extrapolation. Retain only as an explicitly shift-aware optional component.**

## Setup

Ten independently generated executable programs, five training seeds, and
targets

\[
y=P(x)+\alpha N^*(x),\qquad
\alpha\in\{0,.1,.25,.5,1\}.
\]

The non-symbolic component combines sine, cosine, and tanh interactions and is
scaled to the program's training-shell standard deviation. Models are the pure
program, pure MLP, unpenalized residual, penalized scalar-gated residual, and
penalized per-row adaptive gate. Fitting targets contain 2% noise; IID and 4×
magnitude tests are clean. Residual usage is the RMS neural contribution divided
by target standard deviation. All 2,500 planned records are finite.

## IID result

| α | Pure program | Pure MLP | Unpenalized | Penalized scalar | Adaptive |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0.0610 | 0.00309 | 0.000036 | 0.0000004 |
| .1 | 0.100 | 0.105 | 0.0362 | 0.0190 | 0.0198 |
| .25 | 0.245 | 0.160 | 0.0622 | 0.0408 | 0.0419 |
| .5 | 0.454 | 0.156 | 0.0906 | 0.0794 | 0.0752 |
| 1 | 0.719 | 0.155 | 0.117 | 0.149 | 0.117 |

The scalar residual's usage is 0.000036 at α=0 and 0.688 at α=1; its
usage–α correlation is 0.989. At α=1 it reduces pure-program IID error by
79.3%. The adaptive gate is 21.4% better than the scalar gate there. All six
prespecified IID checks pass.

## OOD failure

At 4× magnitude, the residual reverses the ordering. For α=1, the pure program
has NRMSE 0.287 because it safely omits the bounded non-symbolic term; adaptive
residual reaches 1.587, scalar residual 2.356, and unpenalized residual 3.851.
Even at α=.1 the pure program (0.0398) beats adaptive residual (0.399). The
learned branch extrapolates its local approximation aggressively, so a penalty
alone is not an OOD safety mechanism.

## Decision

H6's intended continuum exists in distribution and avoids contaminating exact
symbolic targets. It does not preserve the executor's magnitude robustness.
An unguarded residual is therefore excluded from the extrapolating core.
Real-data experiments may include it as a separately reported fallback, but a
shift detector or conservative shutdown rule is required before calling it a
safe escape hatch.
