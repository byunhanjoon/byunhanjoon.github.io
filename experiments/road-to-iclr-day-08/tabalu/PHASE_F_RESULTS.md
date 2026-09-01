# Phase F — Heterogeneous Typed Execution

Status: **GO on the matched synthetic library; retain typed operators for the
real-data test, without claiming general typed program discovery.**

## Setup

Sixteen independently generated tasks and five training seeds combine two
continuous variables, a three-level category, a four-level ordinal variable,
and timestamps. Every target contains numerical arithmetic, a
category-conditioned computation, an ordinal operation, intraday/weekday
periodicity, and a long-term time term. Training uses calendar 2020 and 1×
continuous magnitudes; the future test uses July–December 2021 and 4×
magnitudes, with 1% target noise used only for fitting.

The full model selects a bounded-cardinality affine program from deterministic
typed features. Controls are learned embeddings, conventional manual
preprocessing followed by an MLP, and typed programs with ordinal, time, or
categorical-condition families removed. All 960 planned records are finite.

## Result

| Model | IID NRMSE | Future 4× NRMSE |
|---|---:|---:|
| Full typed program | 0.00088 | 0.00102 |
| Typed without ordinal | 0.408 | 0.202 |
| Typed without time | 0.435 | 0.166 |
| Typed without category conditions | 0.279 | 0.383 |
| Manual preprocessing + MLP | 0.164 | 0.559 |
| Learned embeddings | 0.310 | 0.711 |

The full typed model averages 6.8 selected operations. Relative to it, the
future error increases 198× without ordinal operations, 163× without time
operations, and 375× without categorical conditions. Every frozen gate passes.
Task-cluster bootstrap intervals and raw results are in
`results/phase_f_typed/`.

## Interpretation

H5 survives this matched-library synthetic test. Explicit type semantics make
the relevant computation sparse and preserve periodic and arithmetic behavior
under joint temporal and magnitude shift. Treating types as learned embeddings
or merely feeding sensible features to an MLP does not recover that behavior.

This is intentionally not a general discovery claim: the generator draws from
the same typed library searched by the model, category cardinalities are tiny,
timestamps are clean, and the neural budgets are finite. Typed execution stays
in the architecture only provisionally, pending heterogeneous real-data
results.
