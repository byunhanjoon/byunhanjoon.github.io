# Phase C — Operand Inference Result

Status: **NO-GO; omit the operand estimator from the current main model.**

## Setup

Twelve independent programs, five training seeds, fixed ground-truth executable
graphs, training noise of 0.10 feature SD, and test noise
`{0, .05, .10, .20, .40}` on IID and 8× inputs. The required variants were raw
operands, bounded correction, confidence-gated reconstruction, and an
unrestricted encoder. Conservative corrections carried an explicit penalty;
the bounded variant cannot change a feature by more than 0.30 SD.

All 2,400 planned prediction records are present and finite.

## Result

At IID noise 0.20:

| Variant | NRMSE | Normalized correction RMS |
|---|---:|---:|
| Raw | 0.495 | 0.000 |
| Bounded correction | 0.470 | 0.026 |
| Confidence gated | 0.461 | 0.067 |
| Unrestricted encoder | 0.470 | 0.034 |

The best conservative improvement was only 7.0%, below the preregistered 15%
requirement. It also damaged clean IID prediction from exact to 0.113 NRMSE,
above the 0.05 limit. On clean 8× inputs, bounded correction reached 0.038,
unrestricted encoding 0.104, and confidence gating 0.582, versus 0.000 for raw
operands. The unrestricted encoder did not reveal hidden robustness.

## Interpretation

With independent latent operands and one noisy observation per feature, there
is little row-wise information from which to reconstruct the exact latent value.
The networks learn small task-directed corrections that modestly help noisy IID
data but introduce systematic bias and extrapolate poorly. Confidence remains
about 0.914 rather than adapting materially with noise.

This fails H2's usefulness criterion. No operand estimator will be included in
the combined architecture based on this evidence. A future revival would need
an identifiable setting—repeated measurements, temporal continuity, or known
measurement-error structure—and a clean-data identity gate.
