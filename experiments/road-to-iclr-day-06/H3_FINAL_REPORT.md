# H3 final report — Full-Scale Semantic Arithmetic Closure

Status: **FINAL**

## Verdict

**FALSIFIED AS WRITTEN.**

The frozen 36-bundle, 288-path all-row matrix was allowed to finish despite
logical falsification after six bundles.  The final gate values from
`results/h3_summary.json` are exact IEA64 cells 0/9, material FP32 FT datasets
3/3, stable MLP/ResNet cells 3/6, timing models 3/3, and equal-dataset
canonical test-loss change +0.282%.

The universal claim fails because exact long-horizon IEA64 closure and the
short-run stable-architecture boundary both break.  This is not rescued by
the material-FT, timing, or loss-neutrality gates.

## What the complete matrix establishes

The key counterexample is a long exact prefix followed by a later mismatch.
That is compatible with the rounding-cell condition in Proposition 3 and
incompatible with interpreting it as an indefinite guarantee.  Stored
checkpoint results establish exact or nonexact aligned predictions at those
checkpoints; they do not establish equality of unrecorded intermediate GPU
states.  The separate deterministic one-update CPU audit remains the only
state-level parameter-and-AdamW-moment claim.

FP32 FT paths remain materially schema-sensitive in the required datasets,
but MLP and ResNet are not a universal stable class at 200 epochs.  IEA64's
median path-time ratios are 1.021 for FT-Transformer, 1.154 for MLP, and 1.048
for ResNet.  These timings are a
within-run engineering observation, not a randomized benchmark: the FP32 arm
always ran before IEA64 within a bundle.

## Decision

Discard **arbitrary-horizon exact closure** and **stable MLP/ResNet boundary**.
Keep IEA64 as a causal instrument and evaluate the prospectively frozen
finite-horizon survival (H7) and post-breach attenuation (H9) successors.
Canonical gathering is still the exact, lower-ambiguity intervention whenever
the schema action is known.
