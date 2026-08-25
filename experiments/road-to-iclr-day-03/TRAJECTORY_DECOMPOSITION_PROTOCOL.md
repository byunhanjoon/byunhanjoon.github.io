# Day 3 extension — function-matched trajectory decomposition

## Frozen question

Day 3 established that scale-controlled invertible basis changes can harm neural
tabular training. This extension asks a narrower mechanistic question:

> When two representations contain exactly the same information, how much of
> the final performance gap comes from different initial functions, and how
> much comes from optimizer-induced divergence after the functions are matched?

The protocol is frozen in
`experiments/day3/configs/trajectory_decomposition_preregistered.json` before
any trajectory outcome is generated.

## Paired intervention

For row-vector inputs `X_changed = X_reference B` and a first affine layer with
weight `W`, the changed model is function matched by setting

`W_changed = W_reference B^{-T}`.

All other parameters and biases are copied exactly. The two models then have
the same deterministic predictions at update zero. Their batches and dropout
masks remain paired. Any subsequent difference in predictions under AdamW is
therefore caused by the training rule rather than missing information or a
different initial function.

The primary measurement is symmetric normalized RMS prediction drift:

`drift = RMS(f_reference - f_changed) / max((RMS(f_reference) + RMS(f_changed))/2, 1e-8)`.

It is recorded on fixed train and validation probes at updates
`0, 1, 5, 20, 100, 200`. Models train for exactly 200 updates; validation
outcomes do not select checkpoints.

## Arms and interpretation

1. **Ordinary initialization + AdamW:** reproduces total basis sensitivity.
2. **Function-matched initialization + AdamW:** isolates optimizer sensitivity.
3. **Covariance-metric initialization + AdamW:** tests a deployable
   initialization intervention that does not know a hidden reference basis.
4. **Function-matched initialization + input-natural first layer:** closure
   control for initialization and update geometry together.
5. **Ordinary initialization + input-natural first layer:** checks whether an
   invariant update is sufficient when initialization is not invariant.

The experiment covers Adult, California Housing, and Diamond; MLP and ResNet;
five seeds; controlled condition numbers 1, 30, and 3000; and the exact natural
cumulative/Helmert to local/adjacent representation pair.

## Decision rule

The strongest desired result is not merely another significant endpoint gap.
It is a held-out-dataset result in which early function-space drift predicts
final paired harm better than condition number. Leave-one-dataset-out
predictions, paired bootstrap intervals, exact cell coverage, and failed cells
will all be reported.

If function matching removes most sensitivity, the interpretation must pivot
toward initialization. If matched AdamW still diverges and early drift predicts
final harm, optimizer trajectory geometry is supported. If neither occurs,
the diagnostic is rejected rather than promoted as a method.

