# H3 early logical falsification

Status: **PRIMARY H3 CANNOT PASS; MATRIX CONTINUES FOR FROZEN H6/H7 TESTS**  
Decision time: 2026-08-29 ~00:31 Asia/Seoul

Six of 36 H3 bundles are complete.  This is enough to prove that two frozen H3
gates are unattainable, regardless of the remaining outcomes.

## Exact-closure gate

H3 requires exact IEA64 checkpoint-prediction closure in at least 8/9
dataset×model cells.  Two distinct cells already fail:

- Credit / FT-Transformer / seed 8101: one view is nonzero by epoch 50; all
  three are material by epoch 200;
- Credit / ResNet / seed 8101: one view is nonzero by epoch 100 and material by
  epoch 200.

Cell aggregation uses the maximum across seeds/views/checkpoints, so later
seeds cannot restore either cell.  At most 7/9 cells can now be exact.  Gate 1
is logically failed.

## Stable-control gate

H3 requires at least 5/6 FP32 MLP/ResNet cells below `1e-8` at epoch 200.  Two
distinct ResNet cells already fail:

- Bank / ResNet / seed 8101: mean final orbit MSE `2.32e-2`;
- Credit / ResNet / seed 8101: mean final orbit MSE `2.44e-2`.

Later seeds cannot reduce the cell maximum, so at most 4/6 cells can satisfy
the gate.  Gate 3 is logically failed.

## Scientific change

H3 falsifies both “IEA64 is exact indefinitely” and a universal
FT-Transformer-versus-MLP/ResNet architecture boundary.  The surviving
mechanism is finite-horizon **rounding-cell survival**:

- Credit FT IEA64 is exact through epoch 20, then fails at view-dependent
  checkpoints;
- Credit ResNet IEA64 delays all three material crossings by roughly 100
  checkpoint epochs versus FP32 and lowers all three final MSEs, but one view
  still fails by epoch 200.

This motivated prospectively frozen H7 on the other 31 bundles.  H6 also
correctly flags Credit ResNet as unstable from epochs 5/10/20 despite its raw
epoch-20 MSE being only `4.78e-14` on average: its extrapolated screen score is
`5.15`, above the fixed `-5` decision threshold.

## Why H3 continues

Stopping now would invalidate or shrink the already frozen H6/H7 prospective
test sets and conceal the prevalence of the failure.  The remaining bundles
therefore continue unchanged.  H3 will receive a final complete report, but it
cannot be labeled supported even if its other gates later pass.
