# H1 pilot decision

Decision time: 2026-08-28 (Asia/Seoul)  
Decision: **ADVANCE TO THE FROZEN SIX-SEED CONFIRMATION**

All four frozen pilot gates pass on the complete 3-dataset × 3-model × 2-seed
matrix (18 paired bundles, 144 trained paths):

- IEA64 has lower final prediction-orbit MSE in 9/9 cells (gate: at least 7);
- equal-cell geometric mean reduction is 100% (gate: at least 90%);
- every IEA64 schema path is bitwise identical to its own canonical reference,
  so the maximum relative path loss change is zero (gate: at most 0.1%);
- all three FT-Transformer cells amplify the fp32 semantic roundoff seed by
  more than `10^4` (gate: at least three cells).

The result is mechanistically concentrated.  At epoch 20 the mean fp32 orbit
MSE is `3.93e-3`, `4.55e-4`, and `6.91e-2` for FT-Transformer on Bank,
Credit, and FreMTPL respectively.  MLP and ResNet remain near numerical zero
(`4.20e-16` to `5.36e-13`), consistent with the Day-5 closure.  Thus H1 is not
yet evidence that ordinary predictive performance improves; it is evidence
that interface accumulation order can be the entire source of a large
architecture-specific schema orbit.

One non-gated caveat is retained before confirmation: changing the canonical
FT-Transformer interface arithmetic also selects a different chaotic path.
Across the six pilot FT seed×dataset references, relative test-loss changes
range from -1.53% to +3.00%.  IEA64 guarantees semantic reproducibility within
a path coupling; it does not guarantee that the selected canonical path is
more accurate.  Confirmation must keep quotient variance and ordinary test
loss separate.

No config, model, dataset, view, checkpoint, or confirmation seed is changed
after this decision.
