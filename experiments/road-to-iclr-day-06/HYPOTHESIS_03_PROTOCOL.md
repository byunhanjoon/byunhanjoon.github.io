# Frozen protocol H3 — Full-Scale Semantic Arithmetic Closure

Status: **FROZEN BEFORE H3 OUTCOMES**  
Freeze date: 2026-08-28 (Asia/Seoul)

## Hypothesis

H1's interface-local float64 accumulation is a pathwise commutation device,
not an early-training regularizer.  If it removes the only schema-dependent
arithmetic injection, exact closure should persist for arbitrarily many matched
updates.  Conversely, ordinary fp32 FT-Transformer paths should continue to
show macroscopic schema divergence when trained on realistic row counts and
for much longer, unless optimization becomes contractive enough to erase it.

## Frozen matrix

- datasets: Bank marketing, Credit default, FreMTPL claims;
- models: MLP, ResNet, dense-stem FT-Transformer;
- seeds: 8101, 8202, 8303, 8404;
- all available rows, split with the existing fixed split seed;
- three nonidentity exact schema views plus canonical;
- fp32 and IEA64 interface arms;
- 200 fixed epochs, checkpoints 0/1/2/5/10/20/50/100/200;
- unchanged AdamW, batch 256, architecture, dropout, data order, and random
  tape within each pair.

This is 36 bundles and 288 trained paths.  Dataset remains the inferential
unit; seeds/views are repeated measurements.

## Frozen gates

H3 supports realistic-scale persistence if all are true:

1. IEA64 is exactly closed at every checkpoint in at least 8/9 cells;
2. fp32 FT final orbit MSE exceeds `1e-5` in at least 2/3 datasets;
3. fp32 MLP/ResNet final orbit MSE stays below `1e-8` in at least 5/6 cells;
4. median IEA64 path fit time is no more than 25% above fp32 within each model;
5. the equal-dataset mean canonical IEA64-vs-fp32 relative test-loss change is
   inside [-1%, +1%].

Gate 5 tests systematic harm, not per-seed equality: the canonical arithmetic
choice can select a different chaotic FT path.  Failure of gate 5 rejects IEA64
as a default practical intervention but does not falsify its closure mechanism.

## Integrity

Every bundle is write-once/resumable, stores predictions at all checkpoints,
and records source/config hashes and path timing.  TF32 is disabled and
deterministic algorithms are enabled.  No early stopping or outcome-dependent
budget change is allowed.
