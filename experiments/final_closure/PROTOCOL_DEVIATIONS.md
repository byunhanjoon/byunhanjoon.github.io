# Final-closure protocol deviations

No deviations recorded at freeze time.

Any implementation correction made after the protocol/config hashes are
written must be appended here with its discovery time, affected cells, reason,
repair, and proof that outcome-dependent choices were not introduced.

## D1 — independent joint-pool cardinality correction

Recorded immediately after the first Experiment-A manifest was emitted and
before any prediction residual or scientific outcome was inspected.  The
frozen protocol correctly defines the independent joint pool as eight fresh
master seeds for every *schema* action, but its following arithmetic sentence
incorrectly calls the earlier 64/128 schema×finite-init×finite-order products
"schema actions."  Schema alone has at most 16 regression actions and 32
binary-classification actions (4 feature × up to 4 category × 1/2 target),
with smaller valid menus when the category factor collapses, so the usual
frozen construction contains 128/256 joint fits rather than 512/1,024.  The 128-fit
canonical-independent pool and every method/budget/512-draw analysis remain
unchanged.  This correction removes the two finite coupled RNG factors from
the independent-RNG estimand exactly as required by Experiment A; expanding
them would silently reintroduce the finite seed menu the experiment is meant
to replace.  No dataset, model, outcome, metric, gate, or analysis changed.

## D2 — checkpointed trajectory reuse and small/20 verification

Recorded before Experiment B execution or outcome inspection.  The literal
first implementation would independently retrain the same action/RNG path to
20, 50, 100, and 200 epochs and again for convergence.  The final runner uses
one common master seed per action across optimization budgets, checkpoints the
identical path at all four fixed epochs, and continues to the prospectively
defined convergence stop.  A path continues through the largest fixed endpoint
that actually consumes it: epoch 200 for the common strength-3 trajectory and
epoch 20 for additional full-product corner rows.  This removes duplicate
prefixes without changing any endpoint, recipe, stopping rule, or prediction.
At small-N/20, the checkpoint is
verified against the retained Experiment-A prediction wherever the physical
fit is shared; Experiment-A values remain the scientific tensor and the longer
trajectory supplies the missing epoch telemetry.  No verification or
checkpoint selects a dataset, action, model, metric, or gate.

The same checkpoint-prefix reuse is applied prospectively to the mandatory
matched-function 20/100/convergence repeat.  A common controlled master seed
and path are used for all three endpoints so that “over convergence” denotes
the same ordinary or exactly matched trajectory, rather than three unrelated
random paths.

## D3 — B=64 support for collapsed schema menus

Recorded before Experiment-A analysis or any prediction-residual outcome was
examined.  Three numerical regression sources have four valid schema actions,
while two classification sources have eight.  The frozen minimum of eight
independent master seeds per action therefore supplies only 32 or 64 distinct
joint fits, which either cannot support the already frozen B=64 comparison or
degenerates it into a full census with artificially perfect schema balance.
The neural joint cache is therefore extended deterministically to 256 physical
fits in every cell: 64/32/16/8 seeds per action for 4/8/16/32-action menus.
This is a prospective finite-cache repair, not a change to the declared
estimand: cached estimator draws remain the protocol's explicitly conditional
finite-pool estimates, every draw has 64 non-repeated physical fits, and direct
multinomial calibration showed the retained-draw probability is at least
99.47% for every menu (effectively 100% for the smaller menus), versus 73.76%
in the unextended 16-action case.  Existing masks and predictions are preserved,
canonical-extension fits are shared with the canonical-schema slice rather
than retrained, and all new seeds use the frozen domain-separated derivation.
Dataset, model, method, outcome, metric, aggregation, and decision gates are
unchanged.

## D4 — repeated rows in collapsed strength-3 trajectories

Recorded before Experiment B execution or outcome inspection.  A 128-run
strength-3 schedule cannot have 128 distinct factor-level rows when collapsed
factors leave only 64 classification or 32 regression combinations.  The
maximum-row-uniqueness assertion is corrected to the attainable
`min(128, product(cards))`.  Repeated factor rows remain 128 distinct physical
training paths, keyed by their within-condition occurrence and assigned fresh
master seeds; they are not collapsed or overwritten.  This preserves the
frozen 128-run schedule, formal strength-3 balance, fixed factor levels, and
endpoint analyses without changing any scientific selection or gate.

## D5 — preserve pre-cap training-index order for exact small-N reuse

Recorded after the first Experiment-B verification path failed and before any
B prediction was accepted, registered, manifested, or analyzed.  The initial
full-split helper sorted the full training partition before reapplying the
completion panel's deterministic 2,048-row cap.  Because sklearn's seeded
split acts on input order, this produced a different small subset despite the
same split seed.  The helper now retains the original train-index order from
the two frozen split calls, then applies the exact completion cap; validation
and test caps are unchanged.  Array-level tests prove the resulting small
numeric, categorical, and target partitions equal Experiment A exactly.  This
repairs protocol compliance and nested reuse; no B outcome passed the guard or
influenced the repair.

## D6 — reject constrained MPS; validate unconstrained MPS exactly

Recorded during Experiment-B and matched-convergence execution, before any
closure outcome was analyzed.  A prospective attempt shared each H100 NVL among
five independent training processes through NVIDIA CUDA MPS with
`CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=20`.  The frozen A-to-B checkpoint identity
guard rejected the attempt: MPS changed floating-point execution, producing
prediction gaps from `1.88e-6` to `8.96e-1`, including failures above the
`2e-6` tolerance.  That constrained mode was stopped.  All B and matched raw
artifacts and registry rows from both before and during the attempted switch
were quarantined together, and those experiments were restarted from an empty
state under ordinary CUDA scheduling.

Before the large-N tail, a separate execution-only test started the default MPS
server with **no** active-thread-percentage constraint.  Fresh epoch-20 A-to-B
replays for MLP, ResNet, FT-Transformer, and TabM were bit-for-bit identical in
validation and test predictions (maximum gap exactly `0.0`), first with two and
then with five simultaneous clients per GPU.  A further 24-replay stress test
(12 validation clients per GPU alongside the six active workers) also had a
maximum gap of exactly `0.0`.  The closure therefore uses this
validated default MPS mode for the remaining restartable work; every still-
missing small-N A checkpoint continues to pass the frozen identity guard.  No
failed-run prediction, dataset, action, model, metric, or gate was used to alter
the frozen scientific design.

## D7 — deterministic intra-cell path sharding

Recorded before intra-cell sharding was used and without inspecting a B
scientific outcome.  The long large-N tail is executed by three processes per
dataset×model cell.  Physical joint-path and canonical-path indices are assigned
disjointly by `index mod 3`; a file lock serializes creation/opening of the fixed
memmaps, a joint-mask barrier prevents races with the eight shared small-N
canonical paths, and shard zero writes manifests only after every canonical mask
is complete.  This changes only scheduling: the path indices, actions, seeds,
checkpoints, predictions, registry keys, and represented-fit counts are exactly
the frozen set.  The initial schedule uses three shards per cell; D9 records the
targeted FT-Transformer extension.  An unsharded invocation remains the
`path_shards=1` default.

## D8 — Experiment-D matrix resharding

Recorded before the eight-way D schedule was used and without inspecting a D
scientific outcome.  The already frozen 48 dataset×split×model cells are assigned
disjointly by matrix index modulo eight, giving two workers per architecture.
No cell is shared between workers, and the existing per-fit completion masks
remain the restart boundary.  This changes only parallel scheduling; D's finite
products, seed menus, fit budget, methods, and analysis are unchanged.

## D9 — targeted FT-Transformer path resharding

Recorded before the six-way FT schedule was used and without inspecting a B
scientific outcome.  Because FT-Transformer is the critical large-N execution
tail, its six dataset cells use six disjoint path owners rather than three;
ownership is the same `index mod path_shards` rule and the same locked memmaps
and mask barriers from D7.  Other architectures remain three-way.  This changes
only scheduling and leaves every physical fit, seed, checkpoint, prediction,
registry key, and manifest count unchanged.

## D10 — telemetry-balanced B shard counts

Recorded from execution timing and GPU utilization only, without inspecting a B
scientific outcome.  The 90-client schedule in D9 oversubscribed the GPUs and
reduced aggregate completed-path throughput.  The retained schedule uses 72 B
clients, with path-shard counts MLP=2, ResNet=3, FT-Transformer=4, and TabM=3
per dataset cell.  These counts balance observed architecture compute cost while
remaining below the validated default-MPS capacity.  Ownership still uses the
D7 modulo rule and barriers; the represented paths, seeds, checkpoints,
predictions, registry keys, and manifests are unchanged.

## D11 — four-worker FT-Transformer tail for Experiment D

Recorded before the 16-way D tail schedule was used and without inspecting a D
scientific outcome.  After all MLP/ResNet/TabM cells completed, the remaining
FT-Transformer cells were split by matrix index modulo 16, yielding four
disjoint cell workers rather than two.  Existing completion masks are reused;
no D cell is shared or changed, and the finite products, seed menus, fit budget,
methods, and analysis remain frozen.

## D12 — adaptive FT canonical-barrier allocation

Recorded from completion masks, timing, and GPU capacity only, without inspecting
a B scientific outcome.  After the Abalone FT cell completed, the retained FT
tail uses six path owners for Bank/Heloc/KDD and ten for Credit/Fremtpl.  This
balances the two physical GPUs and parallelizes the 64-path canonical barriers
while remaining below the validated MPS client capacity.  The D7 modulo
ownership, locked memmaps, and barriers are unchanged, as are all paths, seeds,
checkpoints, predictions, registry keys, and manifests.
