# Protocol Deviations

The normative protocol was frozen at `2026-08-29T10:12:15+09:00`; hashes are
in `PROTOCOL_HASHES.txt`. Add entries chronologically. Never edit or erase an
earlier entry.

## At freeze

No implementation deviations.

The optional MIMIC-III task is prospectively classified as `NOT RUN —
CONTROLLED ACCESS UNAVAILABLE` because no PhysioNet credentials or local data
were present. This is the governing brief's explicit non-blocking branch, not
an outcome-driven deviation.

The equal HPO budget is prospectively eight trials rather than the suggested
twenty. The choice predates all new outcomes and applies identically to every
representation. It is a design limitation, not a later deviation.

## 2026-08-29 — source-schema audit, before target modeling

The frozen Amazon file was acquired with its published SHA-256
`14815cf1312a0d847364866e6876c8c73738993469067242870774c372c04387`.
All 112,590 rows have an empty `categories` sequence. Although 17,704 rows have
a finite positive price, zero rows jointly satisfy the frozen price and
hierarchy eligibility rule. `amazon_leaf_category` is therefore retained in
the panel as `NOT RUN — REQUIRED SOURCE SCHEMA UNAVAILABLE`; it is not replaced
by a different Amazon subset after this discovery.

The frozen OpenML versions 41444 (Medical Charges) and 41442 (Open Payments)
are inactive. Medical Charges is read from its active canonical version 42130,
which exposes the declared `drg_definition`, `average_total_payments`, provider
state, and average-covered-charge columns. This is a source-version correction,
not a task or outcome change. The active Open Payments version 42738, like the
frozen snapshot, omits `Total_Amount_of_Payment_USDollars`. Because that amount
was prospectively declared a required ordinary covariate, `open_payments` is
retained as `NOT RUN — REQUIRED SOURCE SCHEMA UNAVAILABLE` unless the exact
original public benchmark table can be recovered and checksummed without
changing the task.

## 2026-08-29 — ridge implementation clarification, before target fitting

`final_config.json` freezes an eight-trial budget but inadvertently lists only
the neural search factors. The ridge mechanism view therefore uses the fixed,
shared eight-value regularization grid
`[0, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]` for every representation. Each of
the seven frozen bandwidth candidates is screened at ridge `alpha=1` using
unseen validation states; the selected bandwidth is then reused by MPE,
Similarity Encoding, RBF, Nyström, and applicable specialist features. This
clarification was recorded before fitting any real target.

## 2026-08-29 — tree-model implementation clarification, before tree fitting

The frozen files require eight equal trials for CatBoost and LightGBM but do
not enumerate a tree-specific grid. `tree_trials.json` records the shared
eight configurations, 1,500-estimator ceiling, and 50-round validation early
stopping before either tree model was fit. CatBoost uses the categorical field
natively; LightGBM uses training-only categorical integer maps. This does not
change the candidate MPE or any already observed ridge result.

## 2026-08-29 — strictly positive kernel normalization correction

The first BTS ridge cell failed before fitting because raw Gaussian affinities
for a remote airport underflowed to an all-zero row at the smallest bandwidth.
Gaussian and Laplacian partition weights are now evaluated with rowwise
log-sum-exp stabilization. This is algebraically identical to the frozen
partition-of-unity formula and changes no finite-range mathematical value; it
only prevents a numerical zero denominator for kernels that are strictly
positive. The correction is target-independent, applies uniformly to every
task, and is covered by an extreme-distance regression test. No failed BTS
cell produced or exposed a validation or test score.

## 2026-08-29 — compact-kernel validity rule, before ablation fitting

The frozen kernel ablation includes a triangular compact-support kernel, but
does not specify what to do when the primary Gaussian-selected bandwidth leaves
a state outside every landmark ball. For that ablation only, its bandwidth is
the larger of the primary selected bandwidth and the next representable float
above the all-state landmark cover radius. This is the smallest deterministic,
target-independent radius that gives every declared state a nonzero partition.
Gaussian, Laplacian, and inverse-distance ablations retain the primary
validation-selected bandwidth. The rule does not affect the primary MPE or any
main-comparison cell.

## 2026-08-29 — secondary classification optimizer clarification, before fitting

The BTS `ArrDel15` secondary target freezes state-balanced Brier loss but not a
linear-classifier regularization grid. Its mechanism view uses averaged
logistic SGD with the shared eight-value L2 grid
`[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]`, at most 2,000 iterations, and
the same state-balanced training weights. Validation selects alpha and test is
evaluated once. The regression-selected metric bandwidth is reused, so this
secondary analysis cannot retune geometry in its favor.

## 2026-08-29 — neural landmark-token implementation correction

The first neural runner passed normalized partition weights directly to the
backbone. That is the exact collapsed representation for a ridge/linear head,
but it omitted the candidate's explicitly frozen trainable landmark-token
matrix `V` in neural models. Neural implementation version 2 now applies a
bias-free `m x D` layer to the trailing MPE weights before the unchanged
backbone, with `m=D=32`; ordinary covariates bypass that layer. Corrupt- and
equality-MPE use the identical tokenizer. Same-metric normalized Similarity
Encoding remains a direct 32-coordinate input and is no longer aliased to MPE.
All completed version-1 MPE neural cells are detected by missing version
metadata and recomputed; completed non-MPE cells remain valid. Active neural
jobs were interrupted before this correction to limit invalid computation.
This correction was prompted by implementation audit, not by a favorable or
unfavorable result, and makes the comparison conform to the frozen definition.

## 2026-08-29 — neural causal-control scheduler completion

The resume-safe neural runner could execute `mpe_equality` and every
`mpe_corrupt_0` through `mpe_corrupt_9` cell, but its convenience `core`
scheduler initially omitted those names. The scheduler now includes equality
MPE and all ten frozen corrupt-metric MPE controls. Each uses implementation
version 2, the same learned `m x D` landmark-token layer, backbone, HPO trials,
rows, and three seeds as correct MPE. No definition, metric, split, outcome, or
success gate changed; this entry records completion of a prospectively required
control rather than a new experiment chosen from results.

## 2026-08-29 — neural data-transport optimization

The initial neural runner repeatedly sliced and densified the immutable SciPy
CSR design and copied every mini-batch from host memory during every epoch.
The same per-cell float32 design is now densified once and retained on the GPU;
mini-batches are selected there using the unchanged frozen NumPy permutation.
Targets and state-balancing weights are likewise copied once per fit.  Row
membership and order, batch sizes, models, losses, optimizer, HPO trials,
early stopping, seeds, and test-sealing behavior are unchanged.  The legacy
CPU/sparse path remains available.  This is an execution optimization required
to complete the frozen matrix efficiently, not a method or outcome change.

## 2026-08-29 — Gate B source-language operationalization

The frozen protocol defines Gate B's source clause as “all or nearly all” but
does not turn “nearly all” into an integer.  The analysis records the explicit
rule as at least `number_of_runnable_sources - 1` source-mean wins, in addition
to the already frozen 80% task-by-split threshold.  This is the most literal
integer interpretation of “nearly all,” is reported in `gate_summary.json`,
and cannot rescue the paper because Gates A and F remain independently
necessary.  No outcome, cell, or corruption draw is changed.

## 2026-08-29 — neural categorical lookup implementation correction

An implementation audit found that the initial neural `unknown_embedding`
runner sent the training-state-plus-UNK one-hot vector directly to the
backbone.  Its first hidden width therefore acted as the learned embedding
width, rather than the frozen tokenizer width `D=32`.  The corrected runner
applies a bias-free one-hot-to-32 projection before the unchanged backbone;
this is exactly a learned lookup table with one shared UNK vector.  Version
metadata invalidates and recomputes every legacy UNK neural cell while leaving
all other representations untouched.  This strengthens a mandatory baseline,
restores the prospectively defined output dimension, and was made from a
contract audit rather than an outcome-driven method change.

## 2026-08-29 — CUDA multi-process scheduling optimization

The exhaustive neural matrix launches many independent, low-occupancy CUDA
workloads.  NVIDIA Multi-Process Service (MPS) was enabled for new CUDA
contexts after a separate 512-by-512 matrix-multiplication smoke test connected
successfully while the original non-MPS contexts continued uninterrupted.
MPS only replaces driver time-slicing with concurrent kernel scheduling; no
active-thread cap, numerical-precision flag, tensor value, row order, batch,
model, optimizer, HPO trial, seed, early-stopping rule, or test access changed.
Already-running cells retained their original contexts, and the daemon is
stopped after the final GPU job.  This is a transparent execution optimization
for completing the frozen matrix, not an experimental or outcome-driven change.

## 2026-08-29 — atomic cell-boundary MPS transition

To move long-lived pre-MPS bundle workers into shared scheduling without
discarding their expensive candidate result, a worker became transition-
eligible only after its version-2 MPE JSON and state-metric Parquet were both
atomically present.  Eligible old-context workers were then interrupted during
a later, not-yet-persisted representation and the resume-safe scheduler started
a new bundle under MPS.  Every completed payload was preserved; the interrupted
later cell is rerun from its unchanged frozen trial list and seeds, and the
final scheduler pass fills every such gap.  Transitions were batched and halted
on a device when memory headroom tightened.  This changes process scheduling
only and neither selects nor alters any experimental outcome.
