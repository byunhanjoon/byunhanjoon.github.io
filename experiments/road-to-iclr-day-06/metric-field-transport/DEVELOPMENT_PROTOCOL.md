# Metric-Field Transport development protocol

Status: frozen post-outcome development protocol

Frozen: 2026-08-29 Asia/Seoul, before executing the experiments defined here.

## Scientific status

The original MPE program ended `NOT SUPPORTED`. Its test results and all Day-6
outcomes have already been observed. Nothing in this directory may be reported
as a prospective confirmation of MPE.

This ladder uses only the original `train` and `validation` state partitions as
development data. Original `test` targets are sealed: runners must never index
test rows, tune on them, or emit test metrics. If a successor survives this
ladder, it receives a separately frozen protocol and new source families or
temporal cohorts before any confirmatory outcome is inspected.

## Question

Can a metric field help genuinely unseen tabular states after removing the
current MPE's linear-factorization confound, preserving raw metric information,
and training the model on simulated whole-state cold starts?

The intended contribution is not PLE, RBF interpolation, Shepard weighting,
lookup embeddings, or generic cold-start dropout. The candidate residual is a
typed tabular field interface that transports task-trained state effects using
an external target-independent metric and is trained/evaluated under strict
state-disjoint induction.

## Common rules

- Metrics, state splits, rows, targets, and ordinary covariates are inherited
  unchanged from `experiments/mpe_iclr/processed`.
- Farthest-point landmarks are selected from development-training states only.
- Targets have the mean and standard deviation of development-training rows.
- Training loss and evaluation MSE are state balanced.
- Kernel bandwidth uses one scalar: the median positive nearest-neighbor
  distance among development-training states. Raw landmark coordinates are
  affinely standardized per coordinate using means and standard deviations of
  development-training states only. Constant coordinates receive scale one.
  This transform is target independent and invertible on every nonconstant
  coordinate, so it removes an optimizer-conditioning confound without
  discarding metric information.
- Development evaluation uses the original validation states. Original test
  state identifiers may be present in immutable metadata, but their target rows
  are never indexed.
- Results are write-once JSON files. Re-execution resumes complete cells.
- Lower standardized MSE is better. Relative improvement is
  `(baseline - candidate) / baseline`.

## E0: linear-collapse and initialization control

Data:

- tasks: `acs_occupation`, `tlc_dropoff_zone`, `medical_charges`;
- original partitions 0 and 1;
- full-table setting;
- MLP only;
- seeds `20262101`, `20262102`, `20262103`.

Input is the frozen normalized Gaussian landmark-weight vector with `m=32`.
Every condition uses the same ordinary design, row order, batches, optimizer,
hidden network, and seed.

Conditions:

1. `weights_direct`: weights enter the backbone directly.
2. `factor_random_learned`: bias-free learned 32x32 tokenizer with PyTorch
   initialization, reproducing current neural MPE.
3. `factor_identity_learned`: the same trainable tokenizer initialized to the
   identity.
4. `factor_orthogonal_frozen`: a deterministic orthogonal 32x32 tokenizer,
   frozen; it is an information-preserving reparameterization control.
5. `factor_rezero`: `w @ (I + gamma * Delta)`, with `Delta` trainable and
   `gamma=0` initially.

Fixed optimization: AdamW, learning rate `1e-3`, weight decay `1e-4`, width
128, two ReLU hidden layers, dropout 0.1, batch size 2048, at most 160 epochs,
patience 20. The best validation checkpoint is the reported development value.

E0 is diagnostic and cannot promote the method. It supports the optimization
diagnosis if identity/ReZero remove a repeatable deficit of random factorization
without adding information.

## E1: preserve metric coordinates

### E1a ridge screen

All nine runnable tasks, all five original partitions, isolated-field and
full-table settings. Ridge alpha is selected from
`[0, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]` by a deterministic inner split of
the original training states; the original validation states are evaluated
once per declared representation.

Representations:

- `weights_m32`: current normalized Gaussian weights;
- `affinity_m32`: unnormalized Gaussian affinities;
- `distance_m32`, `distance_m64`, `distance_m128`: standardized raw landmark
  distances;
- `distance_all`: standardized distances to all development-training states,
  capped at 256;
- `distance_plus_weights_m128`: standardized raw distances plus 32 normalized
  weights.

For budgets exceeding the training-state count, the budget equals that count.

### E1b neural screen

Tasks `acs_occupation`, `tlc_dropoff_zone`, `citibike_start_station`, and
`medical_charges`, partitions 0 and 1, full-table MLP, the E0 optimizer, and the
same three seeds. Conditions are `weights_m32`, `distance_m32`,
`distance_m64`, `distance_m128`, `distance_all`, and
`distance_plus_weights_m128`.

### E1 promotion gate

Promote to E2 only if one raw-distance condition, selected by source-balanced
E1a full-table means, satisfies all of:

1. beats `weights_m32` on at least three of the four neural-screen source
   families;
2. wins at least 60% of paired E1b task-partition-seed cells;
3. improves the source-balanced E1b mean by at least 1%;
4. has no source-mean degradation greater than 5%.

E1 isolated-field results are mechanistic only and cannot satisfy this gate.

## E2: whole-state-masked task transport

E2 is frozen now but runs only if E1 promotes it.

Data are the four E1b tasks, partitions 0 and 1, full-table MLP, and the same
three seeds. The raw representation is the E1-selected budget; ties choose the
smaller budget.

The backbone produces an ordinary/raw hidden vector `h` of width 128. Each
development-training state has a learned warm vector `u_s` of rank 16 and a
learned scalar intercept. A state-specific low-rank expert predicts

`base(h, phi(s)) + dot(P h, u_s) / sqrt(16) + b_s`.

Cold transport uses the eight nearest development-training states under the
declared metric, with self excluded for a training query. Weights are Gaussian
and normalized; bandwidth is the common train-only scale. Entire training
states, sampled once per epoch with probability 0.5, replace `(u_s,b_s)` by
the neighbor-weighted transport `(u_hat_s,b_hat_s)` for all their rows that
epoch. Validation states always use transported values.

Conditions:

1. `raw_base`: no state expert;
2. `lookup_unknown`: warm experts, zero expert for validation states, no mask;
3. `transport_zero`: whole-state-masked zero-order expert transport;
4. `transport_first_order`: zero-order transport plus a rank-4 correction
   computed from `phi(s)-phi(a_j)`, shared right factor and anchor-specific left
   factors;
5. `transport_shuffled_metric`: condition 4 with a fixed permutation of metric
   state association, as a causal control.

The transport branch is multiplied by a learned scalar initialized to zero;
the raw base is therefore the exact initial predictor. An auxiliary
stop-gradient reconstruction loss on masked training-state experts has weight
0.1. All other optimization settings match E0.

### E2 success gate

The selected transport condition must beat both `raw_base` and
`transport_zero` on at least three of four source means, win at least 60% of
paired cells against `raw_base`, improve the source-balanced mean by at least
1%, have no source degradation above 5%, and beat the shuffled-metric control
on at least 80% of cells. Otherwise task transport is rejected as the lead.

## E3 boundary, not yet authorized

Metric-aware retrieval or larger conditional mixtures are considered only if
E2 shows a positive but sub-gate result. Their definitions must be frozen in a
new addendum before execution. No original test outcome can be used to choose
them.
