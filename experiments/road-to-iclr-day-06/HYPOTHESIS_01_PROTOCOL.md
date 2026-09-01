# Frozen protocol H1 — Semantic Arithmetic Amplification

Status: **FROZEN BEFORE H1 OUTCOMES**  
Freeze date: 2026-08-28 (Asia/Seoul)

## Motivation and novelty boundary

Day 5 showed that matched initial functions remove essentially all ordinary
schema variance for MLP and ResNet but leave transient architecture-dependent
residuals, especially for FT-Transformer.  Generic floating-point sensitivity,
edge-of-stability amplification, group averaging, equivariant architectures,
and equivariant stochastic trajectories are prior art.  H1 does not claim any
of those ingredients.

The candidate contribution is the joined, tabular-pipeline statement:

1. an exact schema action is a coordinate permutation at the dense interface;
2. conjugating the first-layer weights makes the real-arithmetic initial
   functions identical;
3. finite-precision dot products nevertheless depend on coordinate order;
4. a matched optimizer can amplify this semantic roundoff seed; and
5. changing only the accumulator at the schema interface can remove most of
   the resulting prediction-orbit variance at negligible model-state cost.

## Mathematical hypothesis

Let `P_g` be the permutation matrix induced by exact schema action `g`.  For a
canonical row `x` and first-layer weight `W`, the rendered row and matched
weight are

`x_g = x P_g^T`, `W_g = W P_g^T`,

so in real arithmetic `W_g x_g^T = W x^T`.  Under round-to-nearest floating
point, a length-`d` dot product obeys the standard forward-error form

`|fl(W_g x_g^T) - W x^T| <= gamma_d |W| |x|^T`,

where `gamma_d = d u / (1 - d u)` and `u` is unit roundoff.  The error value,
not merely its bound, depends on the summation order induced by `g`.

For a matched update map `theta_(t+1) = Phi_t(theta_t)`, two schema paths obey
the first-order perturbation recurrence

`delta_(t+1) = J_t delta_t + eta_t + O(||delta_t||^2)`,

where `eta_t` is the newly injected arithmetic mismatch and `J_t` is the
Jacobian of the optimizer update.  Hence

`||delta_T|| <= sum_(s<T) [prod_(r=s+1)^(T-1) ||J_r||] ||eta_s|| + h.o.t.`

The hypothesis is therefore conditional, not universal: interface roundoff
matters when the finite-time amplification products are large.  Float64
interface accumulation reduces `u` by roughly `2^29` relative to float32;
after casting the interface output back to float32, exactly matched outputs are
expected whenever the residual is below half an output ulp.

## Experiment

Datasets are the inferential unit:

- `bank_marketing_subscription` (mixed binary classification),
- `credit_card_default` (numeric binary classification),
- `fremtpl_claim_count` (mixed regression).

Models are MLP, ResNet, and dense-stem FT-Transformer.  Each bundle uses one
master seed and three nonidentity exact feature/category schema views.  Class
labels, minibatch order, dropout tape, optimizer, hyperparameters, and all
non-interface arithmetic are held fixed.

For each precision arm, train one canonical reference and three matched schema
paths:

- `fp32`: ordinary float32 interface accumulation;
- `iea64`: float64 interface accumulation followed immediately by float32 cast.

Record aligned validation and test predictions at epochs
`0, 1, 2, 5, 10, 20`.  The pilot uses seeds 6101 and 6202.  Confirmation seeds
6303–6808 may be run only after the pilot decision is written, but their menu is
frozen here.

## Primary quantities

For each dataset/model/seed/view/arm/checkpoint:

- aligned prediction MSE to the corresponding canonical reference;
- maximum absolute aligned prediction gap;
- proper validation/test loss of both paths;
- amplification `A_t = prediction_MSE_t / max(prediction_MSE_0, 1e-30)`.

The primary endpoint is the final-checkpoint orbit MSE, averaged over the three
nonidentity views and then over seeds, with datasets kept equally weighted.

## Frozen gates

H1 passes the pilot and advances to confirmation only if all are true:

1. paired `iea64` orbit MSE is lower than `fp32` in at least 7/9
   dataset×model cells;
2. the equal-dataset geometric-mean reduction is at least 90%;
3. `iea64` changes mean proper test loss by no more than 0.1% relative to its
   own canonical reference; and
4. at least three cells show `fp32` amplification above `10^4`, so a reduction
   is not obtained solely from already-null cells.

H1 is discarded as a broad mechanism if gate 1 or gate 2 fails.  It is narrowed
to an architecture-specific mechanism if the reduction is concentrated in one
model.  It is rejected as a practical intervention if gate 3 fails.

After pilot passage, confirmation passes if at least 7/9 cell-level seed means
remain positive and the dataset-block bootstrap 95% interval for the log risk
ratio lies below zero.  Seeds and views are repeated measurements, not
independent inferential units.

## Mandatory controls and integrity

- epoch-0 real-function matching tolerance: maximum probability/output gap
  below `1e-6`;
- canonical repeated-run identity check on a smoke bundle;
- identical initialization and stochastic seed domains within each pair;
- no TF32 in either arm; deterministic algorithms enabled;
- same train/validation/test rows as the existing completion infrastructure;
- outcomes are written once and manifests include source/config hashes;
- failed gates remain failed and motivate H2 rather than post-hoc H1 repair.

## Possible H2 branches (frozen conceptually, not as experiments)

- If `iea64` collapses epoch-0 error but not later divergence, test compensated
  summation or canonical gather at every schema-dependent reduction.
- If early amplification predicts final orbit risk but IEA does not help, pivot
  to a semantic Lyapunov diagnostic rather than an intervention.
- If neither relationship holds, abandon arithmetic amplification and test a
  distinct hypothesis about stochastic token/member coupling.
