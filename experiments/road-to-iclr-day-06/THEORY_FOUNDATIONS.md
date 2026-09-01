# Day-6 theory foundations

This note states the mathematical objects and claim boundaries behind H1–H9.
It deliberately distinguishes real-arithmetic identities, deterministic
finite-precision observations, and sufficient conditions for bitwise closure.

## 1. Semantic conjugacy

Let a rendered tabular row be a column vector `x in R^d`, and let an exact
schema action `g` induce permutation matrix `P_g`.  The transformed row and
conjugated first affine weight are

`x_g = P_g x`, `W_g = W P_g^T`.

### Proposition 1 — real-arithmetic interface commutation

For every `g`, `W_g x_g = W x`.

**Proof.** `W_g x_g = W P_g^T P_g x = W x`, since a permutation matrix is
orthogonal.  A bias term is unchanged.  Category-ID relabeling after one-hot
rendering is also a coordinate permutation and is covered by the same result.

This proposition says nothing about training or finite precision.

## 2. Coordinate-order floating-point error

Write `u_p` for unit roundoff at precision `p` and

`gamma_d^(p) = d u_p / (1 - d u_p)`, provided `d u_p < 1`.

For one exact dot product `z = sum_i a_i`, standard sequential error analysis
gives

`|fl_p(sum_i a_i) - z| <= gamma_d^(p) sum_i |a_i|`.

Different schema permutations reorder the same products `a_i = W_ji x_i`.

### Proposition 2 — permutation discrepancy bound

For any two coordinate orders `pi` and `sigma`,

`|fl_p(sum_pi a_i) - fl_p(sum_sigma a_i)|
 <= 2 gamma_d^(p) sum_i |a_i|`.

**Proof.** Apply the dot-product forward-error bound to both orders and use the
triangle inequality around the common real sum `z`.

The bound is worst-case and does not predict the realized sign, covariance, or
GPU kernel structure.  H2's failure shows why replacing realized structured
error with `u_p` alone is generally invalid.

## 3. Higher-precision interface closure after casting

IEA64 computes the first affine output in float64 but immediately rounds it to
float32.  Float64 addition remains non-associative; therefore “IEA64 is always
exact” would be false.

Let `R_32` be round-to-nearest float32.  For a real number `z`, define its
rounding-cell margin

`m_32(z) = distance(z, boundary of R_32^{-1}(R_32(z)))`.

### Proposition 3 — rounding-cell closure

For output coordinate `j`, let

`S_j(x,W) = sum_i |W_ji x_i|`.

Write `z_j = (Wx)_j + b_j` for the exact affine output.  Because both operands
of each product are float32 values embedded exactly in float64, each product is
exactly representable in float64; only their reduction (and the final bias
addition) rounds before the float32 cast.

If

`gamma_(d+1)^(64) (S_j(x,W) + |b_j|) < m_32(z_j)`

for every row and interface output coordinate, then every coordinate order has
the same cast interface value:

`R_32(fl_64(W_g x_g + b)) = R_32(fl_64(Wx + b))`.

**Proof.** Treat the bias addition as one additional term.  Every float64
ordered affine reduction lies within
`gamma_(d+1)^(64) (S_j + |b_j|)` of the common real result by the forward-error
bound.  The strict margin condition places that full interval inside one
float32 rounding cell, on which `R_32` is constant.

This condition is sufficient, not necessary.  It also clarifies H2's
bfloat16/float16 reversal: a coarser rounding map has wider constant cells and
can coalesce two reductions despite having larger operand approximation error.

## 4. Pathwise semantic commutation

Let `C_g` conjugate all schema-indexed parameters and optimizer states.  Let
`A_t(theta, xi_t; X, y)` be one deterministic training update with random tape
element `xi_t` (minibatch order, dropout, and other declared randomness).

### Assumption A — downstream bitwise equivariance

Whenever conjugate states receive bitwise-equal aligned interface activations
and the same transformed target/random tape, downstream forward/backward
operations, aligned schema-indexed gradients, and the optimizer update are
bitwise equal after `C_g`.

This is an implementation property, not a general fact about all hardware or
kernels.  The experiments directly audit its conclusion.

### Proposition 4 — pathwise closure by induction

Suppose the initial states satisfy `theta_0^g = C_g theta_0`, Proposition 3's
closure condition holds at every visited update, and Assumption A holds.  Then
for every training step `t`,

`theta_t^g = C_g theta_t`

and all aligned predictions are bitwise identical.

**Proof.** The claim holds at `t=0` by construction.  Given the inductive
hypothesis, Proposition 3 makes the interface activations equal after casting.
Assumption A makes aligned gradients and the optimizer update equal, hence
`theta_(t+1)^g = C_g theta_(t+1)`.  Induction completes the proof.

H1/H3 test this conclusion rather than assuming it.  An exact zero across long
paths is strong evidence for the implementation under the tested conditions,
not a hardware-independent theorem.

## 5. Perturbation amplification

When interface values differ, align both paths into one parameter chart and
write `delta_t = theta_t^g - C_g theta_t`.  A first-order expansion of the
matched update gives

`delta_(t+1) = J_t delta_t + eta_t + r_t`,

where `eta_t` is new finite-precision mismatch and
`||r_t|| = O(||delta_t||^2)` locally.

### Proposition 5 — finite-horizon upper recurrence

If `||J_t|| <= L_t` and `||r_t|| <= c_t ||delta_t||^2`, then

`||delta_T|| <= [prod_(q<T) L_q] ||delta_0||
 + sum_(s<T) [prod_(q=s+1)^(T-1) L_q]
   (||eta_s|| + c_s ||delta_s||^2)`.

**Proof.** Take norms in the one-step recurrence, apply submultiplicativity,
and recursively substitute.

This bound motivates but does not prove H4.  A two-epoch semantic shadow
contains realized injection and early Jacobian products; whether those products
forecast epoch 20 is an empirical persistence claim with frozen failure gates.

## 6. Proper-risk interpretation

For aligned probability/regression prediction `f_g(x)` over uniformly sampled
exact schema actions, define the semantic quotient

`Q(x) = E_g f_g(x)`

and orbit variance

`V(x) = E_g ||f_g(x) - Q(x)||^2`.

For Brier loss or squared error,

`E_g ||f_g(x)-y||^2 = ||Q(x)-y||^2 + V(x)`.

Thus orbit variance is exactly the member-to-quotient excess quadratic risk.
This identity is inherited from the Day-5 Hilbert-risk analysis and is not a
new Day-6 theorem.  Day 6 changes the causal question: which finite-precision
operation creates the orbit after the initial real function has been matched?

## 7. Canonicalization baseline

When the schema action is known and reliable semantic metadata are available,
one can undo it before the first affine map instead of changing arithmetic.

### Proposition 6 — exact canonical-gather closure

Let transformed input be `x_g = P_g x`.  Define the canonicalizer
`K_g(x_g) = P_g^T x_g`.  Then `K_g(x_g) = x` bitwise when `P_g^T` is
implemented as a pure gather/copy of float32 coordinates.  Consequently, a
deterministic training pipeline receiving `K_g(x_g)`, canonical parameters,
the canonical target, and the same random tape has bitwise-identical updates
to the canonical pipeline.

**Proof.** A permutation gather performs no arithmetic and copies the exact bit
pattern of each coordinate, so `P_g^T P_g x = x` holds bitwise.  Both pipelines
then receive identical tensors and state.  Determinism makes each forward,
backward, and optimizer operation identical by induction.

Unlike Proposition 3, this result needs no rounding-cell margin and uses no
float64.  It is therefore the preferred engineering baseline when canonical
schema metadata can be carried to preprocessing.  IEA64 remains informative as
an interface-local causal intervention and may be relevant when independently
constructed dense layouts cannot be canonicalized at the kernel boundary, but
it does not dominate canonicalization.

## 8. Transfer between perturbation sources

H5 compares a semantic roundoff perturbation with independent-seed
perturbations.  The following linear-response identity states the restrictive
case in which their fragility rankings must agree.

For optimizer configuration `c`, let `M_(T,c)` be the linearized map from a
zero-mean injected perturbation `epsilon` to aligned prediction change at
horizon `T`.  If source `r` has covariance `Sigma_r`, define

`R_r(T,c) = E ||M_(T,c) epsilon_r||^2`.

### Proposition 7 — covariance-proportional transfer at a fixed horizon

`R_r(T,c) = tr(M_(T,c) Sigma_r M_(T,c)^T)`.  If
`Sigma_b = alpha Sigma_a` for some `alpha > 0`, then

`R_b(T,c) = alpha R_a(T,c)`

for every configuration, so the configuration rankings are identical.  More
generally, if

`alpha_- Sigma_a <= Sigma_b <= alpha_+ Sigma_a`

in positive-semidefinite order, then

`alpha_- R_a(T,c) <= R_b(T,c) <= alpha_+ R_a(T,c)`.

**Proof.** The expectation identity follows from
`E epsilon epsilon^T = Sigma` and cyclicity of trace.  Substitute proportional
covariances for the equality.  For the bounds, congruence by `M_(T,c)`
preserves positive-semidefinite order, and trace is monotone on that cone.

This is a boundary theorem, not a proof of H5.  Exact-schema roundoff and
independent seeds need not have proportional covariance, and H5 compares
horizons 2 and 20 rather than a common horizon.  Its empirical claim therefore
requires both cross-source spectral overlap and persistence of configuration
rankings across time.  A failure is scientifically expected under source-
specific unstable subspaces or delayed transitions, such as the partial H3
Bank/ResNet transition between epochs 20 and 50.

## 9. Conditional rounding-cell survival

Proposition 3 is evaluation-local.  A learned trajectory can later visit an
affine output arbitrarily close to a float32 rounding boundary even if all
earlier casts were closed.

Let `B_e` be the event that aligned coordinate orders cast to different
float32 values at interface evaluation `e`, over the declared population of
seeds/schema views/data.  Define the conditional hazard

`p_e = P(B_e | B_1^c, ..., B_(e-1)^c)`.

### Proposition 8 — finite-exposure survival identity and bound

Without any independence assumption,

`P(tau > N) = product_(e=1)^N (1-p_e)`

and

`P(tau <= N) <= sum_(e=1)^N p_e`,

where `tau` is the first boundary-crossing evaluation.

**Proof.** Apply the probability chain rule to the intersection
`B_1^c intersect ... intersect B_N^c`.  Taking its complement gives
`1-product_e(1-p_e)`, which is at most `sum_e p_e` by the elementary product
union bound.

This proposition explains why long exact prefixes do not prove indefinite
closure.  H3's Credit FT and ResNet development observations directly exhibit
such prefixes.  H7 cannot observe microscopic `tau` because H3 stores only
prediction checkpoints; it tests the downstream material-survival consequence.

### Proposition 9 — boundary hazard under a bounded-phase assumption

Restrict one affine output to a float32 exponent bin with rounding-boundary
spacing `h`.  Let two ordered float64 reductions be `y_1` and `y_2`, and write
`D = |y_1-y_2|`.  Assume that, conditional on `D` and the closed history, the
phase of the interval `[min(y_1,y_2), max(y_1,y_2)]` relative to the boundary
lattice has density at most `kappa/h`, with `kappa >= 1`.  Then

`p_e <= kappa E[min(1, D/h) | closed history]`

and Proposition 2 gives the conservative bound

`p_e <= kappa E[min(1,
  2 gamma_(d+1)^(64) (S_j+|b_j|)/h) | closed history]`.

**Proof.** Two round-to-nearest results can differ only if the interval between
`y_1` and `y_2` contains a rounding boundary.  Within a constant-spacing bin,
the set of phases producing an intersection has measure at most `min(h,D)`.
Integrating the bounded conditional phase density yields the first inequality.
The second substitutes the two-order discrepancy bound, treating bias as the
additional affine term.

The phase assumption is explicit and may fail: GPU reductions, learned
activations, and quantization cells can be highly structured.  H2 empirically
rejects a scalar unit-roundoff law.  Proposition 9 is therefore a sufficient
population model explaining why precision can reduce hazard, not a universal
calibration formula.

## 10. Modal-mixture curvature

### Proposition 10 — log-convexity of a positive modal mixture

Let `V(t) = sum_k a_k exp(lambda_k t)` with `a_k > 0`, and define
`ell(t) = log V(t)`.  With

`w_k(t) = a_k exp(lambda_k t) / V(t)`, one has

`ell'(t) = sum_k w_k(t) lambda_k`

and

`ell''(t) = sum_k w_k(t) lambda_k^2
             - (sum_k w_k(t) lambda_k)^2 >= 0`.

**Proof.** Differentiate `log V(t)` twice.  The second derivative is the
weighted variance of the modal growth rates and is therefore nonnegative.

This elementary identity motivates H8: increasing log-orbit slope can signal
takeover by a faster unstable mode, whereas H6's constant-slope extrapolation
can mistake harmless persistent growth for a material transition.  Actual
optimizer trajectories need not be positive exponential mixtures, so H8 is a
prospective diagnostic test rather than a consequence of the proposition.

## 11. Post-breach covariance attenuation

### Proposition 11 — linear covariance attenuation after breach

Let the aligned final prediction perturbation be

`delta_T = sum_s M_(T,s) eta_s`.

Assume the injections are zero mean and mutually independent across times.  If
the two arithmetic arms share the same response maps and

`Sigma_(s,IEA64) <= kappa Sigma_(s,FP32)`

in positive-semidefinite order for every `s`, with `0 <= kappa <= 1`, then

`E ||delta_(T,IEA64)||^2 <= kappa E ||delta_(T,FP32)||^2`.

**Proof.** Independence makes the final covariance the sum
`sum_s M_(T,s) Sigma_s M_(T,s)^T`.  Congruence preserves positive-semidefinite
order, sums preserve it, and trace is monotone; expected squared norm is the
trace of the covariance.

This is a restrictive boundary result, not a theorem about the nonlinear H3
paths.  After the arms diverge their response maps can differ, injection errors
can be biased/correlated, and pathwise ratios need not be ordered.  It motivates
H9's prospective post-breach attenuation test but does not rescue H7's hitting-
time claim.

## 12. Current theorem/claim boundary

Supported mathematically:

- exact real-arithmetic schema conjugacy;
- standard coordinate-order error bounds;
- a sufficient rounding-cell condition for identical cast outputs;
- conditional pathwise closure under explicit implementation equivariance;
- unconditional bitwise closure from exact canonical gathering when the action
  metadata are available;
- a local perturbation recurrence and finite-horizon bound.

Supported empirically on the declared panels:

- H1 exact IEA64 aligned-prediction closure at every stored checkpoint on the
  small 3×3×8 matrix;
- an exact one-update state-level conjugacy audit on CPU for MLP, ResNet, and
  FT-Transformer, covering parameters and AdamW first/second moments;
- macroscopic fp32 amplification in FT-Transformer and stable MLP/ResNet
  boundaries at 20 epochs;
- failure of nominal unit roundoff to order hitting times;
- H7 finite-horizon survival extension on 31 prospective bundles / 93 paths:
  95.2% later material hits, 96.8% exact early survival, and 20.4% IEA64
  material failures;
- H9 post-breach attenuation on 25 prospective bundles / 75 paths: 51/51
  eligible final wins, 36/51 rescues, zero twofold worsenings, and +0.760%
  equal-dataset canonical loss.

Prospective boundary evidence:

- the first H3 Bank/ResNet seed remains near numerical scale at epoch 20 but
  has mean fp32 orbit MSE `8.15e-4` at epoch 50 and `2.32e-2` at epoch 200;
  IEA64 is exactly closed.  Thus the short-horizon architecture boundary is
  not a universal long-horizon theorem.
- two Credit cells show that IEA64 exactness is also finite-horizon: Credit FT
  first fails after exact prefixes of 20–100 checkpoint epochs, and Credit
  ResNet has one view fail after an exact 20-epoch prefix.  Universal H3 is
  falsified on the complete matrix; H7 supports survival extension rather than
  indefinite closure.
- H4's two-epoch shadow has pooled material AUROC .916 but misses all three
  datasetwise correlation gates, and H5 fails to transfer reliably to
  independent-seed fragility.  H6/H8 are accurate but add exactly zero over
  their simpler comparators.

Not yet supported:

- hardware-independent exactness;
- predictive-accuracy improvement;
- arbitrary-horizon/full-scale exact closure (H3 falsified);
- reliable two-epoch same-source forecasting (H4 falsified);
- transfer from early schema shadows to independent-seed fragility (H5
  falsified);
- incremental H6 or H8 screening value (both falsified);
- transport of H7/H9 rates to new datasets, hardware, schedules, or
  architectures;
- a general claim covering arbitrary schema encoders or tabular architectures.
