# Schema-quotient risk: theoretical foundations used on Day 5

This note fixes the estimands behind the experiments. It distinguishes exact
identities from empirical claims. All predictions below are aligned to the
same semantic output coordinates before comparison.

## Setup

Let `Z = Z_1 x ... x Z_d` be a finite product of exact representation
symmetries (feature positions, opaque category IDs, target IDs, and any other
declared nuisance). Let a complete learning pipeline with random seed `S`
produce the vector-valued prediction `P_{z,s}(x)`. The schema quotient is the
orbit barycenter

`Q(x) = E_{z,s} P_{z,s}(x)`.

All expectations in the finite experiments are uniform empirical averages.
A nonuniform deployment distribution can replace them, but must be declared;
robustness is distribution-relative.

## Proposition 1: exact proper-loss ambiguity identity

For one-hot target vector `Y` and multiclass Brier loss,

`E_{z,s} ||Y - P_{z,s}||^2
 = ||Y - Q||^2 + E_{z,s} ||P_{z,s} - Q||^2`.

The same identity holds for squared regression loss. It follows by expanding
`Y-P = (Y-Q) + (Q-P)` and using `E(P-Q)=0`. Therefore schema/seed risk is not
merely a diagnostic: it is exactly the avoidable member-loss gap closed by
the quotient barycenter under these losses. It also explains why averaging
cannot worsen empirical Brier/MSE when evaluated on the same orbit.

This statement is deliberately not extended unchanged to every proper score.
General proper scores require the corresponding Bregman orientation and
dual/primal centroid; the Day-5 primary endpoint stays with Brier/MSE.

## Proposition 2: product Hilbert fANOVA

For each nonempty factor subset `A subseteq {1,...,d}`, define recursively

`U_A(z_A) = E[P | z_A] - sum_{B proper-subset A} U_B(z_B)`,

with `U_empty = E P`. Under a product nuisance measure, the components are
orthogonal in the Hilbert space of row-wise aligned predictions. Hence

`P_z - E_z P_z = sum_{A nonempty} U_A(z_A)`

and

`R_schema = E_z ||P_z-E_zP_z||^2
          = sum_{A nonempty} E ||U_A||^2`.

The component weights are nonnegative and include interactions. This is why
one-at-a-time perturbation cards cannot in general identify an optimal joint
action. The implementation tests reconstruction and component sums exactly.

## Proposition 3: persistent versus conditional schema risk

With independent uniform schema and model-seed draws,

`Var_{z,s}(P) = Var_z(E_s[P|z]) + E_z Var_s(P|z)`.

The first term is persistent schema risk: the representation effect remaining
after ordinary seed averaging. Separately,

`E_s Var_z(P|s)`

is same-seed conditional schema risk, the amount exposed by a coupled audit.
They answer different questions and neither can be substituted for the other.
The joint experiment retains the entire `schema x seed` tensor so both and
their interactions are estimable without treating rows or representatives as
independent replications.

## Proposition 4: why quotient-HPO has no arbitrary penalty

For a fixed hyperparameter candidate `h`, validation labels `Y`, and schema
orbit `P_{h,z}`,

`E_z Brier(Y,P_{h,z})
 = Brier(Y,E_z P_{h,z}) + Var_z(P_{h,z})`.

Thus minimizing mean validation Brier over a declared schema distribution
jointly chooses quotient accuracy and ambiguity with coefficient exactly one.
This is empirical risk minimization on the augmented nuisance distribution,
not a hand-selected multi-objective scalarization. It does not guarantee that
the chosen candidate wins on a different nuisance distribution; the held-out
nuisance experiment tests that transfer explicitly.

## Proposition 5: selection-path risk decomposition

Let `B_z` be predictions from a frozen canonical candidate and let ordinary
per-schema HPO output `H_z = B_z + D_z`. Center each field over `z`. Then

`R(H) - R(B) = R(D) + 2 E_z <B_z-E B, D_z-E D>`.

Configuration switching contributes a nonnegative dispersion term but can be
partly canceled or amplified by covariance with baseline schema effects.
Consequently, decision entropy alone cannot establish prediction harm; the
Day-5 HPO panel records all three exact terms and the reconstruction error.

## Proposition 6: factor-targeted averaging under a fixed fit budget

Suppose an action exactly balances a factor subset `M` and uses the remaining
budget for independent repetitions over all other nuisance factors. Exact
averaging over `M` annihilates every fANOVA component whose index set
intersects `M`. If `b` independent complement draws remain, each surviving
component's expected variance is divided by `b`. Therefore the expected
residual risk is

`R(M,b) = (1/b) sum_{A: A intersect M = empty} V_A`,

where `V_A = E||U_A||^2`, subject to the realized cost of balancing `M`.
Enumerating feasible `M` yields a finite, auditable action frontier. This
formula motivates OrbitCover; it does not guarantee transfer of estimated
component weights across datasets, models, or splits. The leave-one-dataset-
out experiment is the empirical gate for that claim.

## Proposition 7: orthogonal-cover risk is a spectral fANOVA filter

Represent a budget-`B` randomized design by its cell-weight vector `W` over
the finite joint nuisance product, and write

`C = E[(W-u)(W-u)^T]`,

where `u` is the uniform full-product weight. If the design family is closed
under independent permutations of every factor's level names, `C` commutes
with the product symmetric-group action. It is therefore scalar on every
tensor-product fANOVA contrast subspace. For component `A`, projection `Pi_A`,
rank `r_A`, and total cell count `n`, its exact multiplier is

`lambda_A = n trace(C Pi_A) / r_A`,

and estimator residual is exactly

`R_design = sum_{A nonempty} lambda_A V_A`.

An orthogonal array of strength `t` has `lambda_A=0` for every `|A|<=t`.
IID sampling with replacement has `lambda_A=1/B` for every nonconstant
component. Thus a strength-1 cover removes all main effects; the mixed
strength-2 cover removes all main and pairwise effects. Higher-order aliasing
is neither ignored nor assumed small: its coefficients and observed component
risks determine the remaining residual exactly. Day-5 verifies the Gram-matrix
residual and the component-weighted sum independently to numerical tolerance.

For the four-factor `4 x 4 x 2 x 4` product, the 64-run `GF(4)^3`
strength-3 cover removes main, pair, and triple components. Its sole nonzero
coefficient is `lambda_feature:category:class:seed = 1/27`. The strength-2
cover has the same `1/27` coefficient on the four-way component plus `1/9` on
each triple component. Therefore the exact strength-2-to-strength-3 gain is
entirely attributable to removing triple interactions; the experiment does
not silently assume the four-way term vanishes. When class is a singleton
(regression), strength 3 enumerates the complete remaining 64-cell product and
has zero residual.

## Claims this theory does not provide

- Exact data symmetry does not imply a finite training algorithm is
  equivariant.
- A low orbit risk does not imply good predictive accuracy.
- Quotient averaging improves Brier/MSE relative to orbit members, but not
  necessarily relative to a separately trained stronger model.
- A finite orbit audit says nothing about nuisance transformations omitted
  from the declared product.
- Rows and representatives are repeated measurements; dataset-level breadth
  is still required for external claims.

## Proposition 8: cover risk is exactly expected squared-loss overhead

Let a randomized cover design `D` produce

`Q_hat_D(x) = (1/B) sum_{b=1}^B P_{Z_b,S_b}(x)`.

Independent random permutations of every factor's level names make each row
of the design marginally uniform on the nuisance product, so
`E_D Q_hat_D = Q` even though rows within a design are dependent. Applying the
same orthogonal expansion as Proposition 1 gives

`E_D ||Y-Q_hat_D||^2 = ||Y-Q||^2 + E_D ||Q_hat_D-Q||^2`.

Consequently, at equal fit budget, every reduction in design residual is an
exact expected Brier/MSE reduction—not merely a robustness proxy. Its absolute
size can still be small relative to irreducible predictive loss, which is why
accuracy/AUC shifts need not be visible.

For iid nuisance draws of budget `B`, residual is `R_joint/B`. A cover with
residual `R_cover` therefore has iid-equivalent budget

`B_equiv = R_joint / R_cover`.

This is prediction-risk compute equivalence, not wall-clock speedup: different
representations or libraries may have different fit costs.

## Proposition 9: training-row order is an exact data symmetry, not an
architectural one

Permuting the training examples jointly in `X` and `Y` leaves the empirical
supervised problem unchanged. A finite implementation need not preserve this
symmetry: minibatch construction, bootstrap indices, tie resolution, and
floating-point reduction order can couple a fixed seed to physical row
positions. It is therefore valid to add row order to the declared pipeline
nuisance product while leaving validation/test row order fixed. As with class
IDs, semantic alignment is essential: the action permutes paired training
examples, never labels independently of features.

## Relationship to classical randomized-OA integration

The removal of low-order ANOVA terms by strength-`t` randomized orthogonal
arrays is classical numerical-integration theory (notably Owen, 1992, and
later OA-LHS work). Proposition 7 specializes that principle to an exact
finite mixed-level product, vector-valued aligned predictions, and complete
stochastic learning pipelines. The paper-level novelty, if any, is the
estimand and supervised application with end-to-end controls; it is not the
orthogonal-array construction or the generic ANOVA filtering idea.

## Proposition 10: cover risk controls quotient model-selection regret

Consider a finite candidate set `m=1,...,M` on a fixed validation sample. Let
`Q_m` be candidate `m`'s full nuisance quotient, `Q_hat_m` its unbiased
randomized-cover estimate, and

`L_m = ||Y-Q_m||_H^2`, `L_hat_m = ||Y-Q_hat_m||_H^2`,

where the Hilbert norm averages rows and output coordinates consistently with
Brier/MSE. Put `e_m=Q_hat_m-Q_m`. Direct expansion and Cauchy--Schwarz give

`|L_hat_m-L_m| <= 2 sqrt(L_m) ||e_m||_H + ||e_m||_H^2`.

If `m_hat` minimizes `L_hat_m` and `m_star` minimizes `L_m`, the standard
empirical-minimization inequality is

`L_{m_hat}-L_{m_star} <= 2 max_m |L_hat_m-L_m|`.

Taking expectations, upper-bounding a maximum by a sum, and using Jensen
therefore yields the finite bound

`E[L_{m_hat}-L_{m_star}]
 <= 4 sum_m sqrt(L_m R_m(D)) + 2 sum_m R_m(D)`,

where `R_m(D)=E||e_m||_H^2` is exactly the cover residual from Propositions 7
and 8. Hence eliminating low-order nuisance components tightens a concrete
upper bound on downstream quotient-selection regret. The result does not say
that every realized cover selects the same candidate, nor that smaller
validation regret must improve test loss under distribution shift; those are
the held-out empirical questions in the model-selection protocol.

## Proposition 11: linear mixed-level cover construction

Let `U` be uniform on `GF(2)^r`. Represent declared factor `j`, with
`2^{q_j}` levels, by the binary vector `A_j U`, where `A_j` has `q_j` rows.
If for every set `S` of at most `t` declared factors the vertically stacked
matrix `A_S` has full row rank `sum_{j in S} q_j`, then enumerating all `2^r`
values of `U` forms a mixed-level orthogonal array of strength `t` over the
declared factors.

Indeed, every full-row-rank linear map from `GF(2)^r` onto
`GF(2)^{sum q_j}` has equal-size fibers, so every joint level tuple for `S`
appears exactly `2^{r-sum q_j}` times. Independent permutations of each
factor's level names preserve the property and make randomized design rows
marginally uniform. This construction treats a four-level factor as one
declared two-bit block: the rank condition is checked on whole factor blocks,
not merely on arbitrary individual binary columns. The high-dimensional
field/row experiment instantiates `r=7`, two four-level factors, and up to 28
binary factors; code exhaustively verifies every declared margin through order
three before any fit.

## Proposition 12: the correct unstructured finite-product baseline

Let a nuisance product have `N` cells with centered prediction vectors
`X_1,...,X_N`, and let `D` be a uniform subset of `B <= N` distinct cells.
The simple-random-sampling-without-replacement estimator obeys

`E ||B^{-1} sum_{i in D} X_i||^2
 = (R_joint/B) (N-B)/(N-1)`.

This is the vector finite-population variance identity, obtained by expanding
the squared norm and using inclusion probabilities `B/N` and
`B(B-1)/(N(N-1))`. Thus IID-with-replacement is not the strongest
unstructured comparator. The Day-5 no-duplicate audit uses this exact
quantity, and the selection addendum samples literal distinct configurations.

## Proposition 13: exact favorable and adverse interaction regions

For the `4 x 4 x 2 x 4` array, normalize total nonconstant fANOVA energy to
one and let `E_k` be energy at interaction order `k`. Strength-2 at budget 16
has equal-budget IID risk ratio

`16 (E_3/9 + E_4/27)`.

It wins exactly when this is below one. Strength-3 at budget 64 has ratio

`64 E_4/27`,

and wins exactly when `E_4 < 27/64`. These are scope conditions, not
assumptions: a pure triple tensor makes strength-2 `16/9` times IID risk, and
a pure four-way tensor makes strength-3 `64/27` times IID risk. Exhaustive
randomized-family averaging verifies the ratios component by component. The
mixed-level implementation computes the corresponding shape-specific
coefficients rather than reusing these full-shape constants.

## Proposition 14: an anytime nested strength schedule

In the `GF(4)^3` strength-3 array indexed by `(u,v,w)`, the 16 rows satisfying
`w=u+v` form a strength-2 subarray. Within it, the four `(u,v)` pairs

`(0,0), (1,3), (2,1), (3,2)`

form a strength-1 subarray after projection to the declared mixed-level
factors. Ordering these subsets first gives literal prefixes of sizes
`4,16,64` and strengths `1,2,3`. A single set of independent random level
permutations applied to all 64 rows preserves both nesting and marginal
unbiasedness. Hence a user can increase the fit budget without discarding
earlier models. This is a construction statement; its higher-order alias risk
still follows Proposition 7 and is measured empirically.

## Proposition 15: prediction-dependent stopping need not preserve quotient unbiasedness

For every fixed checkpoint `b` in the nested schedule,
`E[Q_hat_b]=Q`. This does *not* imply `E[Q_hat_T]=Q` when the stopping
checkpoint `T` is chosen from the observed prefix predictions. Indeed,

`E[Q_hat_T-Q] = sum_b Pr(T=b) E[Q_hat_b-Q | T=b]`,

and the conditional expectations need not vanish because the event `T=b`
depends on the same estimation error. Unbiasedness is retained if the stopping
decision is independent of all cover outcomes (or is based on a genuinely
independent pilot), but not merely because every candidate prefix is
individually unbiased. Consequently the adaptive Day-5 experiment evaluates a
validation-only *selection rule*; it does not promote the stopped prediction
average as an unbiased quotient estimator. This boundary is consistent with
the broader adaptive-stopping bias literature.

## Proposition 16: cover risk bounds pairwise ranking inversions

Let candidate `i` be better than candidate `j` under full-quotient validation
loss, with gap `Delta_ij=L_j-L_i>0`, and write the randomized loss-estimation
errors as `delta_m=L_hat_m-L_m`. An inversion requires

`delta_i-delta_j >= Delta_ij`.

If both absolute errors are below `Delta_ij/2`, inversion is impossible;
therefore a union bound and Markov's inequality give

`Pr(invert i,j) <= 4(E delta_i^2 + E delta_j^2)/Delta_ij^2`.

For multiclass Brier loss, Proposition 10 gives
`|delta_m| <= 2 sqrt(L_m)||e_m|| + ||e_m||^2`. Since two probability vectors
have row-averaged squared distance at most two,

`E delta_m^2 <= (8 L_m + 4) R_m(D)`.

Thus each pairwise inversion probability has an explicit (if conservative)
upper bound in terms of the same cover residual and the quotient loss margin.
This explains why variance reduction can improve a complete ranking, while
also exposing the unavoidable difficulty of nearly tied candidates. It does
not address validation-to-test rank shift; that is a separate source of error
observed in the external selection failure.

## Proposition 17: exact target-shift versus nuisance-selection decomposition

Let `m_V` minimize the full-quotient validation loss, `m_T` minimize the
full-quotient test loss, and `m_hat` be chosen by a finite nuisance action. For
test loss `T`, the identity

`T(m_hat)-T(m_T) = [T(m_V)-T(m_T)] + [T(m_hat)-T(m_V)]`

separates a nonnegative validation-to-test target-shift floor from a
nuisance-selection term. Reducing nuisance estimation error drives `m_hat`
toward `m_V` and the second term toward zero. It does not reduce the first
term. Moreover, the second term can be negative: a noisier selector may
accidentally choose a model that the test split prefers, making better
validation fidelity appear worse on held-out loss. The external OpenML failure
and task-balanced success instantiate the two regimes. Therefore selection
accuracy and validation-to-test transfer must be reported as separate axes.

## Proposition 18: independent-cover cross-scores are unbiased for quotient risk

Let `Q_hat_A` and `Q_hat_B` be independent randomized estimators of the same
quotient prediction `Q`, each marginally unbiased, and define the cross-score

`L_cross = <Y-Q_hat_A, Y-Q_hat_B>_H`.

Writing `e_A=Q_hat_A-Q` and `e_B=Q_hat_B-Q`, independence and zero means give

`E L_cross = ||Y-Q||_H^2 + E<e_A,e_B>_H = ||Y-Q||_H^2`.

Thus two independent strength-2 covers provide an unbiased quotient Brier/MSE
*selection criterion*, whereas the ordinary score of either cover has upward
bias equal to its residual risk (Proposition 8).  The identity holds for
vector-valued multiclass Brier loss and scalar MSE.  It does not require the
rows *within* either cover to be independent; independence is required between
the two randomized covers.

For `B` IID nuisance members with residual vectors `r_b=Y-P_b`, the complete
order-2 U-statistic

`L_U = [B(B-1)]^{-1} sum_{b != b'} <r_b,r_b'>_H`

is likewise unbiased for quotient loss and is the appropriate strong IID
comparator.  Cross-scores may be negative and unbiasedness alone does not
guarantee lower variance, correct ranking in every realization, or transfer
of a validation winner to test data.

## Proposition 19: exact variance of an independent-cover cross-score

Under Proposition 18, additionally suppose the two cover errors are identically
distributed with covariance operator `C=E[e tensor e]` on the finite
prediction Hilbert space.  Put `r=Y-Q`.  Expanding the centered cross-score,

`L_cross-||r||^2 = -<r,e_A+e_B> + <e_A,e_B>`.

All cross-covariances vanish by independence and zero conditional means, so

`Var(L_cross) = 2 <r,C r> + tr(C^2)`.

In particular,

`Var(L_cross) <= 2 ||r||^2 ||C||_op + tr(C)^2`.

This states more than unbiasedness: a cover that removes dominant nuisance
components reduces cross-score noise through the same prediction-error
covariance operator.  It also explains why residual trace alone cannot fully
order selection-score variance—the covariance orientation relative to the
validation residual `r` matters.

## Proposition 20: an unbiased block-U quotient-risk schedule

Let `Q_hat_1,...,Q_hat_K` be independent randomized cover estimates, each
unbiased for `Q`, and let `r_k=Y-Q_hat_k`. For any `K>=2`,

`L_U,K = [K(K-1)]^{-1} sum_{k != l} <r_k,r_l>_H`

is unbiased for `||Y-Q||_H^2`. Every ordered cross term has the expectation in
Proposition 18, so the result follows by linearity. The `K=2` case is the
cross-score; larger `K` reuses every independent cover block in a complete
degree-2 U-statistic. If blocks contain `B` fits, checkpoints `KB` form a
compute frontier for unbiased quotient-risk selection. This result does not
make the prediction average itself exactly invariant, and dependence between
blocks would generally invalidate the identity.

## Proposition 21: cross-score variance controls quotient-selection regret

Let `L_m` be the exact quotient validation loss for candidate `m`, let
`L_tilde_m` be any unbiased score (such as Proposition 18 or 20), and select
`m_hat` minimizing `L_tilde_m`. If `m_star` minimizes `L_m`, the usual
empirical-minimization comparison gives

`L_m_hat - L_m_star <= 2 max_m |L_tilde_m-L_m|`.

Taking expectations, bounding the maximum by a sum, and applying Cauchy--
Schwarz yields

`E[L_m_hat-L_m_star] <= 2 sum_m sqrt(Var(L_tilde_m))`.

For an independent two-cover cross-score, Proposition 19 therefore gives the
fully prediction-space bound

`E regret <= 2 sum_m sqrt(2 <r_m,C_m r_m> + tr(C_m^2))`.

Unlike Proposition 10, this result has no finite-ensemble loss-bias term: the
selection score targets quotient loss directly. It remains conservative
because it replaces the maximum by a sum, and it only controls validation
quotient regret. Proposition 17's validation-to-test floor is untouched.

## Proposition 22: complementary held-out partitions can reverse candidate gaps

Fix a pooled evaluation sample of `N=n+m` rows. For candidate `j`, let `R_j`,
`V_j`, and `T_j` be its exact quotient losses on the pooled, `n`-row
validation, and complementary `m`-row test samples. Row-additivity gives the
identity

`T_j = (N/m) R_j - (n/m) V_j`.

Consequently, for candidates `j,k`,

`T_j-T_k = (N/m)(R_j-R_k) - (n/m)(V_j-V_k)`.

If the candidates have equal pooled loss but `j` wins validation, then `j`
loses test by the exactly rescaled validation gap. More generally, a pooled
advantage must exceed `n/N` times the validation advantage to prevent a rank
reversal. This is a finite-sample complement identity, not a statement that
independently drawn future data are negatively correlated with validation.

The proposition explains why reducing nuisance-score error can reliably move
selection toward the sampled validation winner while leaving complementary
test regret unchanged or worse: Propositions 18--21 control estimation of
`V_j`, whereas the random deviation `V_j-R_j` remains. Repeated training
splits, independent evaluation samples, cross-fitting, or stable/set-valued
selection are needed to address that second layer.

## Proposition 23: exact two-replicate stability-set identities

Let two independent randomized selectors have the same candidate-selection
probabilities `pi_j`, let `m_star` be the exact quotient winner, and return the
union `S` of their selected candidates. Then

`Pr(m_star in S) = 1-(1-pi_m_star)^2`,

`E|S| = 2-sum_j pi_j^2`,

and

`Pr(S is a wrong singleton) = sum_{j != m_star} pi_j^2`.

The first identity is the complement of two independent misses. The set is a
singleton exactly when both replicates select the same candidate, giving the
other two identities. Thus improving the one-replicate exact-winner
probability improves union coverage, while a concentrated selection
distribution makes the set smaller. Concentration is not sufficient for
validity—it may concentrate on a wrong candidate—so wrong-singleton rate must
be reported separately. These are finite-randomization identities, not
distribution-free confidence-set guarantees.

## Proposition 24: a two-block jackknife cancels second-order smooth-score bias

Let `phi(Q)` be a thrice differentiable scalar loss functional of a quotient
prediction, and let `Q_hat_A=Q+e_A` and `Q_hat_B=Q+e_B` be independent,
identically distributed, unbiased estimators with covariance `C`. A Taylor
expansion gives

`E phi(Q_hat_A) = phi(Q) + (1/2) tr(H_phi(Q) C) + O(E||e||^3)`

and, because `(e_A+e_B)/2` has covariance `C/2`,

`E phi((Q_hat_A+Q_hat_B)/2) = phi(Q) + (1/4) tr(H_phi(Q) C) + O(E||e||^3)`.

Therefore the two-block jackknife

`2 phi((Q_hat_A+Q_hat_B)/2) - [phi(Q_hat_A)+phi(Q_hat_B)]/2`

cancels the displayed second-order bias. For validation log loss this requires
the relevant quotient probabilities and estimator neighborhood to stay away
from zero so the Taylor remainder is controlled. Unlike Proposition 18, the
result is approximate and does not make the finite-sample jackknife exactly
unbiased; its variance may also rise. The construction is classical jackknife
bias correction applied to nuisance-quotient blocks, not a new general
jackknife principle.

More generally, for `K>=2` independent identically distributed blocks,

`J_K = [K phi(K^{-1} sum_k Q_hat_k) - K^{-1} sum_k phi(Q_hat_k)]/(K-1)`

has the same cancellation: the full mean has covariance `C/K`, so its
second-order term multiplied by `K` equals that of the average block loss. The
64-fit experiment uses `K=4`; increasing `K` can reduce variance without
forcing finite-sample bias to decrease monotonically.

## Proposition 25: strength-2 selection has an interaction-order alias boundary

Consider two scalar-regression candidates whose quotient risks differ by
`Delta>0`, and write each candidate's centered nuisance field as a pure fANOVA
component.  On the `4 x 4 x 2 x 4` product, a randomized strength-2 16-row
cover annihilates every component of order at most two, so a two-cover
cross-score ranks the two candidates without error whenever their nuisance
fields contain only such components.

For a pure order-`k` component with unit cell variance, Proposition 13 gives
the variance multiplier of one strength-2 cover mean relative to one IID draw:
zero for `k<=2`, `1/9` for `k=3`, and `1/27` for `k=4`.  At 16 rows the
equal-budget cover/IID mean-variance ratios are therefore `16/9` and `16/27`,
respectively.  Consequently, under otherwise matched independent candidate
fields, the pure-triple corner can have *more* score noise and ranking
inversions than IID-32, while the pure-four corner can have less.  This
non-monotone order boundary follows from alias coefficients: “higher order”
alone does not determine whether a strength-2 selector helps.

The controlled selection experiment instantiates this construction over four
positive quotient-risk gaps.  The statement is a counterexample and mechanism
result, not a claim that real nuisance fields are pure or that variance alone
totally orders inversion probabilities in arbitrary heterogeneous candidates.

## Proposition 26: independent screening separates miss risk from deployment noise

Let a pilot, using any possibly biased score, return a random candidate subset
`S`. Independently of that pilot, let `L_tilde_m` be an unbiased deployment
score for every retained `m`, and select its minimum over `S`. Conditional on
`S`, every deployment score still satisfies

`E[L_tilde_m | S] = L_m`.

Let `m_star` be the exact quotient winner, `Delta_max=max_m(L_m-L_m_star)`,
and `sigma_m^2=Var(L_tilde_m)`. Splitting on whether the pilot retained the
winner and applying Proposition 21 within each retained set gives

`E regret <= Delta_max Pr(m_star notin S)
 + 2 E[1{m_star in S} sum_{m in S} sigma_m]`.

Thus screening introduces an explicit miss term but can shrink the deployment
comparison term and compute. If a `B_p`-fit pilot scores all `M` candidates
and `K` retained candidates each receive `B_d` fresh deployment fits, cost is
`B_p M+B_d K`, versus `B_d M` for equal deployment. The Day-5 successful
allocation uses `(B_p,B_d,K)=(16,64,2)`. Reusing pilot blocks in the final
score would generally destroy the displayed conditional-unbiasedness step;
independent randomization is essential.

## Proposition 27: regular cover packing yields unbiased antithetic blocks

Represent every `B`-row cover by its cell-weight vector `W` and let the uniform
quotient weights be `u`. Form an undirected `d`-regular graph on a finite cover
family, and sample an oriented edge `(A,B)` uniformly. Regularity makes both
endpoints uniform over the family, so

`E[(W_A+W_B)/2] = u`.

Writing the single-cover prediction error covariance as `C` and the symmetric
edge cross-covariance as `K=E[e_A tensor e_B]`, the pair-mean covariance is

`C_pair = (C+K)/2`.

This is an exact operator identity; disjointness is useful only insofar as it
makes `K` favorable on the nuisance field. If two packed pairs are sampled
independently, their pair means satisfy Proposition 18, and their outer
cross-score is exactly unbiased with variance

`2 <r,C_pair r> + tr(C_pair^2)`.

For the Day-5 full `4 x 4 x 2 x 4` product, the graph of distinct strength-2
covers has 1,728 vertices and degree 485. Exhaustive graph averaging reduces
the surviving pure-triple covariance multipliers from `1/9` to approximately
`0.0465--0.0470` and the pure-four multiplier from `1/27` to `0.01764`, i.e.
ratios `0.419--0.477`. On the collapsed 32-cell product, every cover's sole
disjoint neighbor is its set complement; their union is the full product, so
the 32-fit pair mean equals the quotient exactly. Regular/sliced/resolvable
designs and antithetic variance reduction are classical; the contribution is
their finite nuisance-cover graph composition with an independent quotient
risk cross-score.

## Proposition 28: equivariant disjoint packs form a quotient-unbiased frontier

Let a finite group act transitively on the nuisance-product cells and on a
family of equal-size covers. Suppose a randomized algorithm returns an
unordered `K`-cover pack and its law is equivariant under this action. Then the
expected total incidence vector of the pack is group-invariant and hence
constant over cells. Because its entries sum to `KB`, the average of all pack
predictions is unbiased for the uniform quotient prediction.

If the covers are mutually disjoint and `KB=N`, their union contains every
product cell exactly once, so the estimator equals the quotient deterministically.
This yields the observed packing frontier: one 16-row strength-2 cover, a
disjoint 32-fit pair, and a mutually disjoint 64-fit four-pack. Products with
at most 32 or 64 cells close at the corresponding checkpoint; on the full
128-cell product the estimator remains randomized.

The controlled full-product operator audit estimates the four-pack pure-triple
covariance coefficients as `0.01465--0.01497`, only `0.624--0.644` of two
independent disjoint pairs, and its pure-four coefficient as `0.007385`, a
ratio of `0.837`. All five 99% Monte Carlo upper endpoints remain below their
two-pair controls. This is an empirical coefficient calibration, whereas the
unbiasedness and full-partition closure above are exact. An exactly unbiased
risk score for a non-exhaustive four-pack still needs a second independently
sampled pack and Proposition 18, costing 128 fits.

## Proposition 29: resolvable coset packs obey a block finite-population law

Let a resolution partition a product into `H` equal covers, let `Z_h` be the
centered prediction mean of cover `h`, and note `sum_h Z_h=0`. Select `K`
distinct covers uniformly without replacement. Conditional on the resolution,
the vector finite-population identity gives

`E ||K^{-1} sum_{h in S} Z_h||^2
 = [(H-K)/(K(H-1))] H^{-1} sum_h ||Z_h||^2`.

In contrast, averaging `K` independent marginally identical covers has risk
`K^{-1} E||Z||^2`. Averaging over randomized resolutions therefore yields the
exact ratio

`R_pack(K)/R_independent(K) = (H-K)/(H-1)`.

The full `4^4` strength-2 base is a two-dimensional linear subspace of
`GF(4)^4`; its 16 affine cosets are disjoint strength-2 covers partitioning all
256 cells. Hence `H=16`, and pack sizes `K=1,2,4,8,16` have ratios
`1,14/15,12/15,8/15,0`. The controlled audit verifies every coset margin,
full closure, and these ratios across all pure triple/four-way fields to
machine precision. Resolvable OAs and finite-population sampling are classical;
the contribution is the explicit nuisance-quotient compute schedule and its
composition with prediction-risk selection.

## Proposition 30: prediction packing controls smooth nonlinear score error

Let `q_i` be the quotient probability assigned to validation label `y_i`, let
`q_hat_i=q_i+delta_i` be an unbiased packed estimate, and suppose both lie in
`[epsilon,1]`. For average log loss `ell(q)=-n^{-1} sum_i log q_i`, the mean
value theorem and Cauchy--Schwarz give

`E[(ell(q_hat)-ell(q))^2] <= epsilon^{-2} E[n^{-1} sum_i delta_i^2]`.

Because the first-order Taylor term has zero expectation and
`|d^2(-log x)/dx^2| <= epsilon^{-2}`, its Jensen bias also obeys

`|E ell(q_hat)-ell(q)| <= (2 epsilon^2)^{-1}
 E[n^{-1} sum_i delta_i^2]`.

Thus any packing construction that reduces true-class prediction MSE tightens
both a log-score RMSE upper bound and a second-order bias bound. This does not
make finite log loss unbiased and the constants deteriorate near zero; it
explains why the same prediction-space covariance mechanism can transfer
beyond quadratic scores. In the frozen six-panel audit, disjoint pair32 and
four-pack64 both lower empirical log-score RMSE and absolute bias on all 6/6
panels. Exact partition cases have `delta=0`, so their nonlinear scores close
exactly without using the bound.

There is a distinct statement for the implemented clipped score
`g_epsilon(x)=-log(max(x,epsilon))`. On `[0,1]`, `g_epsilon` is globally
`epsilon^{-1}`-Lipschitz, including at zero, so the displayed score-MSE bound
continues to hold for the clipped score without an interior-support
assumption. The Taylor/Jensen bias statement does not: the map has a kink at
`epsilon`, and the `epsilon^{-2}` constant can be practically vacuous. A
frozen support audit finds that 10/150 exact classification candidates touch
the `1e-12` floor and five contain literal zeros (13 of 240,267 exact
true-class probabilities overall). Therefore the un-clipped smooth theorem is
not claimed for the complete empirical panel; the reported nonlinear
experiment is explicitly about clipped log loss. Exact partition closure
still holds because the packed and quotient predictions themselves coincide.
As a constructive interior-supported alternative, uniform smoothing
`p_alpha=(1-alpha)p+alpha/C` guarantees epsilon `=alpha/C`. At
`alpha in {1e-6,1e-4,1e-2}`, the frozen robustness audit preserves pair32's
6/6 RMSE and 5/6 regret results and fourpack64's 6/6 RMSE/regret results. Thus
the smooth theorem has a non-vacuous qualitative empirical companion, although
the score target is the smoothed quotient rather than raw log loss.
On the 23 non-exhaustive classification candidates, the second-order expansion
is quantitatively accurate: all 92 candidate-method cells have relative
approximation RMSE below 0.10 and the median is about `6e-5`; 88/92 have
first-order correlation above 0.99. The four correlation exceptions are the
same Adam-MLP/credit-g tensor, but even there second-order relative error is
`0.017--0.057`. Conversely, the worst-case global bound is a certificate, not
a scale prediction: at `alpha=0.01`, the median actual/bound MSE ratio is only
`6.5e-8` (maximum `7.3e-7`).

## Proposition 31: the interaction spectrum chooses among valid pack laws

Let two level-symmetric pack distributions have covariance multipliers
`lambda_S` and `mu_S` on the product-ANOVA subspaces, and let `E_S` denote a
prediction field's squared energy in subspace `S`. Their risk difference is

`R_lambda-R_mu = sum_S (lambda_S-mu_S) E_S`.

Consequently, disjointness, unbiasedness, and even resolvability do not totally
order pack distributions before exhaustion. For the mixed-product 64-fit
comparison, the resolvable four-coset coefficients are exactly `1/63` for each
triple and `1/189` for the four-way component. The sequential graph sampler
estimates the four triple coefficients as `0.01465--0.01497`, all below
`1/63=0.015873`, but its four-way coefficient is `0.007385`, above
`1/189=0.005291`. Hence the graph law is preferable precisely when its weighted
triple savings exceed its four-way penalty. It wins 21/23 observed full-product
candidates; the resolvable law retains exact closure and is preferable for a
sufficiently four-way-dominated field. This is the packing analogue of the
strength-aliasing phase boundary, not a universal claim that either valid
design dominates.

## Proposition 32: orbit-symmetric pack search is a finite convex design problem

For template `j`, let `a_j in R^d` contain its exact covariance multipliers
after group-orbit symmetrization, and let `b>0` be a comparator vector. A
mixture with probabilities `w` has multipliers `A w`; therefore the minimax
relative design solves the linear program

`minimize t  subject to A w <= t b, 1^T w=1, w>=0`.

The attainable multiplier vectors are exactly the convex hull of the template
vectors. A comparator is simultaneously dominated within this library iff
that hull intersects the coordinate box below it. Standard finite-dimensional
convexity also implies an optimum supported on at most `d+1` templates; here
`d=5`, while the computed optimum uses five. Orbit symmetrization makes every
such sparse mixture cell-uniform without adding marginal constraints.

On 32,827 valid 64-fit templates, the optimized worst graph-normalized
coefficient is `1.00164`: none of five coefficients beats even the graph point
estimate, and none beats its 99% lower endpoint. This certifies failure only
inside the enumerated library and against the estimated comparator; it is not
a global optimality proof for the sequential graph sampler.

## Proposition 33: quotient margins mediate accuracy transfer

For example `i`, let `gamma_i` be the gap between the largest and second-largest
quotient class probabilities, and let `e_i=q_hat_i-q_i`. If the estimated
argmax differs from the quotient argmax, then some competitor's error exceeds
the winner's error by at least `gamma_i`; hence

`1{argmax q_hat_i != argmax q_i} <= min(1, 2 ||e_i||_2^2/gamma_i^2)`.

Averaging and taking expectations bounds disagreement with the quotient
classifier wherever margins are positive. It does not totally order two
estimators by accuracy from their aggregate prediction MSE: their error can be
oriented differently across examples, and the bound becomes vacuous near ties.
In the 23-candidate diagnostic, pack64 accuracy non-wins have more near-tie mass
at every audited threshold, but the 100,000-permutation correlations do not
cross 0.05 (`p=0.078--0.104` for thresholds 0.005--0.05). Pair32 shows no
consistent explanation. The margin mechanism is exact; its empirical role in
the remaining accuracy failures is unresolved.

## Proposition 34: test regret separates nuisance-score error from partition shift

For a finite candidate set, let `L_m^V` and `L_m^T` be the exact quotient losses
on validation and test partitions, let `L_hat_m^V` be any approximate
validation score, and define

`epsilon = max_m |L_hat_m^V-L_m^V|`,

`delta = max_m |L_m^T-L_m^V|`.

If `m_hat` minimizes the approximate validation score and `m_T` minimizes exact
test loss, then

`L^T_{m_hat}-L^T_{m_T} <= 2 epsilon + 2 delta`.

Indeed, replace both test losses by their validation counterparts at cost
`2 delta`, then use empirical optimality of `m_hat` and replace both approximate
scores at cost `2 epsilon`. The two terms are controlled by different
experiments. Covers and cross-scores reduce `epsilon`, conditional on a fixed
validation sample; they do not reduce `delta`, which requires repeated or
larger data partitions and assumptions about sampling.

This is a deterministic finite-set inequality, not a distributional guarantee.
In the four-split HistGB/CatBoost diagnostic, the absolute validation-to-test
gap movement is at least 2.51 times and typically 55.3 times the quadrature
pair-cross64 score-RMSE scale; the four observed winner flips have median ratio
100.6. The RMSE quadrature is only a descriptive scale—not `epsilon` and not a
confidence radius—but the separation explains why further nuisance variance
reduction cannot repair the observed partition reversals.

## Proposition 35: partial finite-population packing is not maximal K-antithesis

Let `Z_1,...,Z_H` be a centered finite population of cover-mean errors with
operator second moment `Sigma=H^{-1} sum_h Z_h tensor Z_h`. Draw an ordered
`K`-tuple without replacement. Every position has covariance `Sigma`, while
for distinct positions

`E[Z_i tensor Z_j] = -Sigma/(H-1)`.

This follows by expanding `(sum_h Z_h) tensor (sum_h Z_h)=0` and averaging the
`H(H-1)` off-diagonal terms. Consequently the packed mean covariance is

`Cov(K^{-1} sum_i Z_i) = [(H-K)/(K(H-1))] Sigma`,

which recovers Proposition 29. In contrast, any exchangeable `K`-tuple with
the same marginal covariance and an almost-sure zero-sum constraint must have

`E[Z_i tensor Z_j] = -Sigma/(K-1)`,

because the covariance of its sum is zero. The coefficients agree only when
`K=H`. Thus a partial resolvable OrbitCover pack is negatively dependent but
not a maximally antithetic `K`-tuple; exact zero-sum closure appears only at
full resolution.

This distinction is important relative to 2026 antithetic Gaussian CV, whose
randomization perturbations sum to zero for every chosen `K`. Its nonlinear CV
summands do not therefore average to a constant, and its prediction-error
estimand remains different. The controlled calibration enumerates every
`H=16` finite pack at `K=2,4,8,16`, verifies the `-1/15` coefficient and risk
law exactly, and separately recovers Gaussian coefficients `-1`, `-1/3`,
`-1/7`, and `-1/15` with zero-sum error below `6.3e-15`.
