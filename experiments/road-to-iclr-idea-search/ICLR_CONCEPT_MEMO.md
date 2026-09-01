# ICLR 2027 concept memo

Date: 2026-08-26  
Decision horizon: ICLR 2027 abstract deadline 2026-09-18; paper deadline
2026-09-25 ([official call](https://www.iclr.cc/Conferences/2027/CallForPapers))

## Executive decision

Pursue one primary idea:

> **Same Table, Different Predictor: Auditing and Covering
> Schema-Representation Risk in Tabular Learning**

An increasingly safer title after the latest prior-art audit is:

> **Learning on the Schema Quotient: Attributing Arbitrary-Representation
> Risk Across Tabular Pipelines**

Working framework name: **OrbitANOVA**.

The paper should ask a precise question: how much predictive risk is caused
only by an arbitrary choice of equivalent schema representation, and which
choice causes it? Examples include column position, target-label numbering,
nominal category codes, measurement units, and a coordinate basis within one
field. These choices preserve the declared learning problem when applied
jointly to context/train and query/test data, yet current pipelines can produce
different predictors.

For squared and multiclass Brier loss, the average risk removed by orbit
averaging is exactly prediction variance over equivalent representations. It
can therefore be measured without query/test labels. When the schema nuisance
group is a product, a functional ANOVA decomposes that variance exactly into
main effects and interactions. This turns a generic invariance complaint into
an attributable audit: a pipeline can be robust to feature order but sensitive
to nominal codes, or robust to both separately but sensitive to their
interaction.

Keep one high-risk secondary idea:

> **Features Are Function Spaces, Not Coordinates: Chart-Covariant Tabular
> Learning** (working method: **FieldRiesz**).

FieldRiesz should not be the September lead. A predeclared fixed-strength
semantic-prior experiment is selective, but a post-freeze calibration stress
test shows that validation can reward false adjacency on nominal fields. Its
generic affine trajectory theorem is already covered by recent
preconditioned-norm work; the remaining candidate novelty requires declared
field semantics, not topology discovery by hyperparameter search.

## Why OrbitANOVA is the best continuation of Days 1–3

The local evidence supports an audit paper more strongly than another
performance method.

1. Day 1 found one large Adult representation effect, but the conservative
   discovery rule transferred to only one of six untouched datasets.
2. Day 2 found complementary errors from equivalent numerical bases, but an
   ordinary seed ensemble was stronger on average.
3. Day 3 established a broad causal phenomenon: equivalent recodings can
   change finite-budget neural learning. It also showed that the proposed
   Measure-Orbit performance method lost to an equal-update seed ensemble on
   the untouched tier.
4. Current tabular foundation models explicitly spend inference compute on
   preprocessing and permutation ensembles, but their residual sensitivity is
   usually folded into one final prediction rather than attributed.

The defensible question is no longer “which representation wins?” It is:

> Is a reported prediction or model ranking identifiable from the dataset and
> declared schema semantics, or does it depend materially on arbitrary schema
> coordinates?

## Formal object

Let `D = (X, y)` be a represented training/context dataset, `x` a query, and
`A` the complete learning pipeline, including preprocessing, initialization,
training, and inference ensembling. Let

`Z = Z_feature x Z_class x Z_category x Z_unit x Z_chart`

be a declared product of schema nuisance choices. Permutation and relabeling
factors may be groups; a finite set of scientifically meaningful charts or a
bounded unit distribution need not be closed under composition. A choice `z`
acts jointly on context/train and query/test representations. After aligning
the output back to the reference label coordinates, define

`p_z(x) = rho(z)^-1 A(z D)(z x)`.

There are two different pipeline estimands. A **fixed-recipe orbit** holds a
declared hyperparameter configuration and validation-stopping rule fixed; its
risk is conditional on that recipe and is not a property of an estimator
family. A **selection-rule orbit** reruns a frozen validation-only search
inside each `z`; the search algorithm and its randomness are then part of
`A` and `s`. Selecting a configuration on one reference spelling and silently
calling the result “the model” confounds tuning-spelling luck with the claimed
family. The broad screen should use fixed recipes for feasibility and add a
smaller end-to-end selection-rule substudy. Neither may inspect test labels.

There is one further index when the orbit itself is sampled. If `M_m` is the
development menu drawn with seed `m`, the end-to-end prediction is
`p_{z,s,m}`. Menu main variance records configurations that change coherently
across evaluation schemas; schema×menu records development-menu choices that
alter representation sensitivity. A complete uniformly weighted finite group
has no menu-sampling factor, but an arbitrary chart menu does. Hiding `m`
under a single frozen menu understates selection-rule uncertainty in exactly
the same way that hiding a validation split does.

A 3-dataset x 3-family estimand pilot already finds both regimes. With four
frozen candidates and a semantic train/validation split made before rendering,
selection is stable in 6/9 cells and fixed/selected risks coincide. It is
unstable for Adult CatBoost, Churn forest, and Churn CatBoost; schema risk
increases in all three by 26%, 77%, and 54%. The exact sign test over only
these three selected cells has minimum two-sided `p=.25`, so this common
direction is descriptive. Churn forest provides the clear two-endpoint case:
reselection improves orbit-mean Brier by `0.00129` (paired-row 95% interval
`[-0.00242,-0.00017]`) but raises schema risk from `0.00137` to `0.00241`, a
ratio of `1.77` (`[1.69,1.85]`). CatBoost selection even changes under aligned
target-ID swaps, exposing non-equivariance in the tuning path itself, although
its loss contrasts cross zero. This reinforces that the selection rule is part
of the audited pipeline without establishing behavior for broader searches.

The selection effect itself has an exact diagnostic. Let `h(z)` be the chosen
configuration, `h_0` the identity-chart choice, and
`d_z=p_{z,h(z)}-p_{z,h_0}`. Hilbert variance gives

`SR(p_selected)-SR(p_frozen) = Var_z(d_z) + 2 Cov_z(p_frozen,d_z)`.

In the three unstable cells, switching dispersion is `0.000395`, `0.001747`,
and `0.000736`, while negative cross-covariance cancels 71%, 40%, and 69%.
The switches are therefore partly corrective but not enough to avoid a net
increase. Applying fANOVA to one-hot configuration decisions attributes 60%
of Adult-CatBoost decision variance to target ID, 79% of Churn-forest decision
variance to feature×category, and 58% of Churn-CatBoost decision variance to
feature×class. This extends the audit from fitted predictions to the tuning
path without claiming a new generic variance identity.

The discrete decision also has a useful sufficient margin certificate. Let
`h_0` win on the identity representative, let `gamma` be its smallest
validation-loss gap to a competitor, and let `delta` be the maximum schema
change in any competitor-minus-`h_0` gap. If `delta<gamma`, the identity winner
must remain selected throughout the measured orbit. This elementary argmin
fact makes search margins a mechanistic moderator to record alongside decision
fANOVA. It is sufficient, not necessary, and says nothing about unseen
representatives unless `delta` is bounded there.

Uniform pooling over a complete finite group has a stronger property: changing
the starting representative only reindexes the validation-loss sum, so a fixed
tie rule selects the same configuration. This is a group-averaging corollary,
not a new HPO theorem. Sampled menus, nonuniform measures, and field-chart
families lack that guarantee.

The HPO novelty boundary is important. [Schneider et al.
(2025)](https://proceedings.mlr.press/v293/schneider25a.html) define overtuning
as over-optimization of noisy validation estimates and study test regret.
[Sun et al. (2013)](https://www.jmlr.org/papers/v14/sun13b.html) tune for stable
variable selection, and [SmoothDARTS](https://proceedings.mlr.press/v119/chen20f.html)
regularizes architecture-search perturbations. Clinical model-stability work
also varies databases and phenotype definitions rather than equivalent schema
representatives. Therefore, “HPO can be unstable” and stability-aware tuning
are prior art. The differentiated object is narrower: equivariance of a frozen
selection rule over a declared semantic quotient, aligned predictive-risk and
decision fANOVA, and the exact switch/covariance attribution inside a complete
tabular pipeline.

Seven conditionally prospective semantic train/validation splits confirm that
this is not one tuning split's accident. Each of the three baseline-unstable
cells remains unstable in all seven splits. Selected-path schema risk exceeds
identity-select-then-freeze on 7/7 Adult CatBoost splits (magnitude/binomial
sign `p=.0156/.0156`), 6/7 Churn-forest splits (`.0469/.125`), and 7/7
Churn-CatBoost splits (`.0156/.0156`). In contrast, no mean Brier difference is significant over split
assignments. The joint probability ANOVA reveals that selection mainly
amplifies schema×split coupling: its fraction of total selected-path variance
is 62%, 43%, and 54%, versus 39%, 16%, and 32% for the comparator. Persistent
schema means move only modestly. The one-hot decision ANOVA is also dominated
by feature/class/category interactions with split. Thus the defensible insight
is representation-dependent search randomness, not a universal accuracy gain
or persistent bias. The cases remain conditionally selected from the baseline
panel, so the frozen study must estimate prevalence on all predeclared cells.

That selection-bias limitation is partly addressed by a prospective screen of
the three baseline-stable binary cells. Adult forest becomes unstable on 2/7
new splits, with schema-risk increases both times; one split changes choice on
87.5% of representatives. Adult HistGB and Churn HistGB remain exactly stable
through all seven splits. A decision-only continuation completes the untouched
Otto cells: forest switches on 1/7 splits, while HistGB and CatBoost remain
stable on 7/7. Across the original 3×3 panel, forests ever switch in 3/3 cells,
CatBoost in 2/3, and HistGB in 0/3, for five of nine cells total. This supports
family-structured heterogeneity rather than a universal claim, but the panel
is still too small for population prevalence.

The Otto switch occurs at split `20260830`: feature order alone moves 4/16
representatives from configuration 0 to 3, and its gap-shift/margin ratio is
`40.6`. A promoted full-output run shows the same stability–accuracy trade-off
as Churn forest in a multiclass numerical task. Orbit-mean Brier improves by
`0.005104`, while schema risk rises from `0.003212` to `0.007549` (2.35×) and
hard-label flips rise from 13.8% to 16.9%. Query-row intervals conditional on
the fitted orbit exclude zero (`[.00373,.00500]` for risk difference and
`[-.00757,-.00259]` for Brier difference). Artifacts:
[`selection_otto_prospective_decisions.py`](selection_otto_prospective_decisions.py),
[`selection_otto_prospective_decisions.json`](selection_otto_prospective_decisions.json),
and the promoted record in
[`selection_split_otto_screen`](selection_split_otto_screen/).

The proposed margin diagnostic has initial specificity. All 24 saved
dataset/family/split orbits from the three unstable cases fail the sufficient
certificate, with `delta/gamma` from 1.55 to 132 and minimum local validation
margins as small as `1.95e-5`. It holds for stable Adult- and Churn-HistGB
controls (`delta/gamma=.564` and `.395`). A stable Adult-forest control fails
the bound (`5.00`) even though every exact orbit gap remains positive,
illustrating the intended one-way interpretation: pass certifies measured-menu
stability; fail merely requests the full decision audit.

The diagnosis also predicts a simple repair: aggregate validation loss over
the declared schema menu, choose one configuration, and freeze it across that
menu. On the seven unseen splits this lowers same-split selected-path schema
risk in 7/7 Adult-CatBoost, 6/7 Churn-forest, and 7/7 Churn-CatBoost cases
(magnitude/sign `p=.0156/.0156`, `.0469/.125`, `.0156/.0156`), reducing the joint same-split risk by 27%, 35%,
and 36%. Its mean Brier penalties are `0.000416`, `0.000490`, and `0.000232`
and are unresolved at seven splits (`p=.0625`, `.344`, `.0625`). Schema×split
fractions fall from 62% to 38%, 43% to 16%, and 54% to 27%. Thus the repair
trades a small possible accuracy cost for substantially less representation-
dependent search coupling.

This is not a new generic HPO method. A uniform full-group validation orbit
would make the choice invariant to the starting representative, but the pilot
uses a finite sampled menu that is not closed under composition. Its claim is
menu-relative pooling only. The frozen study should learn an action on a
development sub-orbit and evaluate it on disjoint representatives, compare
ordinary identity HPO, per-representative HPO, pooled validation HPO, and a
loss-plus-schema-risk selector, and report compute separately.

The required held-out-nuisance pilot is now available. A configuration chosen
on feature/category levels `0:2` (both aligned class IDs retained) is evaluated
on disjoint levels `2:4`. It lowers schema risk on 6/7 Adult-CatBoost splits
(magnitude/sign `p=.0313/.125`), 6/7 Churn-forest splits (`.0313/.0313`), and
5/7 Churn-CatBoost splits (`.0625/.219`). Held-out same-split schema risk drops 28%, 35%, and 32%, while all
proper-loss contrasts remain unresolved (`p=.406`, `.625`, `.438`). Thus the
repair transfers beyond its development representatives in two cases and is
borderline in the third.

That sentence is conditional on the one frozen nuisance partition. A
post-frozen decision audit enumerates all 36 balanced `2-of-4 × 2-of-4`
development partitions on each new split. The selected configuration matches
the full-menu choice in 88%, 85%, and 74% of the 252 cases for Adult CatBoost,
Churn forest, and Churn CatBoost, and minimizes validation loss on the
complement representatives in 76%, 75%, and 57%. Only 4/7, 2/7, and 3/7 splits
produce one choice under every partition. Since the complementary products
share validation rows, this is not an independent-sample performance result.
It does establish that nuisance-partition uncertainty belongs inside the
selection rule. The confirmatory protocol must draw independent development
and evaluation menus, replicate the menu seed, and report schema×menu
variance; it cannot choose the most favorable partition after seeing outcomes.
For an exhaustive set of equal-sized balanced folds, averaging each
candidate's development loss simply recovers the full-menu mean because every
representative appears equally often. The folds are therefore a transfer
diagnostic, not a license to vote for a convenient selector. Artifacts:
[`selection_partition_sensitivity.py`](selection_partition_sensitivity.py)
and
[`selection_partition_sensitivity.json`](selection_partition_sensitivity.json).

On the one-hot decisions, a product fANOVA over feature-menu,
category-menu, and split makes the new axis concrete. Pure menu terms explain
3.1%, 5.3%, and 10.1% of decision variance, split main explains 69.2%, 50.7%,
and 38.5%, and menu×split terms explain 27.7%, 43.9%, and 51.4%. The dominant
effect is therefore not a universally bad subset; it is that which subset is
decisive changes with the validation split. These three cases were selected
for baseline instability and the balanced subsets overlap, so the fractions
are mechanistic diagnostics rather than population estimates.

The missing output paths require only two additional uniform configuration
orbits because full-data refits are identical across validation splits. With
those paths filled, every balanced-menu choice can be evaluated on its
complement representatives. Averaging the 36 menu results within each split,
pooled selection lowers schema risk on 7/7 splits for every case; both exact
tests are `p=.015625`. Grand mean reductions are 29.2% (Adult CatBoost), 37.5%
(Churn forest), and 34.7% (Churn CatBoost). Yet only 228/252, 180/252, and
198/252 individual menus reduce risk, and mean Brier is worse on 6/7, 5/7, and
5/7 splits. Menu-involving fANOVA terms explain 20.0%, 27.1%, and 40.0% of the
joint output variance. This is stronger than the single-partition pilot but
still conditional on selected cases and a finite balanced-menu distribution.
Artifacts: [`selection_menu_output_risk.py`](selection_menu_output_risk.py),
[`selection_menu_output_risk.json`](selection_menu_output_risk.json), and the
reused/refit prediction cache
[`selection_menu_config_predictions.npz`](selection_menu_config_predictions.npz).

There is also a necessary negative qualification. Adult-CatBoost split-main
variance rises from `0.000135` to `0.000309`, and Churn-forest from `0.001329`
to `0.002211`, even though their schema×split terms fall. Fixing one pooled
configuration makes split-to-split configuration changes act coherently across
the orbit; per-representative choices can partially average them away. The
repair therefore moves variance between nuisance axes and does not dominate
in total reproducibility. This is exactly why a joint ANOVA is more useful than
a single stability score.

The product distribution `mu` over `Z` is part of the audit specification. For
finite group factors, uniform Haar measure is the unique invariant probability
and supplies a canonical choice. Infinite unit or non-group chart families
need a declared bounded distribution; there is no representation-risk number
without this choice.

There is a useful distribution-free companion when the representative set is
finite. Regard each aligned predictor as one point in the evaluation Hilbert
space and define the **schema radius**

`Rad^2(A,D) = sup_{nu in Delta(Z)} SR_nu(A,D)`.

Minimum-enclosing-ball duality gives

`Rad^2 = min_c max_z E_x ||p_z(x)-c(x)||^2`.

Thus the largest variance induced by *any* weighting of the declared
representatives equals the squared radius of their smallest prediction-space
ball; the dual weights identify the extreme spellings. This is established
convex geometry—the same quadratic dual appears in kernel minimum-enclosing
balls ([Tsang et al., AISTATS 2005](https://proceedings.mlr.press/r5/tsang05a.html))—not a new theorem. It is valuable because it answers “did the
uniform measure hide one outlier?” without pretending that a product measure
is unnecessary. The radius loses factorwise ANOVA structure and can emphasize
a small extreme support, so report it beside, not instead of, the declared
`mu`.

### Exact risk identity

For one-hot target `e_y` and multiclass Brier loss,

`R_g = E_x ||p_g(x) - e_y||_2^2`.

Let `p_bar = E_g p_g`. The bias-variance identity gives

`E_g R_g - R(p_bar) = E_x E_g ||p_g(x) - p_bar(x)||_2^2`.

Call the right side **schema-representation risk** `SR_mu(A, D)`. It is:

- nonnegative;
- zero exactly when the aligned predictor is invariant almost surely under
  the audited transformations;
- label-free on the evaluation rows;
- equal to the average Brier risk removed by exact orbit averaging.

The same identity holds for squared-error regression. More generally, suppose
a differentiable proper loss has Bregman form

`ell(y,p) = D_Phi(e_y,p) + c(y)`.

Use the left Bregman centroid `q` defined by

`grad Phi(q) = E_g grad Phi(p_g)`.

The corresponding ambiguity identity is

`E_g ell(y,p_g) - ell(y,q) = E_g D_Phi(q,p_g)`.

The right side is again independent of `y`. For log loss, `q` is the
normalized geometric mean of the aligned probabilities and the gap is
`E_g KL(q || p_g)`. This is established Bregman bias-variance/diversity
machinery, not a new theorem. It strengthens the audit beyond Brier, while the
orthogonal product-factor ANOVA below remains specific to squared Hilbert
dispersion unless a separate attribution rule is declared.

This quantity is not total uncertainty and not a correctness certificate. A
constant predictor has zero representation risk. The audit measures only the
reducible component caused by arbitrary schema representatives.

### Product-factor ANOVA

Under a product measure over `K` nuisance factors, write the Hoeffding/functional
ANOVA decomposition

`p_g = p_empty + sum_{nonempty S subset {1,...,K}} h_S(g_S)`.

The components are orthogonal, so

`SR_mu = sum_{nonempty S} E_x E_g ||h_S(g_S; x)||_2^2`.

The terms attribute representation risk to feature position, class numbering,
category codes, units, charts, and their interactions. This is standard
functional-ANOVA machinery applied to a new audit object; the paper must not
claim to invent ANOVA.

For small groups, use balanced factorial evaluation. For large products, use a
predeclared Monte Carlo or orthogonal-array estimator, report sampling error,
and keep the transformation distribution fixed across models.

Do not run full pick-freeze attribution on every model/dataset blindly. A
synthetic convergence study found that 16 pairs (`112` fits for five factors)
still give about `16%` relative RMSE for total variance and roughly `30%` for
individual total effects. Use a sequential protocol:

1. screen total schema risk with 16–32 iid orbit members and an unbiased sample
   variance plus row/bootstrap uncertainty;
2. stop exact/near-zero controls immediately;
3. run pick-freeze or a balanced small factorial only for material cases;
4. increase `N` until the selected action's estimated regret or confidence
   interval meets a frozen tolerance, not until every close factor is ordered.

In the deliberately near-tied synthetic, 16-pair top-factor accuracy was only
`56.8%`, but mean selection regret was `1.4%` and the 95th percentile `3.2%`.
For OrbitCover, decision regret is the relevant stopping statistic; for a
scientific factor-profile table, wider uncertainty must remain visible.

### Separate schema from algorithmic randomness—and declare the coupling

A fixed-seed schema orbit and an independently measured seed orbit are not
enough: schema choices can interact with RNG paths. Cross the schema product
with a seed factor `S` and decompose `p_{g,s}` jointly. This yields two distinct
quantities:

- `Var_g(E_s p_{g,s})`: **marginal schema variance**, which persists after
  averaging algorithmic randomness;
- `E_s Var_g(p_{g,s})`: **expected conditional schema variance**, equal to the
  marginal schema components plus every schema×seed interaction.

Similarly, `E_g Var_s(p_{g,s})` includes seed main variance and schema×seed
interactions. Report all three rather than declaring schema or seed “larger”
from two unrelated experiments. Seed is a reference factor, not a schema
nuisance, and orbit risk should be stated conditional or marginal accordingly.

There is an additional subtlety for randomized learners: the interaction term
depends on how random outcomes are paired across representations. Reusing the
same integer seed is an operational **common-random-number coupling**, but a
coordinate change can attach the same pseudorandom weights to different field
functions. It is not an intrinsic distance between the randomized learning
distributions.

Let `P_z` be the distribution over Hilbert-valued prediction vectors induced
by algorithmic randomness at schema representative `z`. For any joint
coupling `pi` of these marginals, define

`C(pi) = E_pi [ |Z|^-1 sum_z ||P_z - Pbar||^2 ]`.

Writing `m_z = E P_z`, a coupling-free sandwich is

`Var_z(m_z) <= |Z|^-2 sum_{z<z'} W2^2(P_z,P_z') <= inf_pi C(pi) <= C(pi_seed)`.

The first term is persistent mean-predictor schema risk; the middle term is a
pairwise transport lower bound on distributional schema risk; the last term is
the measured same-seed witness. For two representatives the transport bound is
exact; for three or more, pairwise optimal couplings need not be jointly
compatible. This prevents an arbitrary seed convention from being promoted to
an intrinsic estimand. It is a promising refinement, but empirical Wasserstein
estimation from a handful of seeds is upward biased and is not yet a paper
claim.

The retained chart grids show why this qualification matters. Persistent
mean-predictor risk is only `17.3%`, `21.2%`, `10.4%`, and `10.6%` of the
same-seed conditional risk for Adult MLP, Diamond MLP, Diamond ResNet, and
Black Friday MLP. Naive empirical pair-transport bounds are `0.000960`,
`0.000705`, `0.001537`, and `0.004232`, implausibly close to the much larger
same-seed values because prediction vectors are high-dimensional and only
16--32 seed samples are available. A split within-chart null correction gives
`0.000233`, `0.000193`, `0.000219`, and `0.000488`, close to the persistent
mean terms, but this correction is only diagnostic. The empirical lesson is
strong—most conditional chart variance is coupling-sensitive—while the
intrinsic multi-marginal estimator is not mature enough to headline the paper.

### Quotient benchmark scores and identifiable comparisons

For a model family `a`, define its declared-quotient score as expected member
risk across equivalent representatives and algorithmic randomness,

`Theta_a = E_{z,s} R(p_{a,z,s})`.

Under Brier/squared loss this is not merely another multiverse average:

`Theta_a = R(E_{z,s} p_{a,z,s}) + Var_{z,s}(p_{a,z,s})`.

With seed treated separately, the representation contribution can be reported
as persistent mean-predictor risk, a declared coupled conditional risk, or the
coupling-free distributional hierarchy above. For a deterministic learner,
the second term is exactly the label-free schema tax `SR_mu`. For a randomized
learner, fANOVA partitions it into schema, seed, and interaction components.
Thus a benchmark row can report both centroid fit and the arbitrary-coordinate
tax, while an OrbitCover deployment spends compute to remove selected tax
components.

For pairwise scientific claims, let

`Delta_ab(z) = E_s[R(p_{a,z,s}) - R(p_{b,z,s})]`

and `Delta_ab = E_z Delta_ab(z)`. Report the quotient comparison `Delta_ab`
with paired data/seed uncertainty, plus a predeclared practical-equivalence
interval `[-tau,tau]` and the representative interval or central quantiles of
`Delta_ab(z)`. Call an ordering **schema-identifiable** only when the quotient
comparison and the supported representative range lie on the same side of the
ROPE. This claim-level layer is label-dependent and is standard multiverse
logic; it complements rather than replaces the label-free prediction audit.
Its value is that the multiverse contains only branches declared to represent
the same learning problem, so disagreement is a failure of identification, not
ordinary hyperparameter heterogeneity.

### OrbitCover: turn attribution into compute allocation

Let `Q_J p = E_{g_J} p_g` be exact marginalization over a chosen set of schema
factors `J`. Conditional expectation deletes every fANOVA component involving
an averaged factor, hence

`SR(Q_J p) = sum_{S: S intersect J = empty} V_S`

and the removed risk is

`F(J) = sum_{S: S intersect J != empty} V_S`.

`F` is a nonnegative weighted coverage function, so it is monotone and
submodular. Selecting factor symmetrizations under a compute budget is thus a
submodular knapsack problem. For a generic exact wrapper, factor `j` costs
`|G_j|` members and joint factors multiply; equivalently `log |G_j|` is an
additive budget cost. For native architectural fixes or cached/shared passes,
use measured cost instead. Standard submodular-knapsack algorithms supply the
optimization guarantee; the coverage theorem itself follows directly from
fANOVA and should not be oversold as deep new mathematics.

For a closed group orbit, this marginalization is genuine symmetrization and
is invariant to the averaged action. For a finite non-group chart menu, it is
only a quotient predictor relative to a semantic raw-field renderer; it does
not make an arbitrarily represented input invariant without access to that
renderer. Use “factor marginalization” for the general theorem and reserve
“symmetrization” for group factors. OrbitCover's compute guarantee is valid in
both cases, but the deployment interface differs.

Call the audit-guided policy **OrbitCover**. Unlike validation-selected TTA, it
uses no evaluation labels and optimizes removable schema risk, not raw
accuracy. It guarantees improvement relative to the *average member risk*
under the declared schema distribution, not relative to a potentially lucky
reference spelling. That distinction must stay explicit.

Include iid sampling from the full schema product as a fallback action. An
`m`-member iid orbit mean has expected residual squared schema variance
`SR/m`, so its expected removed fraction is `1-1/m`. OrbitCover chooses between
exact factor coverage and this generic Monte Carlo action. At budget two this
correctly avoids wasting the forest's passes on its zero class-ID component;
at budget four, every model-specific factor below removes more than generic
sampling's expected `75%`.

More generally, factor marginalization and Monte Carlo compose. If exact
averaging over `J` costs `c_J`, budget `B` permits
`m=floor(B/c_J)` iid draws of the remaining factors, with exact expected
residual

`SR_B(J) = SR(Q_J p) / m`.

The empty set recovers generic orbit sampling; `m=1` recovers pure factor
coverage. A remainder smaller than `c_J` is unused, so both nominal and
realized cost must be reported. This is ordinary conditional Monte Carlo /
Rao--Blackwellization, not a new estimator identity: OrbitCover's distinctive
role is using the schema attribution to choose which nuisance factors to
Rao--Blackwellize. Weighted coverage remains the exact `m=1` theorem; under
the hybrid integer budget, enumerate small factor sets or solve the resulting
discrete problem rather than claiming the same submodular guarantee unchanged.

The allocation principle also has direct prior art. Liu et al. compare
Rao--Blackwellization against iid minibatching under a fixed evaluation budget
and construct a budgeted hybrid
([ICML 2019](https://proceedings.mlr.press/v97/liu19c.html)). Conditional-QMC
work selects variables to preintegrate using variance/Sobol importance and
tractability ([Liu and Owen,
2023](https://epubs.siam.org/doi/10.1137/22M1479129); [Liu,
2024](https://epubs.siam.org/doi/10.1137/23M1548918)). Therefore neither the
hybrid estimator nor “ANOVA chooses what to integrate out” is an independent
novelty claim. OrbitCover belongs in the paper only as a schema-specific
downstream validity test: does the audit predict which arbitrary
representation axis should consume deployment compute?

The Adult pilot already predicts different allocations:

| Pipeline | Best factor at cost 2 | Risk removed | Best factor at cost 4 | Risk removed |
| --- | --- | ---: | --- | ---: |
| CatBoost | class ID | `52.1%` | feature position | `85.3%` |
| ordinal forest | generic orbit MC | `50%` expected | category code | `93.0%` |
| native HistGB | class ID | `~100%` | class ID | `~100%` |
| native XGBoost | class ID | `55.3%` | category code | `83.8%` |

Thus a universal “shuffle columns” or “average class IDs” policy wastes passes
for at least one family. The next experiment must compare OrbitCover with
uniform schema sampling, seed ensembling, and each library's default policy at
matched total fits/passes.

An internal Churn prospective check selects `J` without labels on 500 query
rows and evaluates residual risk on 500 disjoint rows. Across six pipelines
and budgets 2/4/8, all 18 choices match the evaluation-row oracle. The three
material cases are especially diagnostic: CatBoost chooses iid / feature /
feature+class and removes 50.0% / 87.4% / 100%; HistGB chooses class / class
with two complementary draws / feature+class and removes 76.1% / 88.0% /
100%; the forest correctly retains iid sampling at all three budgets because
neither single four-level factor beats 75% at cost four. This is a row-split
pilot on a development dataset, not the frozen held-out dataset/model gate.

## What counts as an equivalent schema representation

The benchmark must distinguish nuisance changes from changes in semantics.

Use three admissibility tiers and never pool them without labels:

1. **Exact schema symmetries:** feature/row permutations, target-ID
   permutations with output alignment, and bijections of declared opaque
   nominal IDs. These form finite groups and provide the cleanest zero-risk
   tests.
2. **Same field-function space:** invertible bases spanning the same declared
   within-field functions. These preserve information and hypothesis space
   for a linear first layer, but can deliberately expose coordinate-dependent
   optimization and regularization. They require a semantic raw-field
   renderer for deployment.
3. **Semantics-backed reparameterizations:** physical unit/origin changes or
   other domain-declared charts. These are valid only when metadata establishes
   the semantic equivalence and should be reported separately from the exact
   group core.

The claim is not that every finite-budget learning algorithm must follow the
same parameter trajectory under all invertible maps. It is that a scientific
or deployment conclusion should disclose how much it depends on a coordinate
that the task specification declares arbitrary. Tier 1 supports the strongest
software-invariance language; Tiers 2--3 diagnose representation-dependent
inductive bias and require the declaration itself to be reviewed.

| Factor | Valid nuisance transformation | Important boundary |
| --- | --- | --- |
| feature position | permute whole fields and update metadata | dropping or mixing fields is not equivalent |
| class label | permute target IDs and align probabilities back | class names with external semantics remain aligned |
| nominal category code | apply a bijection to opaque IDs within a field | semantic text values are not opaque IDs |
| numerical units/origin | declared affine unit changes within one measurement | arbitrary nonlinear warps need a semantic justification |
| contrast/basis | invertible recoding within one declared field-function space | global rotations across fields are stress tests only |
| row order | jointly permute context rows | a time-ordered context is not an unordered set |

Category relabeling is especially important. If values such as `HS-grad` are
used as language, replacing their strings destroys semantic information and is
not a nuisance. The valid audit first declares the field nominal and maps its
opaque IDs through a bijection.

## Existing pilot

The reproducible diagnostic is
[`schema_orbit_pilot.py`](schema_orbit_pilot.py). It uses a cached public
TabPFN v2.5 checkpoint, aligns predictions after feature, class, and category
permutations, and evaluates the full small factorial. These are three pilot
datasets, not benchmark evidence.

| Dataset | TabPFN passes | Schema risk | Dominant ANOVA component | Brier reduction from orbit mean |
| --- | ---: | ---: | --- | ---: |
| Breast Cancer | 1 | `0.0010086` | feature position `90.4%` | `0.0010086` |
| Breast Cancer | 8 | `0.0001279` | feature position `83.0%` | `0.0001279` |
| Wine | 1 | `0.0022868` | feature position `76.3%` | `0.0022868` |
| Wine | 8 | `0.0001806` | feature position `64.6%` | `0.0001806` |
| Adult (1k/1k) | 1 | `0.0024538` | category code `53.5%` | `0.0024538` |
| Adult (1k/1k) | 8 | `0.0008847` | category code `83.1%` | `0.0008847` |

The empirical Brier identity closed to at worst about `3e-9`; the three-factor
ANOVA closed to about `2e-10`. The shipped eight-member TabPFN ensemble reduced
schema risk by about 87% on Breast Cancer, 92% on Wine, and 64% on Adult, but
did not eliminate it.

An equal-pass policy ablation behaved causally: removing internal feature
shuffling raised residual schema risk about fourfold on Breast Cancer and
tenfold on Wine; removing both feature and class shifts raised Wine risk about
twelvefold. The default joint ensemble remained best. Thus the pilot supports
the measurement framework, not a claim that a hand-designed ensemble already
beats TabPFN.

### Exploratory Adult cross-model gate

[`cross_model_orbit_gate.py`](cross_model_orbit_gate.py) applies the same
`4 x 4 x 2` feature/category/class grid to seven additional pipelines on the
same Adult 1,000/1,000 subsample. Hyperparameters are diagnostic, not tuned.

| Pipeline | Schema risk | Dominant component(s) | 16-seed risk |
| --- | ---: | --- | ---: |
| one-hot logistic | `2.7e-22` | numerical noise only | `4.9e-32` |
| LightGBM native categorical | `2.1e-31` | numerical noise only | `5.5e-32` |
| CatBoost native categorical | `0.000721` | feature `47.9%`; feature×class `37.4%` | `0.001127` |
| ordinal-code random forest | `0.002034` | category `75.7%`; feature×category `17.3%` | `0.000673` |
| sklearn native categorical HistGB | `0.001372` | class ID `~100%` | `5.4e-32` |
| XGBoost native categorical | `0.002346` | category `43.4%`; category×class `38.4%` | numerical zero |
| one-hot Adam MLP | `0.023889` | distributed across all terms | `0.025359` |

The profiles differ even among native categorical boosters. LightGBM is exact
under this grid; CatBoost is category-code invariant but position/class
sensitive; XGBoost is dominated by category and category×class terms. For the
ordinal forest, schema risk is about three times seed risk. The Adam MLP's two
risks are comparable.

An eight-seed joint factorial changes the interpretation for stochastic
models. For the ordinal forest, marginal schema variance is `0.001464`, seed
main variance is `0.000085`, and schema×seed interaction variance is
`0.000532`: the category-code effect largely persists across seeds. For
CatBoost, the corresponding values are `0.000079`, `0.000462`, and `0.000599`:
most fixed-seed sensitivity is an interaction with training randomness rather
than a consistent schema bias. This contrast is itself a factor profile.

These average effects create individual predictive multiplicity. Across the
Adult grid, hard predictions change on `4.4%` of rows for CatBoost, `7.0%` for
the ordinal forest, `2.5%` for HistGB, and `6.9%` for XGBoost. The respective
95th-percentile maximum class-probability spans are `0.125`, `0.193`, `0.124`,
and `0.243`. These are label-free diagnostics; they are not estimates of total
error or statistical significance.

The HistGradientBoosting class-ID effect has a concrete implementation
mechanism. Its categorical splitter sorts high-support levels by
gradient/Hessian, excludes low-support levels, and always assigns excluded
levels to the right child. Flipping binary labels reverses gradient order but
not that fixed-side convention, so the scanned candidate partitions are not
closed under complementation. The effect vanishes to numerical precision on
all-numerical data and when Adult codes are treated as ordinal. This is an
illustrative pipeline artifact, not a universal boosting claim.

On the full official Adult split, the class swap alone changes `1.79%` of hard
predictions, reaches a maximum aligned probability gap of `0.434`, and induces
schema risks `0.000654` under Brier and `0.001048` under log loss. The minimal
[`histgb_label_flip_reproducer.py`](histgb_label_flip_reproducer.py) isolates
the mechanism: a native categorical field with a nine-sample rare level gives
a `0.1008` maximum gap, while ordinal, one-hot, and a 20-sample rare-level
control close at floating precision. With nine rare samples, the reference and
flipped root bitsets select `{2,3}` and `{0,1}` while category `4` stays right
in both; with 20 samples, the bitsets `{2,3,4}` and `{0,1}` are true
complements and closure returns.

### Direct Day-3 chart-orbit bridge

[`chart_orbit_pilot.py`](chart_orbit_pilot.py) revisits Day 3's five Adult
ordinal encodings—local contrasts, cumulative, standardized cumulative, path
spectral, and sample-whitened. Day 3 verified that these are invertible bases
of the same declared within-field function space. The companion pilot retains
test probabilities for a `5 chart x 32 seed` MLP grid rather than only scalar
metrics.

| Component | Brier prediction variance | Fraction of joint variance |
| --- | ---: | ---: |
| chart main (plug-in) | `0.000249` | `13.3%` |
| seed main | `0.000614` | `32.9%` |
| chart×seed | `0.001004` | `53.8%` |
| joint total | `0.001867` | `100%` |

Expected conditional chart risk is therefore `0.001253`; averaging the five
charts deletes `67.1%` of the chart-plus-seed joint variance and leaves only
the seed main effect. Across the full joint grid, `11.81%` of test rows change
hard label and the 95th-percentile maximum probability span is `0.307`. The
label-free log-loss gap is `0.003058`.

Because the plug-in chart-main term contains finite-seed noise, the companion
[`seed_coupling_analysis.py`](seed_coupling_analysis.py) estimates persistent
mean-predictor chart risk with a paired U-statistic: `0.000216` with jackknife
95% interval `[0.000194, 0.000238]`. Disjoint-seed energy tests separate nine
of ten chart pairs in both half-sample directions; local versus path-spectral,
the orthogonal-basis pair, does not separate. This is much stronger evidence
than the original five-seed interaction table while retaining the coupling
caveat.

The exact-equivalence negative control is important. Full-rank sample
whitening maps each equivalent design column space to coordinates differing
only by an orthogonal transform; L2 logistic regression then closes chart risk
at `1.9e-18`, with zero hard-label changes. Thus the MLP signal is not a change
in available field functions. It is finite-budget pipeline dependence on a
coordinate chart. The large chart×seed term also motivates the coupling
qualification above: a shared integer seed is an operational witness, not by
itself an intrinsic distributional distance.

[`chart_covariant_training_pilot.py`](chart_covariant_training_pilot.py) then
performs a causal closure intervention. It sample-whitens each ordinal field,
so every equivalent basis differs only by a block-orthogonal map; transports
the first-layer initialization by that map; and trains with
rotation-equivariant SGD. At 100 epochs, the five training curves agree within
`7.6e-9`, chart risk is `1.93e-15`, no hard labels change, and Brier risk is
`0.195452`—slightly better than the 32-seed AdamW orbit mean `0.195479`.
Ordinary AdamW breaks the aligned trajectory (`0.000104` chart risk at 40
epochs), while a field-block VectorAdam repair closes it to `2.88e-13`.
Rotation-equivariant SGD and VectorAdam are established ideas, so the
contribution is the schema-field construction and causal audit closure, not
invention of optimizer equivariance.

The regression transfer is mixed but valuable. On Diamond, a 5-chart ×
16-seed MLP grid has total squared prediction variance `0.001150` (chart main
`0.000221`, seed main `0.000302`, chart×seed `0.000627`); a ResNet grid has
total `0.002366` with chart×seed `0.001547`. The covariant construction again
closes chart risk (`5.2e-15` for SGD; `5.7e-13` for field-VectorAdam), but the
single-model field-VectorAdam RMSE is `0.1532` versus `0.1467` for the
80-member raw MLP orbit mean. It is an exact, causal intervention—not yet a
performance winner.

Black Friday removes the small-data caveat. A new 5-chart × 16-seed MLP grid
uses the official split with 100,000 training and 25,000 test rows and two
declared ordinal fields. Total joint prediction variance is `0.006320`: chart
main `0.000738`, seed main `0.001753`, and chart×seed `0.003829`. Expected
conditional chart risk is therefore `0.004567`, or `72.3%` of joint variance.
The exact chart-and-seed centroid lowers standardized MSE from `0.49735` for
the mean member to `0.49103`; the 95th-percentile prediction span is `0.654`
target standard deviations and the maximum `2.168`. Equivalent-coordinate
sensitivity therefore remains material at 100k rows and is not an Adult-only
classification artifact.

The same 80 retained Black Friday fits permit a direct covariant-training
transfer. Ridge regression is invariant to numerical precision
(`2.3e-30` chart risk). Sample whitening plus transported initialization and
rotation-equivariant SGD leaves `7.3e-14` chart risk, although its standardized
MSE is `0.49950`. Field-block VectorAdam retains essentially the raw-model MSE
(`0.49731` versus `0.49735`) while reducing conditional chart risk from
`0.004567` to `1.54e-6`—a `99.97%` closure. It still trails the full 80-member
orbit centroid (`0.49103`), so the result supports a causal repair and a
stability/performance trade-off, not a claim that covariance replaces
ensembling.

Equivalent charts also change the supported architecture conclusion. Across
16 paired seeds, ResNet beats MLP under the cumulative chart by MSE `0.001404`
(95% interval `[0.000870, 0.001937]`, unadjusted `p=5.0e-5`), while local,
standardized-cumulative, path-spectral, and whitened charts yield no detectable
difference. This is not a robust reversal in favor of MLP, but it does show
that “ResNet significantly wins” is representation-dependent. Full
chart-and-seed orbit means favor ResNet (`0.02010` versus MLP `0.02161`), which
is the identifiable comparison under the declared chart distribution.

The claim-level negative control is equally informative.
[`adult_architecture_chart_pilot.py`](adult_architecture_chart_pilot.py) adds a
paired 16-seed ResNet grid on Adult. MLP-minus-ResNet Brier differences range
only from `-0.00173` to `-0.00129` over the five charts, and every per-chart
paired interval is below zero. The quotient comparison is `-0.001479` with
paired-seed interval `[-0.001878, -0.001080]`: the Adult MLP advantage is
schema-identifiable within this declared chart set. ResNet nevertheless has
large prediction instability—total joint variance `0.003575`, including
`0.002106` chart×seed, and `13.7%` hard-label changes—showing that a sensitive
model can still support a stable pairwise conclusion.

[`claim_identifiability_analysis.py`](claim_identifiability_analysis.py)
contrasts this with Diamond. There the quotient MLP-minus-ResNet difference is
`0.000292` with interval `[-0.000064, 0.000649]`; chart means span
`-0.000040` to `0.001404`, and only cumulative coding yields a detectable
ResNet advantage. Equivalent representation does not always change a claim;
the audit says when it does. These tests remain exploratory until a practical
equivalence margin, multiplicity rule, and dataset-level replication are
frozen.

The completed Day-3 natural-basis grid supplies a broader scalar check without
new model fitting. Across 25 datasets, two architectures, two exactly
equivalent cumulative/local bases, and three paired seeds, the point-estimate
MLP-versus-ResNet winner changes on 7 datasets (`28%`; Wilson interval
`[14.3%, 47.6%]`). The median normalized shift in the architecture gap is
`0.38%`, the 90th percentile `1.52%`, and the maximum `3.59%`. With a
post-hoc `0.1%` ROPE, 16/25 comparisons are identifiable, four representative
ranges cross both ROPE boundaries, two are entirely practically equivalent,
and three touch one boundary. All four cases crossing both sides of the ROPE—
Churn, Compustat direction, HELOC, and Polish-2—were then extended
conditionally prospectively to 13 new seeds. None retains opposite
seed-marginal chart means. HELOC consistently favors MLP and Polish-2 strongly
favors ResNet under both charts; Churn and Compustat have quotient intervals
crossing zero. Same-seed winner direction still changes between charts in
30.8% of the new paired cells (9/13 for Compustat), exposing chart×seed
interaction rather than persistent opposite rankings. Thus the 7/25 result is
only a low-seed descriptive screen, and there is currently no confirmed broad
architecture reversal. The analysis in
[`broad_claim_identifiability.py`](broad_claim_identifiability.py) is
label-dependent and cannot replace prediction-space OrbitANOVA because Day 3
did not retain row predictions. The full selected-case confirmation is in
[`claim_confirmation_panel.json`](claim_confirmation_panel.json).

### The risk is concentrated, structured, and cheaply approximable

The row-level signal is not diffuse. In the Adult chart grid, the highest-risk
`1%`, `5%`, `10%`, and `20%` of rows contribute `25.6%`, `52.6%`, `64.1%`,
and `78.1%` of expected conditional chart risk. This concentration turns the
audit from a dataset-level scalar into a potential inference allocation rule.

[`chart_subgroup_audit.py`](chart_subgroup_audit.py) also finds structure that
is visible without outcome labels. The sole binary-coded field splits the test
rows into groups of 10,860 and 5,421. Their persistent mean-predictor chart
risks are `0.000294` and `0.000158` (ratio `1.86`; bootstrap difference 95%
interval `[0.0000877, 0.000182]`); expected conditional chart risks are
`0.001466` and `0.000826` (ratio `1.77`); joint hard-label-change rates are
`14.8%` and `5.88%`. The archived preprocessing does not preserve the level
provenance, so this is a **binary-coded subgroup audit**, not a claim about a
named demographic attribute or fairness harm.

Across the 14 education-code levels with at least 100 test rows, log training
frequency and persistent chart risk have Spearman correlation `-0.776`
(`p=0.0011`). Rowwise chart risk also follows decision-boundary proximity:
its Spearman correlation with the Bernoulli uncertainty of the orbit centroid
is `0.923` for expected conditional risk and `0.850` for persistent risk.
These are mechanism clues, not independent confirmatory tests: rare field
levels and ambiguous rows are where arbitrary coordinate charts matter most.

[`orbit_cascade_pilot.py`](orbit_cascade_pilot.py) freezes a simple label-free
wrapper around this concentration. On disjoint development seeds and rows it
selects standardized-cumulative plus path-spectral as two probe charts. For an
evaluation seed, the probe disagreement is calibrated on unlabeled
calibration rows; only high-disagreement rows are escalated to all five
charts. Evaluation uses the other 16 seeds and 8,141 held-out rows.

| Average passes | Residual to five-chart centroid | Equal-cost random escalation | Reduction |
| ---: | ---: | ---: | ---: |
| `2.62` | `0.000123` | `0.000268` | `54.2%` |
| `3.04` | `0.0000748` | `0.000221` | `66.2%` |
| `3.54` | `0.0000353` | `0.000165` | `78.6%` |
| `4.02` | `0.0000129` | `0.000110` | `88.3%` |

Every two-way seed/row bootstrap interval for the random-minus-adaptive
advantage excludes zero; at the lowest cost it is `[0.000105, 0.000189]`.
The main control matches the realized cost seed by seed and starts from the
same probe pair, so the gain comes from allocating passes to rows, not from a
better fixed subset. Labels are used only in a clearly separated diagnostic:
adaptive Brier also lies closer to the full centroid's Brier than equal-cost
random escalation at every budget.

Call this supporting wrapper **OrbitCascade**, but do not present adaptive TTA
or dynamic ensembling itself as novel. Instance-aware learned-loss TTA already
exists ([Kim et al., NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/2ba596643cbbbc20318224181fa46b28-Abstract.html)); confidence-based adaptive
ensembling predates it ([Inoue, AISTATS 2019](https://proceedings.mlr.press/v89/inoue19a.html));
AdapTTA varies augmentation count per input
([Mocerino et al., 2021](https://arxiv.org/abs/2105.06183)); and recent
stress-aware TTA explicitly uses augmentation disagreement to allocate compute
([Yang, CVPRW 2026](https://openaccess.thecvf.com/content/CVPR2026W/Viscale/html/Yang_SA-TTS_Stress-Aware_Test-Time_Scaling_for_Vision_Models_CVPRW_2026_paper.html)).
The narrower contribution is that schema passes target a formally declared
semantic quotient, their approximation error is label-free and observable,
and OrbitANOVA supplies the factors and probe candidates. OrbitCascade is an
E3 consequence of the audit, not a second headline.

Black Friday provides a harder prospective-style transfer. Its conditional
chart risk is less concentrated—the top `1%`, `5%`, `10%`, and `20%` of rows
contribute `7.6%`, `21.2%`, `32.2%`, and `48.0%`—and the selected probes change
to cumulative plus whitened. Nevertheless, at realized costs `2.60`, `3.00`,
`3.50`, and `4.00`, adaptive escalation lowers equal-cost random residual by
`18.6%`, `22.5%`, `25.3%`, and `26.4%`. All four two-way bootstrap intervals
exclude zero; the lowest-budget advantage interval is
`[0.000180, 0.000286]`. The smaller but positive transfer is more credible
than the Adult magnitude alone and shows that OrbitCascade should report
dataset heterogeneity rather than one universal speedup.

Diamond adds an architecture transfer using the same frozen split. MLP selects
local plus standardized-cumulative probes and reduces random-escalation
residual by `39.6%`--`43.4%` over realized costs `2.57`--`3.97`. ResNet
selects local plus whitened and reduces residual by `51.1%`--`61.0%` over
costs `2.59`--`4.00`. All eight two-way seed/row intervals exclude zero. The
different pairs are useful evidence that the audit should configure the
wrapper per pipeline. In one MLP budget, lower quotient-approximation error
does not translate to lower realized MSE than random escalation; this is the
expected limitation of a label-free target and forbids an accuracy guarantee.
Together Adult, Black Friday, and two Diamond architectures support
OrbitCascade as a reproducible consequence, not an independent novelty claim.

### Cross-dataset profile changes

Churn and nine-class Otto show that the Adult result is neither isolated nor a
single universal tree pathology.

| Dataset/model | Schema risk | Dominant factor | Hard-label flip fraction |
| --- | ---: | --- | ---: |
| Churn CatBoost | `0.000598` | feature + feature×class | not frozen |
| Churn ordinal forest | `0.001083` | category + feature×category | not frozen |
| Churn native HistGB | `0.001026` | class + feature×class | `2.3%` |
| Churn XGBoost | `8.9e-16` | numerical noise | `0%` |
| Otto LightGBM | `0.002529` | feature position | `8.0%` |
| Otto CatBoost | `0.002686` | feature position | `11.6%` |
| Otto random forest | `0.001367` | feature position | `9.2%` |
| Otto native HistGB | `0.002272` | feature position | `7.7%` |
| Otto XGBoost | `0.000804` | feature position | `4.1%` |

Across the saved Adult, Churn, and small-Otto conventional grids, 12/17
pipeline cells exceed a `1e-10` numerical threshold. For those material cells,
the exact Brier tax is 0.21%--1.13% of mean member Brier (median 0.53%), root
schema risk is 0.024--0.052 probability-vector units, and hard-label changes
span 2.3%--11.6%. These are modest proper-loss fractions with nontrivial
decision instability, not claims about total error or Bayes excess risk.

All four sampled nine-class ID permutations close for the Otto tree models;
their feature position does not. Thus LightGBM is exact on Adult and Churn but
not Otto, while XGBoost's category/class effect is large on Adult, numerical on
Churn, and pure feature-position on Otto. This is the empirical reason to
measure a factor profile rather than label a model family invariant or
non-invariant once.

The Otto stochastic effects again require the joint seed view. CatBoost's
marginal feature variance is `0.000334`, versus `0.002328` feature×seed
interaction; the forest values are `0.000170` and `0.001163`. Deterministic
LightGBM, HistGB, and XGBoost retain their position effects without a seed
explanation, consistent with ordered feature search and tie-breaking being part
of the full pipeline.

The small Churn and Otto grids are now preserved as
[`cross_model_orbit_churn.json`](cross_model_orbit_churn.json) and
[`cross_model_orbit_otto.json`](cross_model_orbit_otto.json). Regeneration in
the preserved TabM environment matches every qualitative profile; small
numeric differences from the exploratory log (for example Otto XGBoost
`0.000804` rather than `0.000878`) are treated as library/environment
sensitivity, not averaged or hidden.
The full LightGBM/XGBoost scale check is preserved separately in
[`cross_model_orbit_otto_full.json`](cross_model_orbit_otto_full.json).

The full Otto split provides an important size caveat. With 39,601 training
and 12,376 test rows, LightGBM and XGBoost position risk persists at `0.001151`
and `0.000383`, changing `5.1%` and `3.3%` of hard predictions. sklearn HistGB
closes to `2.4e-32`, so its small-sample Otto position effect likely came from
ties or regime-specific training behavior. The broad audit must stratify by
sample size rather than extrapolate every 1k-context result.

### Sensitivity to the declared chart weights

[`schema_radius_analysis.py`](schema_radius_analysis.py) solves the
minimum-enclosing-ball dual directly from prediction Gram matrices. Across the
four available chart/model archives, the worst possible reweighting is only
modestly larger than uniform:

| Dataset/model | Uniform persistent risk | Schema radius squared | Ratio | Mean conditional ratio |
| --- | ---: | ---: | ---: | ---: |
| Adult MLP | `0.000249` | `0.000292` | `1.17` | `1.15` |
| Adult ResNet | `0.000311` | `0.000334` | `1.07` | `1.11` |
| Diamond MLP | `0.000221` | `0.000259` | `1.17` | `1.11` |
| Diamond ResNet | `0.000295` | `0.000358` | `1.21` | `1.04` |
| Black Friday MLP | `0.000738` | `0.000855` | `1.16` | `1.05` |

Split-half persistent ratios are `1.04`--`1.22`. The optimal weights are not
uniform and sometimes put zero weight on one chart, but no single isolated
chart makes the uniform audit qualitatively misleading. These are plug-in
mean-predictor geometries; many-seed uncertainty still applies. Schema radius
should be a secondary robustness endpoint in the broad audit, especially for
finite non-group chart sets. Product-measure OrbitANOVA remains the primary
estimand because only it supports semantic factor attribution and prospective
sampling.

## Honest novelty boundary

The closest work materially narrows the claims.

- Liu, Yang, and Adomavicius already demonstrate **task-irrelevant prediction
  sensitivity** for general-purpose LLMs, TabPFN, and LimiX under variable and
  row order, numeric precision, names, and serialization formats
  ([PNAS Nexus, June 2026](https://academic.oup.com/pnasnexus/article/5/6/pgag197/8699520)).
  They use synthetic DGPs, repeat TFM experiments over 100 datasets, compare
  against token-generation randomness, and trace order sensitivity to
  attention. This kills “we discover that harmless tabular representation
  changes alter predictions,” including a TabPFN-only version. The paper's
  opening must instead make two advances explicit. First, the phenomenon is
  not confined to language-like models: conventional forests, boosters, MLPs,
  and ResNets exhibit large and model-specific category-code, target-ID, and
  within-field chart effects, contrary to treating supervised learners as
  essentially invariant except for small RNG noise. Second, OrbitANOVA turns
  pairwise performance sensitivity into an aligned prediction-space proper-risk
  estimand, product interaction profile, randomness coupling audit, and
  targeted quotient approximation/closure. Their paper is the empirical
  premise; this paper must be the general measurement-and-design framework.
- The most direct new neighbor is **PREF**, an under-review TMLR submission
  from May 2026. It treats preprocessing as an experimental variable across
  boosted trees, MLPs, and TFMs and defines Preprocessing Sensitivity and Model
  Volatility indices from absolute performance deltas
  ([anonymous, 2026](https://openreview.net/pdf?id=1JhhSxdBS1)). It includes
  scaling, categorical/text encoding, dimensionality reduction, selection,
  and augmentation in both mismatched and curated best-practice spaces. This
  precludes a claim of the first broad tabular preprocessing-sensitivity card.
  The surviving distinction is substantive: OrbitANOVA restricts branches to
  equivalence-preserving schema actions with output alignment; measures
  prediction dispersion whose proper-score value is exactly label-free;
  attributes simultaneous factors and interactions rather than single-knob
  metric deltas; crosses schema with training randomness; and uses the
  decomposition to cover the declared quotient. The paper is dead if the
  benchmark drifts into generic preprocessing choices where one representation
  simply contains more useful information than another.

- **EquiTabPFN** defines a target-permutation equivariance gap, proves convex
  symmetrization improves expected loss, and builds a target-equivariant model
  ([Arbel et al., NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/5a66c7adffdbde9dd5e78820cbf6935c-Abstract-Conference.html)).
  OrbitANOVA must not claim the first permutation gap or group averaging result.
- A 2026 mechanistic study traces feature, row, and class permutation behavior
  in current tabular foundation models and shows that removing positional
  parameters can make approximate invariance exact
  ([Biloš et al., 2026](https://arxiv.org/abs/2605.21288)). Feature-order
  sensitivity alone cannot carry the paper.
- TabPFN already ensembles feature/class shifts and multiple preprocessing
  configurations in its public implementation; the original paper describes
  feature-order, category, and preprocessing ensembles
  ([Hollmann et al., Nature 2025](https://www.nature.com/articles/s41586-024-08328-6)).
- Functional ANOVA is established for variance attribution and interaction
  purification
  ([Lengerich et al., AISTATS 2020](https://proceedings.mlr.press/v108/lengerich20a.html)).
- A contemporaneous remote-sensing benchmark already uses fANOVA to attribute
  performance variation to architecture, initialization, fine-tuning, and
  their interactions
  ([Shiri et al., 2026](https://arxiv.org/abs/2608.04702)). This removes any
  broad claim that interaction-attributed benchmark choices are new. The
  remaining distinction is that OrbitANOVA decomposes aligned predictions
  only over semantically equivalent representatives, so its total has an
  exact label-free proper-risk meaning and a zero invariance target.
- Metamorphic testing and test-time augmentation are established general
  frameworks; MAgg extends metamorphic aggregation beyond label-preserving
  transformations
  ([Wei et al., ICML 2024](https://proceedings.mlr.press/v235/wei24i.html)).
- Greedy test-time augmentation policy search and learnable transformation
  selection are established
  ([Lyzhov et al., UAI 2020](https://proceedings.mlr.press/v124/lyzhov20a.html));
  partial group augmentation now has detailed Fourier/sample-complexity theory
  ([Tahmasebi et al., COLT 2026](https://proceedings.mlr.press/v336/tahmasebi26a.html)).
  OrbitCover's differentiated claim is label-free selection over declared
  schema nuisance *factors* using measured risk-component coverage, not the
  first learned or compute-limited augmentation policy.
- Metamorphic testing of supervised classifiers explicitly listed affine
  feature changes, class-label permutations, and attribute permutations as
  expected relations more than a decade ago
  ([Xie et al., 2011](https://pmc.ncbi.nlm.nih.gov/articles/PMC3082144/)).
  OrbitANOVA must not claim to discover those invariances; it changes the unit
  of analysis from pass/fail software tests to a proper-risk-valued,
  factor-attributed audit of modern complete pipelines.
- Expert preprocessing is already known to change tabular rankings
  ([Schalz et al., NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/ae00e5ce7142d02e30a8235ede1ec6fc-Abstract-Datasets_and_Benchmarks_Track.html)),
  and TabArena studies ranking and ensemble effects
  ([Erickson et al., NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/1697e3fb412da11dc9488249f9e7bbc9-Abstract-Datasets_and_Benchmarks_Track.html)).
- DynaTab explicitly learns dynamic feature ordering for order-sensitive
  backbones
  ([Habib et al., 2026](https://proceedings.mlr.press/v308/habib26a.html));
  feature ordering alone is therefore an especially crowded contribution.
- Reparameterization invariance of neural optimization has a mature
  Riemannian treatment
  ([Kristiadi et al., NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/395371f778ebd4854b88521100af30ad-Abstract-Conference.html)),
  and VectorAdam already repairs Adam's failure of rotation equivariance
  ([Ling et al., NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/1a774f3555593986d7d95e4780d9e4f4-Abstract-Conference.html)).
  The chart-covariant pilot may contribute a field-semantic construction and
  an audit-closing intervention, but not a generic optimizer-invariance
  theorem.
- Sobol/pick-freeze sensitivity analysis already handles vector and functional
  outputs ([Gamboa et al., 2013](https://arxiv.org/abs/1311.1797)). The broad
  estimator should use and cite this machinery rather than claim a new
  variance-estimation method.
- Proper-score bias-variance decomposition and Bregman information already
  provide the general label-free ambiguity identity
  ([Gruber and Buettner, AISTATS 2023](https://proceedings.mlr.press/v206/gruber23a.html));
  a unified ensemble-diversity treatment gives the same centroid machinery
  ([Wood et al., JMLR 2023](https://www.jmlr.org/papers/v24/23-0041.html)).
  OrbitANOVA's novelty cannot be the proper-loss decomposition itself.
- Ensemble-diversity theory warns that generic diversity is not an objective
  to maximize because it trades off with fit
  ([Wood et al., JMLR 2023](https://www.jmlr.org/papers/v24/23-0041.html)).
  OrbitCover is narrower: disagreement along a declared semantic nuisance
  orbit is removable arbitrariness, and selection minimizes its residual
  rather than rewarding unconstrained diversity among models.
- Predictive multiplicity already studies disagreement among equally good
  predictors ([Marx et al., ICML 2020](https://proceedings.mlr.press/v119/marx20a.html)).
  The differentiated object here is *structured schema-induced multiplicity*:
  every competing predictor is produced by the same full pipeline on an
  explicitly semantics-equivalent representative, with nuisance-factor
  attribution and a removable-risk interpretation.
- ML multiverse analysis already quantifies sensitivity across combinations of
  researcher choices and explicitly studies whether optimizer conclusions
  change across hyperparameters
  ([Bell et al., NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/750337e1301941f81ae31a90e0a1c181-Abstract-Conference.html)).
  Later work includes latent-representation pipelines
  ([Wayland et al., ICML 2024](https://proceedings.mlr.press/v235/wayland24a.html)).
  Schema nuisances differ because their branches are declared semantically
  equivalent, admit alignment, and therefore support a zero-risk invariance
  target rather than merely a map of defensible analytical choices. The
  label-free prediction-risk identity and factor-coverage intervention, not
  claim sensitivity in general, must carry the distinction.
- Representation dependence in probabilistic inference is a much older
  conceptual problem; unrestricted representation independence is incompatible
  with useful default inference under broad conditions
  ([Halpern and Koller, JAIR 2004](https://doi.org/10.1613/jair.1292)).
  This supports, rather than weakens, the memo's restricted stance: declare a
  small semantics-backed nuisance product and never demand invariance to every
  information-preserving bijection.

The potentially new combination is:

1. a declared product of *schema* nuisance groups for complete supervised
   tabular pipelines;
2. an exact label-free proper-loss gap for arbitrary equivalent schema
   representatives, using the appropriate Bregman centroid;
3. orthogonal attribution of that risk to nuisance factors and interactions;
4. a broad audit showing how these components change across model families,
   preprocessing ensembles, and leaderboard rankings; and
5. a design use: identify which architectural invariance or ensemble factor
   actually closes each model's risk.

A possible sixth contribution is the coupling hierarchy for randomized
learners: mean-predictor risk, a coupling-free transport lower bound between
prediction distributions, and an implementation-specific same-seed witness.
This should enter the claimed contribution list only if a many-seed estimator
and null calibration survive; the algebra alone is elementary
multi-marginal-transport geometry.

## Required experiments

### E0. Algebra, alignment, and estimators

- Unit-test feature, class, category, unit, and basis actions and their inverse
  output alignments.
- Verify the Brier/squared identity in float64 and float32.
- Verify the log-loss identity with a normalized-geometric centroid and
  reverse-KL dispersion; do not use arithmetic probability averaging for that
  exact claim.
- Verify full-factorial ANOVA closure on two- and three-factor cases.
- Compare Monte Carlo component estimates with exact factorial values where
  enumeration is possible.
- For each finite representative set, verify schema-radius/minimum-ball dual
  closure and compare uniform risk with worst-distribution risk; do not use the
  radius for factor attribution.
- Freeze a sequential screening/attribution rule and report estimator RMSE,
  confidence intervals, stopping counts, and OrbitCover selection regret.
- Cross stochastic pipelines with seed in the same design and recover marginal
  schema, seed-main, and schema×seed components.
- Include invariant negative controls and deliberately position-sensitive
  positive controls.

### E1. Five-dataset/model gate

Use Adult, Wine, Breast Cancer, Diamond, and one multiclass dataset. Include:

- one-hot logistic/ridge negative control;
- CatBoost and one boosted-tree pipeline;
- MLP or ResNet;
- TabM;
- TabPFN v2.5 single-pass and default ensemble.

Audit feature position, class labels, nominal codes, and unit changes. The gate
passes only if:

1. alignment and invariant controls close numerically;
2. at least two nontrivial model families have schema risk above run-to-run
   numerical noise;
3. factor profiles differ meaningfully across models or datasets; and
4. at least one factor changes a practical conclusion—worst-schema risk,
   compute allocation, or a pairwise model ranking.

### E2. Frozen broad audit

Reuse the Day 3 25-dataset tier and its five-dataset prospective extension, but
retain their original provenance labels. The paper must not call all 30 jointly
preregistered.

Recommended model matrix:

- logistic/ridge and random forest as controls;
- CatBoost, XGBoost, and LightGBM;
- MLP, ResNet, RealMLP if integration is reliable, and TabM;
- TabPFN v2.5 and the current reproducible TabPFN release;
- EquiTabPFN or another exact-equivariance baseline if its public checkpoint
  can be evaluated fairly;
- one additional current TFM only if installation and licensing are stable.

Primary endpoints:

- total schema-representation risk under Brier/squared loss;
- factor and interaction shares;
- orbit-worst minus orbit-mean risk;
- residual risk after each model's default ensemble.

Secondary endpoints:

- schema variance versus seed and split variance;
- marginal versus conditional schema variance and schema×seed interactions;
- model winner-flip probability and Kendall rank stability;
- runtime per unit of representation risk removed;
- sampling error of the Monte Carlo orbit estimate;
- schema radius divided by declared-uniform risk for finite representative
  sets, as a sensitivity analysis to nuisance weights;
- number of pipeline fits consumed by screening versus attribution and the
  regret of the selected factor policy.

### E3. Design intervention

Use the decomposition diagnostically, not as post-hoc storytelling.

1. Freeze the factor screen on a development subset.
2. Use OrbitCover to choose factors under budgets of 2, 4, 8, and the model's
   default pass count; estimate selection uncertainty by bootstrap or repeated
   pick-freeze designs.
3. For each model family, remove or ensemble only the selected factor using an
   existing principled mechanism: permutation-invariant architecture,
   relabeling ensemble, one-hot/contrast-invariant categorical handling, or
   fieldwise chart closure.
4. Compare with the model's default ensemble and an equal-pass generic
   ensemble on untouched datasets.

A publishable result need not beat every default model in raw accuracy. It
must show that attributed interventions close the predicted variance component
and improve worst-schema or average risk at a competitive compute cost.

## Evaluation discipline

- Treat the learner as the full pipeline; refit every train-time transform
  after acting on the represented training table.
- Apply transformations jointly to train/context, validation, and test/query;
  this is not covariate shift.
- Align class outputs and contrast coordinates back before measuring variance.
- Declare which categorical fields contain semantic text and exclude them from
  opaque-code permutations.
- Use arithmetic centroids and orthogonal ANOVA for Brier/squared loss. For log
  loss, use a normalized-geometric centroid and reverse-KL gap; report its
  scalar audit separately rather than pretending it has the same Euclidean
  factor decomposition.
- Keep global cross-feature rotations as stress tests, not required schema
  invariances.
- Freeze transformation distributions and orbit sample counts before the
  broad model comparison.
- Bootstrap over datasets for cross-dataset conclusions.
- Compare against seed, split, and equal-compute inference ensembles.
- Report constant-predictor and exactly invariant controls so zero schema risk
  is not confused with good prediction.

## Paper-shaped claims

The abstract should be able to support three claims.

1. **Object:** equivalent schema representations induce a measurable,
   label-free component of proper-score risk for complete tabular pipelines.
2. **Attribution:** the product structure of schema nuisances decomposes this
   risk into identifiable main effects and interactions; different model
   families fail for different arbitrary choices.
3. **Consequence:** default preprocessing ensembles reduce but do not uniformly
   eliminate representation risk, and the decomposition predicts which
   targeted invariance or ensemble closes it and whether leaderboard rankings
   are stable.

Do not claim that schema risk estimates total uncertainty, predicts all errors,
or that every information-preserving transformation should be ignored.

## Kill criteria

Stop the ICLR paper if any of these occurs.

1. Direct prior art already defines the same multi-factor tabular schema group,
   label-free proper-loss gap, and ANOVA attribution.
2. After invariant controls and output alignment, residual schema risk is tiny
   relative to seed/split noise on the broad natural transformations.
3. The result is almost entirely TabPFN feature position, already covered by
   EquiTabPFN and the mechanistic study.
4. Nominal-code and unit factors disappear when production-quality pipelines
   receive correct metadata.
5. Model rankings and worst-schema conclusions remain unchanged throughout.
6. Monte Carlo cost makes the audit less useful than simply increasing each
   model's default ensemble.

## Secondary idea: FieldRiesz, now conditional

FieldRiesz represents each field by a finite function space with mass matrix
`M_j` and semantic stiffness `S_j`, then uses

`K_j = M_j + lambda S_j`

for first-layer initialization, regularization, and metric-gradient updates.
Under `phi'_j = B_j phi_j`, `K'_j = B_j K_j B_j^T` and the update transforms
covariantly. The existing pilot in
[`chart_covariance_pilot.py`](chart_covariance_pilot.py) closes a cumulative/
local spline chart pair to `5.77e-15` after 25 updates, while matched AdamW
diverges to a prediction gap of `0.7064`.

That exact closure is real but insufficiently novel. *Preconditioned Norms*
already gives necessary and sufficient affine-invariance conditions for
matrix-parameter updates
([Veprikov et al., 2025/26](https://arxiv.org/abs/2510.10777)); recent work also
states symmetry-compatible optimizer design directly
([Lau and Su, 2026](https://arxiv.org/abs/2605.18106)) and analyzes Adam's gauge
sensitivity
([Singh, 2026](https://arxiv.org/abs/2608.05136)).

The only plausible novelty is the semantic field metric and selective tabular
group. The first diagnostic
[`semantic_metric_synthetic.py`](semantic_metric_synthetic.py) was
inconclusive because one nominal target let false adjacency behave as generic
shrinkage. A stronger frozen suite now draws latent field functions from path,
ring, or isotropic Gaussian priors and compares validation-selected ridge
metrics over identical one-hot function spaces. Candidate topology strength is
fixed at four while true strength varies over one, four, and sixteen, crossed
with three sample/noise regimes and 200 paired trials per cell.

The matched path metric beats a permuted path with a nonzero mean-difference
interval in all nine ordinal cells and isotropic in six; the matched ring does
the same in all nine cyclic cells and beats isotropic in six. For nominal
targets, isotropic beats path, ring, and permuted path in all nine cells. The
three weak-topology settings are unresolved versus isotropic and can favor it
in mean, so the metric coefficient cannot be treated as semantic metadata—it
must be calibrated. This supplies the previously missing synthetic
selectivity, but it is Bayes-structured evidence rather than real-data proof.
Artifacts: [`field_topology_bayes_suite.py`](field_topology_bayes_suite.py) and
[`field_topology_bayes_suite.json`](field_topology_bayes_suite.json).

A final prior-art pass further constrains the method claim. Classical smoothing
splines already write derivative penalties as quadratic forms in arbitrary
basis coordinates, so mass/stiffness matrices are not new. [Otto et al.
(2025)](https://www.jmlr.org/papers/v26/24-1315.html) provide a unified Lie-
derivative framework for enforcing, discovering, and promoting symmetry in
basis-function regression, neural networks, and fields. [PH-Reg](https://proceedings.mlr.press/v235/zhang24z.html)
already matches representation topology to regression targets. FieldRiesz can
only differentiate itself through the product of semantic *input-field*
metadata (nominal/path/ring), chart transport across equivalent encodings, and
one covariant training stack. Each ingredient in isolation is established.

The fixed-strength suite clears a controlled topology-mechanism gate, but not
the stronger semantic-selection gate. In a post-frozen extension, every
topology family chooses stiffness from `{0,1,4,16}` jointly with ridge strength.
The matched path beats isotropic in 9/9 ordinal cells and the matched ring does
so in 9/9 cyclic cells. Yet on nominal tasks, tuned path, ring, and permuted-
path families significantly beat isotropic in 8/9, 8/9, and 9/9 cells; their
zero-stiffness selection rates average only 62.8%, 62.2%, and 63.1%.
Independent validation is exploiting accidental adjacency in the realized
nominal functions. Because the tuned families have more candidates, this is
not a fair performance comparison; it is precisely a diagnostic that
unconstrained model selection cannot certify semantic admissibility.

The repaired claim is narrower and more defensible: nominal/path/ring type is
task metadata, equivalent charts within that type form the nuisance orbit, and
stiffness may be tuned only inside the declared family. This result also links
FieldRiesz back to the primary paper: a category ordering can improve one
spelling's validation/test error while remaining arbitrary on the schema
quotient. Keep FieldRiesz standalone only if multiple neural architectures
retain a benefit from *declared* metadata and achieve competitive reference-
chart loss on real data. Otherwise it remains an OrbitANOVA intervention/
control. Artifacts:
[`field_topology_strength_selection.py`](field_topology_strength_selection.py)
and
[`field_topology_strength_selection.json`](field_topology_strength_selection.json).

## Work plan to the deadline

| Date | Decision/output |
| --- | --- |
| Aug 26–28 | implement general OrbitANOVA estimator, exact controls, and E1 harness |
| Aug 29–31 | run E1; freeze schema factors, distributions, models, and gates |
| Sep 1–7 | execute broad audit and targeted interventions on two GPUs |
| Sep 8 | claim freeze and submit/no-submit decision |
| Sep 9–15 | paper draft, figures, related work, and reproducibility package |
| Sep 16–17 | external technical review and correction pass |
| Sep 18 | genuine abstract submission; author list final |
| Sep 19–24 | final audits, appendix, anonymization, code snapshot |
| Sep 25 | paper submission |

The local setup—two 96 GB H100 NVL GPUs, cached TabPFN checkpoints, and a
completed 30-dataset harness—makes the broad audit feasible. The risk is not
raw compute; it is integrating enough strong model families and keeping the
schema equivalence definitions defensible.

## Final recommendation

The strongest current thesis is:

> A tabular model should not get credit for one favorable spelling of a
> dataset. Equivalent schema choices induce a distinct, reducible component of
> predictive risk. Because those choices form a product, that risk can be
> measured without test labels and attributed to the exact arbitrary decision
> that caused it.

This idea uses the causal foundation built in Days 1–3, has a successful
three-dataset pilot including a category-code effect not explained by feature
order alone, and has an honest novelty boundary against EquiTabPFN,
metamorphic testing, and functional ANOVA. It is currently more defensible and
more executable than claiming a new invariant optimizer.
