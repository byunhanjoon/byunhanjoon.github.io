# ICLR portfolio novelty review

**Review date:** August 27, 2026

## Bottom line

Develop **OrbitANOVA** as the primary paper. It is ICLR-shaped and currently
about **3.5/5 ready**, but it is not submission-ready. Keep **FieldRiesz** as a
secondary intervention or high-risk second direction.

If the current artifacts were submitted as the paper, my mock decision would
be **weak reject**: the originality of the full composition is plausible, but
the broad prevalence and audit-guided action result are still promises. That
is a verdict on readiness, not a recommendation to abandon the direction.
The bottleneck is now demonstrated payoff, not another decorative formula.

This is not a claim that OrbitANOVA introduces a new generic theorem. Its
paper-level novelty is the complete schema-specific chain:

```text
declare equivalent schema spellings
  -> align complete-pipeline predictions
  -> price their dispersion in proper-loss units
  -> attribute it to nuisance factors, interactions, and randomness
  -> audit the tuning path itself
  -> predict and validate a targeted repair
```

If the last arrow does not transfer to unseen cases, the work risks reading as
excellent instrumentation rather than an ICLR contribution.

## The primary idea without jargon

**Shortest version:** arbitrary schema spelling is an unreported
hyperparameter. OrbitANOVA measures its predictive cost and identifies which
part of the pipeline should be repaired.

Reordering columns or replacing category IDs should not change the learning
problem. Yet a full pipeline can produce a different predictor after those
rewrites. OrbitANOVA deliberately tries every declared-equivalent spelling,
realigns the outputs, and measures how much prediction changes.

For squared and Brier loss, the variance across those predictions is exactly
the loss removed by averaging them. This gives a label-free price for arbitrary
representation choice. A factorial decomposition then identifies whether the
price came from feature order, category IDs, target IDs, units, field charts,
their interactions, or coupling with seed, split, search menu, and selected
configuration. The diagnosed component determines the candidate repair.

## Compact mathematical spine

Let `z=(z_1,...,z_k)` index the declared schema product and let `P_z(x)` be
the aligned prediction of the complete fitted pipeline. For the uniform finite
product used by the exact audit,

```text
P_bar(x) = E_z P_z(x),
SR(P)    = E_(x,z) ||P_z(x)-P_bar(x)||_2^2.
```

For regression squared loss and multiclass Brier loss, for every target `Y`,

```text
E_z L(Y,P_z) - L(Y,P_bar) = SR(P).
```

Thus schema risk is label-free but already measured in proper-loss units. It
is not a new identity; its role is to make arbitrary representation dependence
a risk-valued property of a complete pipeline rather than a pairwise accuracy
delta.

Log loss is different. Its label-free member-to-centroid gap uses the
normalized geometric (left-Bregman) centroid and equals an average reverse KL.
It should be reported as a separate scalar audit; the Euclidean orthogonal
fANOVA below must not be relabeled as a log-loss decomposition.

On the balanced nuisance product, write the Hilbert-valued functional ANOVA

```text
P_z = P_empty + sum_(A != empty) P_A(z_A),
SR(P) = sum_(A != empty) ||P_A||_2^2.
```

If `Q_J P = E[P | z_{-J}]` exactly averages nuisance factors in `J`, then

```text
SR(Q_J P) = sum_(A != empty, A intersect J = empty) ||P_A||_2^2.
```

So the audit predicts how much schema risk exact factor marginalization would
remove before fitting another action. This conditional-expectation fact is
classical Rao--Blackwellization; the new empirical question is whether a
factor-specific cover or covariant model achieves the predicted removal at a
better matched-resource frontier than an ordinary iid ensemble.

Selection is part of the pipeline. If `h(z)` is the configuration chosen under
schema spelling `z`, choose a frozen reference `h_0` and define

```text
d_z = P_(z,h(z)) - P_(z,h_0),
Delta SR = Var_z(d_z) + 2 Cov_z(P_(z,h_0), d_z).
```

Decision fANOVA on the one-hot `h(z)` localizes which schema, split, and menu
factors cause switching. This is the deepest current extension because a
stable base learner can still become schema-sensitive through its tuning rule.

## Component-by-component novelty

| Component | Novelty now | Reviewer interpretation |
| --- | ---: | --- |
| Harmless tabular rewrites change predictions | 1/5 | Established by task-irrelevant robustness, permutation, and preprocessing studies. Concede it. |
| Equivalent transformations as model tests | 1/5 | Metamorphic testing is established, including for supervised classifiers. Pairwise consistency checking is not the contribution. |
| Brier/squared ambiguity identity | 1/5 | Classical bias--variance/Bregman information. Use it; do not claim it. |
| Functional ANOVA attribution | 1/5 | Established sensitivity-analysis machinery. The vector-output application is not enough alone. |
| Equivalent-schema quotient and admissibility contract | 3/5 | A sharper object than generic preprocessing sweeps or arbitrary multiverses. It must be operational and reproducible. |
| Aligned, label-free, proper-risk-valued product audit | 3--3.5/5 | Plausibly differentiated when output relabeling, simultaneous interactions, and complete pipelines are central. |
| Schema crossed with seed/split/search randomness | 2.5--3/5 | fANOVA benchmarking already crosses architecture, initialization, and training choices. Value requires a schema-specific estimand, selection decisions, changed conclusions, and many paired seeds. |
| Tuning-path quotient and switch attribution | 3.5/5 | The strongest depth extension: the selected configuration is part of the pipeline, and schema can couple to menu and split. Broad prevalence is still unproven. |
| OrbitCover / OrbitCascade / covariant closure | 1.5--2.5/5 alone | Generic averaging, partial augmentation, adaptive escalation, and covariant optimization are occupied. Their value is as audit-predicted consequences. |
| Held-out audit-guided action transfer | up to 4/5 | This can turn a measurement paper into a design paper, but only after success on unseen dataset/model or nuisance cases at matched resources. |

## Why the closest work does not yet subsume it

- [Liu, Yang, and Adomavicius (PNAS Nexus 2026)](https://academic.oup.com/pnasnexus/article/5/6/pgag197/8699520)
  establishes task-irrelevant sensitivity for LLMs and tabular foundation
  models. OrbitANOVA must supply the conventional-pipeline extension,
  risk-valued product attribution, and action—not rediscover sensitivity.
- [PREF (2026 TMLR submission)](https://openreview.net/pdf?id=1JhhSxdBS1)
  studies broad preprocessing robustness. OrbitANOVA is narrower: every branch
  must preserve the declared task, predictions are aligned, zero representation
  risk is meaningful, and simultaneous interactions/randomness are primary.
- [A Data-Centric Perspective on tabular evaluation](https://arxiv.org/abs/2407.02112)
  already shows that expert feature engineering, HPO regime, and test-time
  adaptation can change model rankings. Those branches deliberately add
  dataset-specific information. OrbitANOVA must instead show that equivalent
  spellings alone alter the tuning path and aligned predictor, then use that
  structured effect to choose a repair.
- [EquiTabPFN (NeurIPS 2025)](https://proceedings.neurips.cc/paper_files/paper/2025/hash/5a66c7adffdbde9dd5e78820cbf6935c-Abstract-Conference.html)
  occupies target-permutation equivariance gaps and symmetrization. OrbitANOVA
  cannot claim the first group gap or averaging theorem.
- Order-specific headlines are especially crowded: [adversarial row/column
  permutations for table LLMs](https://arxiv.org/abs/2605.00445),
  [DynaTab](https://proceedings.mlr.press/v308/habib26a.html),
  [GOTabPFN](https://arxiv.org/abs/2606.05441), and a
  [mechanistic TFM permutation study](https://arxiv.org/abs/2605.21288) already
  study or exploit ordering; the last also turns a mechanistic audit into exact
  invariant architectural edits. Feature-order sensitivity and audit-guided
  invariant surgery can be factors or baselines, never the paper's novelty
  carrier.
- [Same Content, Different Representations (ICLR 2026)](https://proceedings.iclr.cc/paper_files/paper/2026/hash/4c638a7cf71c060b4bed15500da38800-Abstract-Conference.html)
  runs a controlled representation study for Table QA, while [representation-
  stable table retrieval](https://arxiv.org/abs/2604.24040) averages embeddings
  over equivalent serializations and amortizes that centroid. These use
  language/table-retrieval tasks rather than fitted predictive-table pipelines,
  but they eliminate broad claims to the controlled-study, centroid, or cheap-
  closure templates.
- [Bregman information (AISTATS 2023)](https://proceedings.mlr.press/v206/gruber23a.html),
  [functional-output ANOVA (Technometrics 2010)](https://doi.org/10.1198/TECH.2010.10029),
  [functional-ANOVA purification (AISTATS 2020)](https://proceedings.mlr.press/v108/lengerich20a.html),
  [ML multiverse analysis (NeurIPS 2022)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/750337e1301941f81ae31a90e0a1c181-Abstract-Conference.html),
  and [PRESTO (ICML 2024)](https://proceedings.mlr.press/v235/wayland24a.html)
  occupy the underlying decomposition, interaction, and branch-sensitivity
  concepts, including multiverses over methods, hyperparameters, datasets, and
  latent representations. The surviving delta is the equivalence-restricted
  schema estimand and the measurement-to-action evidence chain.
- [Metamorphic testing of supervised classifiers](https://pmc.ncbi.nlm.nih.gov/articles/PMC3082144/)
  already tests expected behavior under semantics-preserving changes, including
  attribute and label transformations. OrbitANOVA is not the first invariance
  test; it must go beyond pairwise pass/fail checks through a declared product
  measure, aligned proper-risk value, interaction/randomness attribution, and
  prospective repair selection.
- [Design Choices That Matter (2026)](https://arxiv.org/abs/2608.04702)
  already uses fANOVA to attribute benchmark variation to architecture,
  initialization, fine-tuning, learning choices, and interactions. Merely
  crossing schema with seed or training factors is not novel; OrbitANOVA needs
  the zero-risk equivalence target, prediction-space proper-risk total, and
  tuning-decision path.
- [MAgg (ICML 2024)](https://proceedings.mlr.press/v235/wei24i.html),
  [learnable test-time augmentation (UAI 2020)](https://proceedings.mlr.press/v124/lyzhov20a.html),
  and [partial group augmentation theory (COLT 2026)](https://proceedings.mlr.press/v336/tahmasebi26a.html)
  occupy aggregation over metamorphic views, learned augmentation policies,
  and compute-limited group augmentation. OrbitCover can only be an audit-
  specialized action whose held-out allocation beats matched generic sampling.
- [Predictive multiplicity (ICML 2020)](https://proceedings.mlr.press/v119/marx20a.html)
  already studies disagreement among similarly accurate predictors. The
  narrower OrbitANOVA object is structured schema-induced multiplicity: the
  same complete pipeline produces every predictor from a declared equivalent
  spelling, so the disagreement has named nuisance factors and a zero target.

No search establishes absence of prior art. The paper should state that this
is the closest-work boundary found, not issue a novelty certificate.

## Likely reviewer objections

| Objection | What would answer it |
| --- | --- |
| “PREF plus ANOVA.” | Keep the quotient contract strict; show output alignment, exact risk value, interactions, and schema×randomness rather than generic preprocessing deltas. |
| “Not every factor is an orbit.” | Reserve orbit/Haar/symmetrization claims for the exact finite-group core. Call nonclosed chart and unit sets declared representative families with an explicit measure. |
| “The identities are textbook.” | Concede this in the abstract and sell a new estimand, benchmark finding, and predictive intervention. |
| “You chose transformations that make models look bad.” | Publish admissibility metadata, invariant controls, nuisance distributions, radius/reweighting sensitivity, and null closures before outcomes. |
| “Seed coupling is an implementation artifact.” | Separate persistent mean-predictor risk, same-seed operational coupling, and coupling-free distributional quantities. |
| “Averaging always helps, so what?” | Compare equal-compute iid ensembles, factor covers, covariant closure, and adaptive policies; require held-out transfer. |
| “HPO instability is already known.” | Demonstrate the equivalent-schema selection path, configuration-switch identity, menu×schema×split attribution, and a schema-specific repair. |
| “Effects are cherry-picked.” | Use the frozen 12--15 dataset panel, sequential screen, prospective promotion rules, many paired seeds, and dataset-level replication. |

## Decisive novelty experiment

The paper should make its strongest skeptical baseline executable. On the same
development cases, derive an action under four increasingly structured audits:

1. pairwise metamorphic violation counts;
2. PREF-style single-knob sensitivity/volatility;
3. undifferentiated total prediction variance; and
4. OrbitANOVA's full product, randomness, and tuning-decision decomposition.

Give all four audits the same preregistered action library: abstain; an iid
seed/checkpoint ensemble; an iid schema ensemble; a factor-balanced cover;
pooled HPO selection; and an available factor-specific covariant closure such
as feature-order symmetrization, field-VectorAdam, or FieldRiesz. Freeze how
each audit summary maps to that library on development cases. A simpler audit
may legitimately abstain or select a generic ensemble when it cannot identify
a factor, but its rule and hyperparameters must receive the same development
budget rather than being intentionally weakened.
Action eligibility must be determined from schema metadata before outcomes
and masked identically for every audit; a field-geometry closure, for example,
cannot be offered only on cases where FieldRiesz already looked favorable.
Eligible action fits may be cached once for offline evaluation, but each audit
policy must be blind to unselected action outcomes and charged the fit,
inference, storage, and tuning cost of the action it actually selects. Report
total research compute separately from that matched deployment frontier.

Freeze each audit-to-action rule and give it the same fit, inference, storage,
and tuning budget on unseen dataset/model or nuisance cases. OrbitANOVA earns
the extra framework only if its interaction or tuning-path information changes
the selected action and improves held-out residual schema risk or the matched
proper-risk frontier over the three simpler audits. If all four choose the same
repair, or if the structured choice does not transfer, the likely reviewer
summary is accurately “PREF plus ANOVA,” and the paper should not claim a new
design method.

Freeze two transfer modes separately. For **unseen nuisance levels within a
known dataset/model case**, compute the audit and choose the action only on the
development sub-orbit, then score residual risk on disjoint orbit levels. For
**unseen dataset/model cases**, learn the audit-to-action policy on development
cases; on a new case it may consume only the same budgeted train/validation
diagnostics, never test outcomes, before choosing an action. Report both modes
rather than letting easier within-case transfer stand in for cross-case
generalization.

By default, “train/validation diagnostics” also excludes test covariates:
label-free schema risk computed on test rows would still make the selector
transductive. Reserve both test rows and evaluation nuisance levels unless a
separate transductive-adaptation endpoint is explicitly declared.

With only 12--15 datasets, evaluate the second mode by an outer dataset-level
split or cross-fitting scheme fixed before audit outcomes. Keep every model
family from one dataset in the same outer fold; model cases are repeated
measurements, not independent task replications. Train the policy and all
thresholds on the other datasets, choose from train/validation diagnostics in
the held-out dataset, and aggregate the final contrast once per dataset.

## Submit gate

Submit OrbitANOVA only if all three conditions hold:

1. A frozen 12--15 dataset atlas finds material, heterogeneous schema risk in
   several conventional and modern model families after invariant controls.
2. Selection-path effects survive the full predeclared dataset × family panel,
   not only the currently selected sensitive cases.
3. At least one action chosen from development audits transfers to unseen
   dataset/model or nuisance cases at matched fit, inference, time, memory, and
   stored-model budgets.

Without condition 1, the phenomenon is too narrow. Without condition 2, the
tuning result is a case study. Without condition 3, the work is a strong audit
paper whose ICLR novelty is vulnerable.

## Protocol audit before freeze

The current broad-audit draft is close to executable, but six details should
be resolved in the final manifest:

1. Distinguish a **deterministic execution** from a **deterministic learner**.
   A fixed seed makes random forests and stochastic boosters reproducible; it
   does not remove seed as a scientific factor. Every promoted stochastic
   family needs the paired multi-seed decomposition even if its screen used one
   locked seed.
2. Give every schema action a machine-checkable admissibility record: inverse,
   metadata transport, target-output alignment, field span/reconstruction when
   applicable, and an invariant-pipeline null test. A human label such as
   “equivalent” is not enough.
3. Freeze discovery and confirmation units separately. Cases may be promoted
   from a cheap screen, but effect estimates and p-values for the promoted
   mechanism need new seeds/splits/tasks; repeated controls on a selected
   dataset are stress tests.
4. Reserve the held-out action-test dataset/model or nuisance cases before
   inspecting which factors dominate. Otherwise an apparently predictive
   audit-to-action result can be selected retrospectively.
5. Store the sampling axes explicitly in every prediction artifact
   (`dataset, pipeline, schema, seed, split, menu, row, output`). Never flatten
   overlapping menus, rotations, seeds, or query rows into fake replication.
6. For tabular foundation models, audit the released production wrapper and,
   where the implementation permits, a de-ensembled base path separately.
   Record internal feature/class shifts, preprocessing ensembles, context
   ordering, context length, checkpoint, and inference passes. An internal
   permutation ensemble is a model mechanism and compute cost, not an
   independent orbit replication.

The existing protocol already states most of these principles. They should be
enforced by the manifest and integrity tests, not left only in prose.

## FieldRiesz placement after Day 4

FieldRiesz says that a scalar field should be represented as a finite function
space. Training data supplies empirical mass `M`, declared semantics supplies
roughness `S`, and a cross-fitted anchor residual supplies

```text
h(x) = c^T (M + tau S)^(-1) phi(x).
```

The formulation is coherent and chart-covariant, but finite elements, spline
penalties, Riesz maps, topology matching, symmetry promotion, and
rotation-covariant optimization are established. Its exact tabular composition
may be novel, yet the present direct panel wins only 17/45 cells against raw
RAPLE, its spatial replication is mixed, and the strongest product result is
sensitive to rank completion and control orientation. It therefore remains a
useful OrbitANOVA intervention until independent semantic replication,
temporal benefit, and a reliable selector pass the frozen promotion gates.

## Best synthesis for one paper

Do not force FieldRiesz to win every benchmark. Use it as a targeted
factor-closure experiment inside OrbitANOVA:

1. On development cases, run OrbitANOVA and identify pipelines whose dominant
   removable component is a within-field chart factor or chart×seed
   interaction.
2. Freeze the action rule: apply a Riesz/whitened field interface and the
   transported metric or field-vector optimizer only when that factor profile
   crosses a predeclared threshold. Use ordinary PLE/training otherwise.
3. On unseen dataset/model cases, test whether the intervention removes the
   predicted chart component (for example, more than 95% closure) while
   preserving proper loss at matched fits, epochs, inference passes, and
   parameter budget.
4. Compare against indiscriminate FieldRiesz, ordinary seed/schema ensembling,
   whitening+SGD, and field-VectorAdam. The claim is that the audit chooses the
   right repair, not that the repair itself is a new generic optimizer.

This synthesis gives OrbitANOVA a mathematically explicit intervention and
gives FieldRiesz a realistic success criterion: factor-specific closure and
competitive loss, rather than universal SOTA performance.

## Decision

Do not search for another ornate formula merely to increase novelty. Execute
OrbitANOVA's broad audit and action-transfer gate. Continue FieldRiesz only as
a predeclared intervention within that program or as a separate project whose
promotion depends on new evidence.
