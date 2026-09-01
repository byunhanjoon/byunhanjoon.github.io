# Day-7 literature subtraction — tabular neural directions

Audit date: 2026-08-30. Primary papers and official proceedings were preferred.
This is a targeted collision audit, not proof of novelty.

## 1. The 2026 frontier changed the target

Tabular foundation models are now a central baseline, not an optional appendix.
TabICLv2 reports a scalable classification/regression model and stronger
TabArena/TALENT performance; TabH2O unifies both task types; Mitra and O'Prior
make synthetic-prior design itself a first-order research object. Recent OOD
testing reports systematic degradation across nine TFMs under distribution
shift. Consequently, a 2027 direction should either explain a failure mode that
these models do not solve or change their prior/decision rule in a theoretically
identifiable way.

Primary anchors:

- [TabICLv2](https://arxiv.org/abs/2602.11139)
- [TabICL](https://proceedings.mlr.press/v267/qu25d.html)
- [TabH2O](https://arxiv.org/abs/2605.18383)
- [Mitra](https://arxiv.org/abs/2510.21204)
- [O'Prior](https://arxiv.org/abs/2605.18971)
- [OOD evaluation of nine TFMs](https://arxiv.org/abs/2607.26000)
- [Statistical foundations of PFNs](https://proceedings.mlr.press/v202/nagler23a.html)

## 2. Why “add semantics/geometry” is occupied

Knowledge-Enriched Machine Learning already formalizes deterministic side
knowledge through concept kernels and provides KE-TALENT. Similarity encoding,
graph kernels, harmonic extension, Laplacian regularization, kriging/GPs,
hierarchy encodings, Nyström maps, metric embeddings, and cold-start transfer
cover nearly every representation ingredient used in Days 4–6.

- [Knowledge-Enriched Machine Learning for Tabular Data](https://proceedings.mlr.press/v288/kim25a.html)
- [Similarity encoding](https://doi.org/10.1007/s10994-018-5724-2)
- [Manifold regularization](https://jmlr.org/papers/v7/belkin06a.html)
- [Laplacian smoothing theory](https://proceedings.mlr.press/v51/sadhanala16.html)

Therefore MPE, FieldRiesz, a hierarchy embedding, a new landmark map, or a
generic metadata tokenizer is not a strong standalone novelty claim.

## 3. Why “predict when a representation helps” still has room

Recent work explains tabular methods through data uncertainty, and graph SSL
has explicit notions of safety. General negative-transfer and safe
semi-supervised-learning literatures are also relevant. These works prevent a
broad claim to inventing uncertainty-aware routing or safe graph use.

- [Unveiling the Role of Data Uncertainty in Tabular Deep Learning](https://arxiv.org/abs/2509.04430)
- [Graph-based semi-supervised learning through the lens of safety](https://proceedings.mlr.press/v161/sheshadri21a.html)
- [Relatively Smart learning](https://proceedings.mlr.press/v336/dughmi26a.html)
- [Testable learning with distribution shift](https://proceedings.mlr.press/v247/klivans24a.html)
- [Minimax regret under distribution shift](https://proceedings.mlr.press/v178/agarwal22b.html)
- [Model assessment under temporal shift](https://proceedings.mlr.press/v235/han24b.html)

The narrower residual not found in this audit is:

```text
an arbitrary neural tabular base over ordinary columns
+ optional deterministic knowledge attached to one field
+ complete unseen semantic states
+ residualization that asks what the field adds after the other columns
+ an exact zero-fallback help/harm decomposition
+ an impossibility result for metadata-only routing
+ a state-shift-aware certificate deciding whether to expose that bias.
```

The novelty would be the conditional *value and certification* of an inductive
bias, not its encoding. This distinction also answers why a correct metric can
hurt and why support distance, smoothness, or metadata quality alone cannot be
universal selectors.

“Certification” is itself occupied language: testable learning certifies test
performance from labeled source and unlabeled target data, while robust/minimax
regret methods protect against uncertainty sets. The lead must not claim a new
general theory of safe shift. Its remaining distinction is a certificate for
the *incremental residual value of a supplied field operator*, with the zero
neural adapter as comparator and semantic states—not rows—as the sampling unit.

## 4. Orbit/symmetry direction is now narrower

Functional ANOVA, proper-score variance decompositions, group averaging,
orthogonal arrays, antithetic Monte Carlo, stable selection, and recent
antithetic cross-validation theory occupy the ingredients of OrbitCover. The
remaining composition—complete-pipeline schema nuisances, aligned predictions,
and a quotient-risk estimator—is differentiated, but the 2026 collision space
and the observed target shift/convergence boundary reduce its upside.

Key anchors are retained in the existing
`road-to-iclr-day-05/RECENT_LITERATURE_AUDIT.md`.

## 5. Temporal/OOD and prior-design directions are crowded

Temporal tabular shifts and Fourier remedies already have a 2025 ICML paper;
generic TFM OOD benchmarking appeared in 2026. Likewise, “use a richer
synthetic prior” collides directly with Mitra and O'Prior. A viable successor
must introduce a precise family of *structured-field uncertainty tasks* and
prove what an in-context learner should infer—not merely add another synthetic
generator component.

- [Limits under temporal shift](https://proceedings.mlr.press/v267/cai25j.html)
- [Task-specific prompt theory for ICL](https://proceedings.mlr.press/v258/chang25b.html)

## 6. Search limitations

The lead spans small-area estimation, empirical Bayes, graph signal recovery,
safe transfer, hierarchical models, cold-start recommendation, spatial
statistics, and tabular foundation models. A submission needs an expert pass
through those communities. In particular, the exact algebra is classical; the
claim must rest on the tabular conditional-state formulation, certification
under a declared state shift, and broad neural evidence.

## 7. Architecture-transfer update and literature subtraction

The current benchmark frontier makes a one-backbone result untenable.
[TabArena](https://proceedings.neurips.cc/paper_files/paper/2025/hash/1697e3fb412da11dc9488249f9e7bbc9-Abstract-Datasets_and_Benchmarks_Track.html)
shows that rankings depend strongly on validation, tuning, and ensembling;
[TabReD](https://proceedings.iclr.cc/paper_files/paper/2025/hash/571799482291411607c54984153190b0-Abstract-Conference.html)
shows that recent tabular advances must be tested under realistic temporal and
industry conditions. NeurIPS 2025 also adds materially different foundation
model families trained on text-bearing tables
([TabSTAR](https://proceedings.neurips.cc/paper_files/paper/2025/hash/faf6e23e198314c7728eaa6ac44ae079-Abstract-Conference.html))
and real tables
([TabDPT](https://proceedings.neurips.cc/paper_files/paper/2025/hash/fc0e3f908a2116ba529ad0a1530a3675-Abstract-Conference.html)).
[TabFSBench](https://proceedings.mlr.press/v267/cheng25e.html) further
establishes feature shift as a distinct tabular failure mode across model
classes.

General safety language must remain narrow. Active, anytime-valid risk control
with label budgets already exists
([Xu et al., NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6eb05d8bc6bd7bb6868c64b5802125bd-Abstract-Conference.html)),
and adaptive model comparison under temporal shift is already formalized
([Han et al., ICML 2024](https://proceedings.mlr.press/v235/han24b.html)).
Rashomon-set work studies variation among good models, especially for fairness,
interpretation, and predictive multiplicity; it prevents claiming that model
specification uncertainty itself is new.

The narrower object not found in this targeted audit is the *uniform residual
value of an optional tabular field operator across a predeclared set of neural
bases*. This is a model-reliance question for an incremental inductive bias,
not a new Rashomon-set construction or generic risk-control method. The exact
collision risk remains high: a full paper needs a broader pass through model
class reliance, specification-curve, robust model selection, and safe-transfer
literatures before treating this residue as novel.

## 8. Direct collision with generic learned kernel relevance

The learned cycle result triggered a broader neural-process audit. It found a
material direct boundary that changes the Day-7 ranking:

- [ACE](https://proceedings.mlr.press/v258/chang25a.html) (AISTATS 2025)
  explicitly represents task latents, amortizes GP prediction, and includes
  latent kernel selection and runtime prior conditioning;
- [Flow Matching Neural Processes](https://proceedings.neurips.cc/paper_files/paper/2025/file/a92519f525c00085095fa41c5c46cdb5-Paper-Conference.pdf)
  train transformer neural processes over multiple GP kernel families;
- [task-agnostic amortized GP hyperparameter inference](https://proceedings.neurips.cc/paper/2020/hash/f52db9f7c0ae7017ee41f63c2a7353bc-Abstract.html)
  already uses a transformer to infer kernel structure across tasks;
- recent theory explicitly studies the costs of amortizing GP inference with
  neural processes
  ([Young, 2026](https://proceedings.mlr.press/v341/young26a.html)).

Therefore the claim “a transformer treats structure relevance as a latent task
variable” is occupied in generic stochastic-process form. The Day-7 learned
experiment remains a valid implementation and phase-boundary result, but its
novelty can only rest on a narrower tabular composition: an arbitrary ordinary-
column base, an external semantic-field operator, residual incremental value,
and complete unseen states. That composition has not yet been demonstrated by
the synthetic cycle experiment, so it is demoted from lead status.
