# Road to ICLR — Day 4 research record

Day 4 asks whether the representation sensitivity from Days 1–3 can become a
novel, performance-bearing tabular method. The explored Day 4 technical direction is
**FieldRiesz**: model each scalar field as a measured function space with
empirical mass `M`, semantic stiffness `S`, and Riesz operator `K=M+tau S`.
The most promising performance extension is the cross-fitted residual Riesz
representer `h(x)=c^T K^-1 phi(x)`, which connects this geometry to the existing
RAPLE TabReD results in the adjacent project.

Late Day 4 continuation: [`SEMANTIC_MULTIVIEW_REPORT.md`](SEMANTIC_MULTIVIEW_REPORT.md)
tests a fully neural, LightGBM-free contrastive direction across PLE, MLP,
ResNet, and FT-Transformer on official temporal Weather and Cooking Time
splits. The full-row VICReg variant is falsified as a broad default: it wins
1/6 cells versus PLE (-0.69% mean RMSE gain), while correct cyclic geometry is
only +0.03% better than a permuted-geometry control. Its predeclared
PLE-preserving follow-up is recorded in
[`FIELD_LOCAL_DISTILLATION_REPORT.md`](FIELD_LOCAL_DISTILLATION_REPORT.md).
It clears only 3/6 validation cells versus 5/6 required, so no test metrics or
Delivery ETA transfer were run and the contrastive branch is stopped.

The Day 1 exact-identity signal was also tested against Q-PLE, fixed T-PLE, and
an equal-parameter quantile-bin embedding in
[`SUPPORT_IDENTITY_TRANSFER_REPORT.md`](SUPPORT_IDENTITY_TRANSFER_REPORT.md).
A gated exact-support residual token fails its frozen architecture gate in all
six Weather/Cooking-Time MLP, ResNet, and FT-Transformer cells. Delivery ETA
was therefore not opened for this method. The result rejects a target-free
low-cardinality support token as the general explanation of the Adult gain;
generic discrete capacity often matches it, and it is not reliably additive
to T-PLE.

The follow-up mechanism audit in
[`ADULT_IDENTITY_MECHANISM_REPORT.md`](ADULT_IDENTITY_MECHANISM_REPORT.md)
shows that Adult exact support succeeds across all target-free/supervised and
additive/separate variants on MLP, ResNet, and FT-Transformer. Exact levels,
not the original selector or interface, are the stable Adult mechanism. A
training-only signal selector with exactly zero-start gates is evaluated in
[`SIGNAL_GATED_SUPPORT_REPORT.md`](SIGNAL_GATED_SUPPORT_REPORT.md): it improves
the transfer panel from 0/6 to only 1/6 passing cells, so it is safe but not a
general performance method. A zero-start attention–MLP hybrid is best among the
pilot backbones on Cooking Time but not Weather.

The type-agnostic follow-up is documented in
[`UNIVERSAL_RANK_REPORT.md`](UNIVERSAL_RANK_REPORT.md). Exact/hash identity,
rarity positional encoding, generic cyclic Fourier features, and atom-interval
PLE all fail their broad or transfer gates. Empirical-midrank PLE is useful as a
complementary chart but fails its 33-cell standalone confirmation. The resulting
shared multi-chart method is evaluated in
[`TRICHART_REPORT.md`](TRICHART_REPORT.md): Q-PLE, T-PLE, and midrank tokenizers
share one backbone and receive supervised prediction-consistency training.
Across four datasets, three seeds, and the practical MLP/ResNet/FT panel it wins
29/33 validation cells versus Q-PLE and 28/33 versus T-PLE, but misses its strict
gate because Maps Routing is negative on average versus T-PLE.

The subsequent safe-fusion experiment, also recorded in
[`TRICHART_REPORT.md`](TRICHART_REPORT.md), freezes the trained T-PLE predictor
and learns a zero-start residual from Q-minus-T and midrank-minus-T token
charts. This **AnchorTriChart** variant passes its three-seed regression gate:
33/33 validation-safe cells, 25 strict wins, and positive mean descriptive test
gain on all four datasets. It also passes a three-seed binary-classification
replication on Adult, Churn, and Higgs-small: 27/27 validation-safe cells and
26/27 test wins. Adult's exact-support encoder nevertheless remains clearly
better, so atom identity and multi-chart correction are distinct mechanisms.

The later capacity audit changes that interpretation. A frozen residual built
from another T-PLE model matches AnchorTriChart: the chart-specific residual
wins only 6/11 validation cells over this capacity control (+0.055% mean), and
a per-field chart gate falls to 3/11 (-0.003%). AnchorTriChart is therefore a
useful two-model cascade, not established evidence for Q/rank charts.

The equal-compute continuation is reported in
[`HETEROPLE_REPORT.md`](HETEROPLE_REPORT.md). A fixed T+Q classification /
T+midrank regression policy wins 10/12 cells on a fresh prospective panel but
fails its full mean-gain gate because of one near-zero-loss classification
outlier. The exact-compute HeteroBag-3 follow-up replaces one member of T+T+T
with Q for classification or midrank for regression. On another untouched
panel it passes every frozen clause: 10/12 test wins, +0.843% mean gain,
positive task-family means, and positive dataset means on 4/4 datasets. The
sharper Adult control also passes: atom-aware T+Q beats atom-aware
T+T in 2/3 MLP/ResNet/FT validation cells (+0.424% mean), and is also positive
in a post-gate attention+MLP hybrid (+0.293%). This supports complementarity
between exact atom identity and coordinate-chart diversity.

Portfolio note: OrbitANOVA, introduced in the public
[`Day 3 post`](../../blogposts/road-to-iclr-day-03.html), remains the primary
ICLR paper. FieldRiesz is a conditional secondary direction or a covariant
intervention for that paper; Day 4 does not supersede the primary freeze.
In one line, OrbitANOVA measures the proper-loss cost of arbitrary but
equivalent schema spellings, attributes it to nuisance factors and randomness,
and uses that audit to choose a targeted repair.

Start with:

- [`HETEROPLE_REPORT.md`](HETEROPLE_REPORT.md): equal-compute heterogeneous-view
  audits, the failed broad fixed-policy gate, and the passing Adult exact-atom
  additivity result;
- [`TRICHART_REPORT.md`](TRICHART_REPORT.md): the historical frozen-anchor
  result, its later capacity-control correction, Adult atom boundary, and the
  earlier shared-model result;
- [`UNIVERSAL_RANK_REPORT.md`](UNIVERSAL_RANK_REPORT.md): the type-agnostic
  rank, identity, cyclic, and atom-interval experiments that motivate TriChart;
- [`ADULT_IDENTITY_MECHANISM_REPORT.md`](ADULT_IDENTITY_MECHANISM_REPORT.md):
  the selection-by-interface audit that isolates Adult's exact-support signal;
- [`SIGNAL_GATED_SUPPORT_REPORT.md`](SIGNAL_GATED_SUPPORT_REPORT.md):
  cross-fitted field selection, zero-start gates, and attention–MLP extension;
- [`SUPPORT_IDENTITY_TRANSFER_REPORT.md`](SUPPORT_IDENTITY_TRANSFER_REPORT.md):
  exact-support versus equal-capacity bin embeddings and the stopped transfer;
- [`FIELD_LOCAL_DISTILLATION_REPORT.md`](FIELD_LOCAL_DISTILLATION_REPORT.md):
  the validation-only field-local follow-up and stopping decision;
- [`SEMANTIC_MULTIVIEW_REPORT.md`](SEMANTIC_MULTIVIEW_REPORT.md): the neural-only
  contrastive continuation, controls, two-dataset results, and next gate;
- [`PORTFOLIO_NOVELTY_REVIEW.md`](PORTFOLIO_NOVELTY_REVIEW.md): plain-language
  OrbitANOVA verdict, reviewer scorecard, closest-work boundary, decisive
  novelty experiment, and submit gates;
- [`ICLR_DIRECTION_FREEZE.md`](ICLR_DIRECTION_FREEZE.md): the proposed paper
  spine, claim boundary, model interfaces, and promotion gates;
- [`day4.md`](day4.md): the readable research post and ICLR verdict;
- [`THEORY_FIELDRIESZ.md`](THEORY_FIELDRIESZ.md): formal construction and
  chart-covariance proposition;
- [`LITERATURE_MAP.md`](LITERATURE_MAP.md): recent literature and novelty
  boundary;
- [`CANDIDATE_MATRIX.md`](CANDIDATE_MATRIX.md): promoted and falsified ideas;
- [`results/day4_summary.json`](results/day4_summary.json): machine-readable
  headline summary.

## Reproduce the summaries

From the repository root:

```bash
/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python \
  experiments/road-to-iclr-day-04/analyze_day4.py
```

This writes:

- `day4_architecture_transport.csv`;
- `day4_broad_mass_screen.csv`;
- `day4_california_geometry.csv`;
- `day4_california_field_ablation.csv`;
- `day4_bike_cyclic.csv`;
- `day4_residual_riesz.csv`;
- `day4_residual_vs_raple.csv`;
- `day4_maps_sparse_screens.csv`;
- `day4_strict_selector.csv` and `day4_fdr_selector.csv`;
- `day4_wrong_geometry_permutations.csv`;
- `day4_isospectral_control.csv` and `day4_isospectral_rotations.csv`;
- `day4_semantic_spectral_profile.csv`;
- `day4_spatial_product_riesz.csv` and
  `day4_spatial_surface_sensitivity.csv`;
- `day4_spatial_surface_strength.csv` and
  `day4_spatial_surface_validation_selection.csv`;
- `day4_spatial_reference_mass.csv`, `day4_spatial_rho_mixture.csv`, and
  `day4_spatial_reference_selection.csv`;
- `day4_spatial_reference_rotations.csv`;
- `day4_temporal_checks.csv`; and
- `day4_summary.json`.

Verify the algebraic covariance claims with:

```bash
PYTHONDONTWRITEBYTECODE=1 \
/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python \
  -m pytest -q -p no:cacheprovider \
  experiments/road-to-iclr-day-04/test_fieldriesz.py \
  experiments/road-to-iclr-day-04/test_day4_artifacts.py \
  experiments/road-to-iclr-day-04/test_semantic_multiview.py \
  experiments/road-to-iclr-day-04/test_support_identity_transfer.py
```

The self-contained Day 4 suite is checked by the command above. The adjacent exploratory
OrbitANOVA record has a separate 17-test identity suite when it is available
locally.

## Pilot entry points

- `adult_exact_multiview_pilot.py`: atom-aware T+Q versus atom-aware T+T with
  equal model count and parameter budget, including optional hybrid transport;
- `heterobag_three_member_pilot.py`: prospective T+T+alternate versus T+T+T
  with exactly matched three-model compute;
- `analyze_heterobag_three_member.py`: frozen HeteroBag-3 gate and task/dataset
  summaries;
- `multiview_equal_compute_pilot.py`: T+Q/T+midrank versus T+T on frozen OpenML
  panels with fixed 50/50 averaging;
- `analyze_multiview_equal_compute.py`: developmental, selection, and fixed-
  policy panels with the untouched prospective gate;
- `trichart_shared_pilot.py`: shared-backbone Q-PLE/T-PLE/midrank fusion with
  supervised per-view loss and optional prediction consistency;
- `trichart_frozen_anchor_pilot.py`: frozen T-PLE regression anchor plus a
  zero-start residual over Q/T/midrank chart differences;
- `trichart_frozen_anchor_classification.py`: matched binary-classification
  replication on Adult, Churn, and Higgs-small;
- `analyze_trichart_frozen_anchor.py`: consolidated three-seed regression,
  classification, and Adult exact-support comparison;
- `analyze_trichart.py`: frozen 33-cell confirmation, independent-ensemble
  reference, alignment ablation, and machine-readable decision;
- `universal_mass_identity_pilot.py`: type-agnostic midrank, identity/mass,
  cyclic-control, and empirical-CDF atom-interval tokenizers;
- `analyze_universal_rank.py`: standalone rank confirmation and stopped
  geometry branches;
- `semantic_multiview_pilot.py`: neural-only paired PLE/topology charts with
  MLP, ResNet, FT-Transformer, no-alignment, and wrong-cycle controls;
- `analyze_semantic_multiview.py`: frozen two-dataset comparison and decision;
- `field_local_distillation_pilot.py`: exact-PLE, zero-gated semantic residuals
  applied only to declared fields, evaluated on validation only;
- `support_identity_transfer_pilot.py`: parameter-matched Q-PLE, T-PLE,
  exact-support residual tokens, and equal-capacity bin-token controls across
  MLP, ResNet, and FT-Transformer;
- `analyze_field_local_distillation.py`: predeclared 5/6 promotion gate;
- `support_heat_pilot.py`: MLP/ResNet, support, mass, Riesz, fractional-mass,
  permuted-geometry, and TabReD experiments;
- `tabm_support_pilot.py`: parameter-matched TabM transport;
- `ft_support_pilot.py`: one-token-per-field Transformer transport;
- `bike_cyclic_pilot.py`: three-seed chronological UCI Bike Sharing test of
  Hour mass, path, correct ring, and permuted-ring geometry;
- `residual_riesz_pilot.py`: shared out-of-fold LightGBM anchor plus mass,
  raw RAPLE, correct-Riesz, mass, node-permuted, and exact generalized-
  spectrum-preserving residual controls, plus sparse selectors;
- `residual_spectral_profile.py`: strength-swept, chart-invariant residual-
  energy retention against repeated node and `M`-isospectral controls;
- `residual_gain_certificate.py`: simultaneous i.i.d. calibration lower bounds
  for an additive anchor-plus-intervention predictor (not neural retraining or
  temporal shift);
- `residual_isospectral_null.py`: analytic/simulated Haar-orientation reference
  for a fixed residual-retention curve;
- `king_county_spatial_pilot.py`: predeclared Latitude/Longitude replication on
  chronological OpenML King County house sales;
- `spatial_product_riesz_pilot.py`: declared two-field residual surfaces with
  empirical-ANOVA purification, joint spectral calibration, separable product
  or train-only haversine graph stiffness, exact-spectrum controls, and
  chart-covariant reference-mass completion for empirical-null modes;
- `synthetic_reference_completion.py`: 200-repetition missing-interval sanity
  check for completion and full-space controls (not benchmark evidence);
- `support_diagnostic.py`: train-only atom/support activation diagnostic;
- `innovation_pilot.py`: falsified global common/innovation direction.

The repository's default `python` is Python 2. Use the explicit environment
above. The validated snapshot is Python 3.10.16, NumPy 1.26.4, pandas 2.3.3,
SciPy 1.11.4, scikit-learn 1.4.2, PyTorch 2.7.0+cu126, LightGBM 4.7.0, and
pytest 9.0.2. Generated result tables are derived from the checked-in raw CSVs;
no test labels are used to construct support nodes or mass/stiffness operators.
Summary regeneration and tests require only the checked-in artifacts. Pilot
reruns that use local RAPLE or TabReD arrays can override the preserved machine
defaults with `DAY4_RAPLE_ROOT`, `DAY4_TABRED_LEGACY_ROOT`, and
`DAY4_WEATHER_ROOT`. The underlying licensed/local datasets are not copied into
this directory.

Some early pilot CSVs rely on script defaults or shared experiment metadata
rather than a one-to-one metadata companion. Before a publication freeze, add
an immutable manifest with dataset provenance and hashes, split construction,
model/configuration, seed, control family, source artifact, and software
environment for every retained fit. The current bundle is an exploratory
research record, not yet an archival benchmark release.

## Current claim boundary

The California result supports a semantic field-geometry hypothesis. Adult and
Black Friday support a mass-normalization hypothesis. Churn, broad academic
screens, and official TabReD temporal checks show that unconditional
application is not viable. In the complete direct shared-anchor panel over five
datasets, three backbones, and three seeds, dense semantic Riesz beats raw
RAPLE in only 17/45 cells (-0.15% mean), while beating one node-permuted control
in 33/45 (+0.57%). Repeated node and exact `M`-isospectral controls strengthen
the California/Weather mechanism result: the semantic operator wins 80/90
control comparisons (+0.90%) over five exact rotations and three backbones,
or 18/18 unique cells after averaging controls within cell. Strength probes
also show sensitivity to `tau`. Because California/Weather were selected after
the broad pilot, 18/18 has no confirmatory p-value and does not replace
independent dataset replication. The selectors do not yet turn the mechanism
into broad performance. A predeclared chronological King County spatial
replication is mixed/negative, so California is not yet independently
replicated. The first product-field residual surface improved California but
failed its wrong-geometry control. Empirical-ANOVA purification and joint
spectral calibration repair the California control hierarchy (8/9 versus raw,
wrong, and finite-support isospectral controls), but the frozen King County transfer
still loses its finite-mass-spectrum control on average. A group-token pilot is
positive versus raw on both spatial datasets but has contradictory wrong-
geometry/anchor outcomes. The original product control is rank-confounded
(empirical rank 69 versus semantic rank 144). Reference-mass completion makes
the full 144-mode comparison exact, but the semantic gap changes sign over
`rho={0.001,0.01,0.1}` and shrinks to +0.04% after averaging five full-space
rotations at `rho=0.01`. This is a differentiated research direction and a
useful audit construction, not a finished ICLR or SOTA claim.
