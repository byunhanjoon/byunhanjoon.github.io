# Road to ICLR — Day 4 research record

Day 4 asks whether the representation sensitivity from Days 1–3 can become a
novel, performance-bearing tabular method. The explored Day 4 technical direction is
**FieldRiesz**: model each scalar field as a measured function space with
empirical mass `M`, semantic stiffness `S`, and Riesz operator `K=M+tau S`.
The most promising performance extension is the cross-fitted residual Riesz
representer `h(x)=c^T K^-1 phi(x)`, which connects this geometry to the existing
RAPLE TabReD results in the adjacent project.

Portfolio note: OrbitANOVA, introduced in the public
[`Day 3 post`](../../blogposts/road-to-iclr-day-03.html), remains the primary
ICLR paper. FieldRiesz is a conditional secondary direction or a covariant
intervention for that paper; Day 4 does not supersede the primary freeze.
In one line, OrbitANOVA measures the proper-loss cost of arbitrary but
equivalent schema spellings, attributes it to nuisance factors and randomness,
and uses that audit to choose a targeted repair.

Start with:

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
  experiments/road-to-iclr-day-04/test_day4_artifacts.py
```

The self-contained Day 4 suite contains 26 tests. The adjacent exploratory
OrbitANOVA record has a separate 17-test identity suite when it is available
locally.

## Pilot entry points

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
