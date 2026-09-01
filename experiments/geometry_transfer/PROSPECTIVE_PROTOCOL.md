# Prospective protocol — frozen before outcome acquisition

Frozen on 2026-08-29 after retrospective Gate R passed and before any source
outcome was downloaded or inspected for this program.

## Scientific question

Can nested training-state-only evaluation predict the sign and magnitude of
geometry benefit on outer states whose outcomes remain unseen until every
prediction is sealed?

## New source panel

1. **NOAA GHCN-Daily** (`noaa_ghcn_tmax`): state is weather station; geometry
   is target-independent great-circle station distance; target is daily TMAX
   in 2024; ordinary covariates are calendar harmonics and elevation. Select up
   to 30 continental-US stations deterministically by a hash of station ID from
   stations whose inventory reports at least 300 TMAX days in 2024. Completeness
   may select a station; predictive performance may not.
2. **UCI Beijing Multi-Site Air Quality** (`beijing_pm25`): state is monitoring
   station; geometry is great-circle distance from fixed published station
   coordinates; target is PM2.5; ordinary covariates are date/time, weather,
   wind direction, and meteorology. Rows with missing target are removed.
3. **Chicago Divvy 2020 Q1** (`divvy_trip_duration`): state is start station;
   target is log trip duration; ordinary covariates are start time, rider type,
   and end station. Primary geometry is station great-circle distance. The
   domain-specific operator uses the target-independent trip-connectivity graph
   with geodesic edge lengths; no duration values enter the graph.
4. **BLS May 2024 state OEWS** (`bls_oews_wage`): state is detailed SOC
   occupation; geometry is shortest path in the SOC prefix hierarchy; target is
   log annual mean wage; ordinary covariates are geographic area and published
   employment. Suppressed/non-numeric targets are removed. This is a new
   official hierarchy, not the ACS occupation microdata used retrospectively.

If a frozen URL is unavailable, record `NOT RUN — SOURCE UNAVAILABLE`; do not
replace it after inspecting another source's outcomes. A source may be skipped
for fewer than eight usable states or fewer than 500 usable rows. No replacement
source is authorized.

## Frozen splits and models

- Three outer state splits per source, seeds 7601, 7602, 7603.
- Each split assigns 70% states to observed training and 30% to sealed test,
  stratified only by deterministic state hash/order where applicable.
- Three inner state folds partition outer training states.
- Base model: CatBoost regression excluding semantic state; 100 trees, depth 7,
  learning rate 0.08, L2 5, fixed seed, at most 75,000 fitting rows with at
  least one row retained per training state.
- Training residual means are three-fold row-OOF within the currently observed
  state set. Diagonal Sigma is sample residual variance divided by state count.
- Operator hyperparameters use training geometry only; no outcome-based
  bandwidth selection.

## Frozen operators

1. inverse-distance metric 3-NN;
2. row-normalized Gaussian RBF at median observed-state pair distance;
3. domain-specific: harmonic metric-graph extension for NOAA/Beijing/Divvy and
   kernel-ridge path interpolation for BLS;
4. zero-residual geometry-free fallback.

The operators are fixed linear maps conditional on the observed states. The
primary aggregates are source × operator means over the three outer splits.

## Sealing procedure

For every outer split, first compute and write an immutable prediction record
containing inner-fold predicted gain, uncertainty, support heuristics, source,
split, operator, and a hash. Only then may the outer-test target be joined to
compute actual gain. Outer outcomes never enter base fitting, residual means,
Sigma, operator construction, or the predicted benefit.

## Frozen outcomes and criteria

- Primary loss: state-balanced squared error on the source-standardized target.
- P1: source/operator aggregate Spearman(predicted, actual) >= 0.60.
- P2: sign accuracy >= 75%.
- P3: nested Geometry Transfer estimate exceeds support distance, cover radius,
  and raw-smoothness predictors under leave-one-source-out evaluation.
- P4: at least three independent source families show qualitatively correct
  behavior (positive within-source association or >=2/3 operator signs).
- P5: if any operator is harmful, its prediction is nonpositive or in the
  central near-zero uncertainty bin.

All cells and unavailable sources remain in the audit. No criterion or operator
may be changed after the first outer outcome is evaluated.
