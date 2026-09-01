# Frozen OpenML breadth follow-up: competence routing

Frozen: 2026-09-01, after the seven-dataset external gate failed for insufficient breadth
and before any routing outcome on the dataset identities below.

## Fixed panel

Use every compatible binary/regression identity in the pre-existing Day-8 OpenML panel
that was not observed in the seven-dataset check.

- classification: seismic-bumps (363700), heloc (363676),
  credit_card_clients_default (363627), APSFailure (363616), bank-marketing (363618),
  diabetes (363629);
- regression: wine_quality (363708), miami_housing (363686), Food_Delivery_Time
  (363672), airfoil_self_noise (363612), concrete_compressive_strength (363625),
  physiochemical_protein (363693), superconductivity (363705).

Churn and diamonds are excluded because their identities were already inspected. No
compatible unseen identity is removed or replaced based on performance. A technical
failure is recorded rather than substituted.

## Data and episodes

Use official OpenML task repeat 0/fold 0. Fit median imputation and z-scoring on the
official outer-train rows only. Retain numeric columns only; discard structurally
all-missing training columns and then take the first 32 numeric columns in OpenML source
order. This cap was fixed after a label-free structural audit found up to 170 numeric
features and prevents the quadratic expert from changing the compute class.

For every dataset, use 40 paired episodes from seed family 135001, each with 96 context
rows from outer train and 128 query rows from official test. Classification sampling is
stratified. Regression targets are standardized by outer-train mean and standard
deviation. Store exact task/dataset IDs, source checksum, selected columns, and split
hashes.

## Methods and inference

Use the unchanged six experts, three-fold context CV, synthetic-development fixed
weights, and synthetic-development competence temperature/shrinkage. Comparators are
uniform, hard CV selection, and per-episode best-individual oracle. There is no OpenML
tuning and no dataset-identity input.

Primary contrast and gate match `REAL_PANEL_COMPETENCE_PROTOCOL.md`: dataset-balanced
fixed-minus-competence gain with 10,000 paired hierarchical bootstrap draws over
datasets then episodes. Strong transfer needs both task intervals positive. Scoped
transfer needs one positive interval and no harm worse than 0.005 log loss or 0.02 MSE
in the other.

This is an untouched dataset-identity breadth follow-up. A pass supports transfer only
to this numeric, small-context panel; it does not establish mixed-type performance,
state-of-the-art accuracy, or novelty of competence weighting.
