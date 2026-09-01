# E2 — Current-TFM phase diagram

Status: **small exploratory phase complete for TabICLv2 and Mitra; TabPFN-3
access-incomplete**.

## Frozen scope

The frozen `prior_dial_v1_1` panel uses seven rho values, six independent episodes per
rho (exactly one from each mechanism family), 64 context rows, 64 query rows, and eight
features in both classification and regression. Each model/episode has three independent
fits with the same seed: clean coordinates, a matched context/query random monotone-PWL
nuisance view, and an identity refit. The primary robustness statistic subtracts the
identity-refit disagreement.

TabICLv2 single/default and official Mitra classifier/regressor checkpoints completed all
252 model episodes (756 fits) with no failures. TabPFN-3 default/OOD remain named in the
frozen config but were not run: their official client/checkpoints and credentials are
not locally available, and TabPFN v2.5 was not substituted. This is consequently a
current-family screen, not the full confirmatory E2 requested by the program.

## Integrity

All six immutable bundles have the expected 42 episodes, complete rho/mechanism grids,
aligned metadata/CSV/array order, valid serialized transform states, finite labels and
predictions, and exactly recomputed prediction losses. Identity refits were deterministic
for every host/episode. Classification probability row-sum drift was at most `5.96e-8`;
derived probabilities were explicitly renormalized before recomputing losses and the raw
bundles were not altered.

## Results

The central result is a family separation in matched-reparameterization stability, not a
monotone performance curve over rho. Values below average all 42 paired episodes per task
type; CIs are diagnostic episode-bootstrap intervals for the paired TabICL-minus-Mitra
difference.

| Task | Model | Mean excess disagreement | Paired excess vs Mitra [95% CI] | Ratio to Mitra |
|---|---|---:|---:|---:|
| Classification (TV) | TabICLv2 single | 0.08956 | 0.08203 [0.07449, 0.08921] | 11.89x |
| Classification (TV) | TabICLv2 default | 0.07100 | 0.06347 [0.05680, 0.07012] | 9.43x |
| Classification (TV) | Mitra | 0.00753 | — | 1.00x |
| Regression (normalized absolute) | TabICLv2 single | 0.16053 | 0.14928 [0.12856, 0.17117] | 14.27x |
| Regression (normalized absolute) | TabICLv2 default | 0.13559 | 0.12434 [0.10816, 0.14207] | 12.05x |
| Regression (normalized absolute) | Mitra | 0.01125 | — | 1.00x |

Every one of the 42 paired cells had greater excess disagreement for either TabICLv2
mode than for Mitra in both task types. Default TabICL is more stable than single-mode
TabICL on average, but both remain well separated from Mitra. Mean matched loss gaps were
0.0240/0.0172/-0.0015 for TabICL single/default/Mitra in classification and
0.0080/0.0053/-0.0023 in regression. The six-episode per-rho clean-loss intervals are
wide, so no rho-dependent ranking claim is authorized.

This passes G2 in the program's explicitly scoped sense: two current TFM families occupy
materially and reproducibly different robustness points beyond their identity/inference
noise. It does not establish an architectural cause, universal TFM behavior, or a
TabPFN-3 result. The raw/rank simple-learner phase is reported separately in E1; the
matched raw/rank/fixed/gated comparison is the next E3 kill test.

Artifacts:

- `results/processed/e2_phase_summary_v2.csv`
- `results/processed/e2_family_contrasts_v2.csv`
- `results/processed/e2_integrity_v2.csv`
- `results/processed/e2_cells_v2.csv`
- `figures/e2_tfm_phase_v2.png`
