# Experiment log

## 2026-08-31 — repository and environment audit

Question: can the prescribed program start from the current workspace without overwriting prior work?

Findings:

- The pre-existing top-level Day 8 artifacts implement a different retrieval/geometry direction. They are preserved unchanged. The reparameterization program is isolated under the required `configs/`, `src/`, `scripts/`, `tests/`, `results/`, and `reports/` paths.
- Two NVIDIA H100 NVL GPUs were available with approximately 95 GB free each at audit time.
- The active research interpreter is `/home/byunhanjoon/miniconda3/bin/python` (Python 3.10.16). The system `python` is Python 2.7 and must not be used.
- Installed core versions: PyTorch 2.7.0+cu126, TabPFN 6.3.0 with local TabPFN-2.5 classification/regression checkpoints, TabICL 2.0.3 with cached TabICLv2 checkpoints, XGBoost 2.1.4, CatBoost 1.2.10, LightGBM 4.7.0, scikit-learn 1.4.2, OpenML 0.15.1.
- TabPFN-3 is not available as a local open checkpoint through the installed package. The official v3 client requires an authenticated account/API route; this is recorded as a model-access limitation, not silently relabeled as TabPFN-3.
- Mitra weights are public through official AutoGluon model cards, but the AutoGluon Mitra integration is not installed in the shared interpreter. It will be isolated rather than changing the shared environment destructively.

Implemented before model outcomes:

- Explicit transform API with metadata, missingness preservation, train-only fitting, state serialization, forward/inverse audits, affine, order reversal, signed power, asinh, random monotone PWL, held-out monotone spline, empirical-CDF, Gaussian-rank, atomic-spacing, and composition transforms.
- Frozen-split task loader, train-only marginal descriptors, and four-way frame construction.
- Classification/regression performance, calibration, and posterior-disagreement metrics.
- Explicit minimal/default TabPFN-2.5 and TabICLv2 adapters plus XGBoost, CatBoost, and LightGBM controls.
- Immutable per-job prediction bundles, JSON metadata, checksums, provenance, telemetry, and append-only manifest logic.

Verification:

- `PYTHONPATH=. /home/byunhanjoon/miniconda3/bin/python -m pytest -q`
- Result at this point: 21 passed. The first run exposed and led to correction of an invalid multiclass XGBoost objective on binary tasks; the full suite then passed.

Decision: proceed to the Tier 0 runtime smoke. Do not implement RSPF or inspect confirmatory data.

## 2026-08-31 — Tier 0 smoke and Phase I seed freeze

Question: does the four-way implementation distinguish a matched task reparameterization from a context/query mismatch, preserve exact transforms, and provide a suitably insensitive control?

Findings:

- All 12 smoke jobs completed and their raw prediction bundles passed checksum-validated cache reuse.
- XGBoost produced exactly identical predictions under matched positive-affine and monotone-PWL transforms on both smoke datasets.
- TabPFN-2.5 single-estimator predictions were unchanged under identity, mildly changed under matched affine/PWL transforms, and much more strongly degraded by context-only/query-only mismatch. This is implementation evidence only, not Phase I thesis evidence.
- Transform inverse relative errors were at floating-point precision. The tested four-way separation therefore is not explained by information loss or context/query mix-up.
- Three Phase I model/warp seeds, `[20260831, 20260832, 20260833]`, were frozen before any OpenML Phase I outcome. The fixed data split remains `20260831` so seeds quantify inference/warp variation rather than silently changing task membership.
- The transform library now includes an exact categorical-label permutation control over declared pandas categorical support, including string and integer-coded categories, missingness, metadata preservation, serialization, and round-trip/equality-class audits.

Decision: proceed to model integration and one-task OpenML validation before launching the frozen Phase I grid. No G1 decision has been made.

## 2026-08-31 — isolated Mitra integration and four-way fit correction

Question: can official Mitra be evaluated without destabilizing the shared environment, and are the four protocol cells paired at the fitted-context level?

Findings:

- AutoGluon Tabular 1.6.1 with its Mitra extra was installed under `/data/byunhanjoon/reparam_mitra_env`. AutoGluon's initial PyTorch 2.13 CUDA 13 wheel could not initialize on the host driver, so only this isolated environment was corrected to PyTorch 2.10.0+cu128. CUDA then detected an NVIDIA H100 NVL successfully.
- End-to-end classifier ICL, classifier default fine-tuning, and regressor default fine-tuning smokes all returned finite predictions. Official Hugging Face revisions and SHA-256 checkpoint identities are captured by the worker. Default fine-tuning uses one estimator, 50 steps, and bfloat16, matching AutoGluon's exposed defaults rather than silently substituting ICL-only inference.
- A first full-runner OpenML identity cell completed all four independent fits but then hit a provenance-bookkeeping `UnboundLocalError`. The manifest records this failure and its partial raw bundle remains untouched. The defect was corrected and covered by a unit test.
- Inspection of that identity bundle showed nonzero prediction differences between independent Mitra fine-tunes despite the same nominal seed. Therefore, fitting independently for all four cells would confound query-only/context-only comparisons with optimization nondeterminism.
- The runner now performs two fitted-context groups per job: one original-context fit predicts clean and query-only, and one transformed-context fit predicts matched and context-only. Mitra serves both queries from the same fitted predictor in sequence. An identical-query integration test produced bit-identical predictions, confirming the shared-fit path.
- All 12 pilot tasks load successfully. Their split-disjointness, exact split IDs, schemas, and train-only descriptors are stored in the immutable config-keyed panel artifact under `results/panel/`.

Decision: preserve identity refit variation as an explicit noise baseline, use shared fits within each context representation, and rerun the OpenML identity cell before any transformed Phase I cell.

## 2026-08-31 — frozen Phase I kill test and Gate G1

Question: do current TFMs change predictions under fully matched task-isomorphic numerical reparameterizations after ruling out mismatch, transform loss, and ordinary refit noise?

Execution and integrity:

- Completed the frozen 12-dataset × 7-model × 13-setting × 3-seed grid: 3,276/3,276 complete, zero missing/failed/unavailable under config digest `5c75900a...` and prediction code digest `595cc6ae...`.
- Raw bundles passed checksums, finite-prediction checks, row alignment, shared-fit pairing, missingness preservation, and numerical inverse validation.
- The first analyzer pass exposed that `1e-9` was an unrealistically strict float64 inverse threshold. Full-grid inspection found a maximum `3.51e-7` spline round-trip error and zero records above `1e-6`. The validator now uses a tested `1e-6` numerical inversion tolerance.
- A second pass exposed 2–6 positive-affine order ties for seven transform instances on the student-dropout task. Direct reconstruction showed only adjacent one-ULP encodings of semantically identical decimal grades, with round-trip error at most `2.9e-15`. The tested validator permits such ties only below `1e-12` and continues to reject material collisions.
- The seismic `ghazard` column is all-missing in the frozen split; it caused descriptor warnings but no information/missingness change. Excluding the dataset does not remove the TabICLv2 result.
- Final verification: 56 tests passed; analysis emitted dataset-level 10,000-draw bootstrap tables and three plots in `results/analysis/phase1/595cc6ae397a9a55/`.

Findings:

- TabICLv2-default classification excess JS is `0.000752` [0.000363, 0.001350], about 1.03 percentage points total variation and 0.80% argmax flips; accuracy changes by -0.21 percentage points [-0.35, -0.08].
- TabICLv2-default regression predictions differ by `0.02724` normalized absolute units [0.01405, 0.04384], with a small positive normalized loss gap.
- Default inference reduces but does not eliminate TabICLv2 sensitivity. Nonlinear effects persist at severities `<=1` and after excluding affine transforms or the seismic task.
- Mitra performance is unchanged and excess disagreement is near its independent-refit noise floor. Classification tree controls are roughly six to eight times quieter than TabICLv2-default; regression controls are not perfectly invariant.
- Context/query mismatch remains tens to hundreds of times larger than matched effects, confirming that the four-way protocol distinguishes the phenomena.

Decision: Gate G1 passes narrowly through Route B for TabICLv2 posterior instability. Route A fails because two current strong families do not show systematic degradation. Cross-family TabICLv2/Mitra heterogeneity is a strong Route C signal but remains mechanistically unexplained, so Gate G2 is open. Proceed to Phase II/III and controlled explanation work; do not implement RSPF or expensive pretraining yet. See `reports/PHASE1_KILL_TEST.md`.

## 2026-08-31 — Phase II/III protocol freeze and infrastructure validation

Question: is the larger audit frozen, split-aware, and executable across the prescribed model and transform controls before inspecting any new development-suite outcome?

Findings:

- A 20-task development suite was selected from official TabArena curation metadata using task/schema metadata only and frozen in `reports/DEVELOPMENT_SUITE_FREEZE.md`. It contains eight regression, seven binary, and five multiclass tasks.
- The Phase II grid has 13,440 jobs: 20 datasets, 12 model configurations, 28 transform settings, two independent context/query split seeds, and one model/warp seed. The Phase I three-seed result justifies reallocating Phase II replication to independent data memberships.
- All 20 OpenML tasks resolved successfully before model evaluation. Each has numerical features; schemas span 0–94 categorical columns.
- Mandatory controls were added: LightGBM, Random Forest, linear/logistic, RealMLP, and TabM. TabM/RealMLP run in isolated workers because pytabkit/Lightning can segfault during interpreter teardown; a result is accepted after that specific exit only when every prediction and telemetry artifact exists. Mixed-schema TabM and RealMLP smokes returned finite probabilities.
- The complete transform matrix now includes decreasing affine, asinh, empirical CDF, quantile Gaussian, atomic spacing, categorical bijection, and composition cells in addition to the Phase I families.
- Dataset descriptors now include robust IQR scale, binned entropy, and tail-heaviness. A group-held-out cross-dataset ridge/random-forest analysis and six prespecified Phase II figure generators are implemented.
- A four-job, two-split mixed-schema runner smoke completed. It exposed a pandas categorical dtype-only mismatch in the semantic round-trip audit; code/value/category-order equality was preserved, and the audit was corrected to test those exact invariants. All four revised records pass artifact validation.
- Final pre-launch verification: 71 tests passed. Phase I historical coverage remains separately reproducible under its original code/config digests.

Decision: launch the frozen Phase II grid under one immutable source digest. Gate G2 remains open; no method or pretraining work is authorized yet.

## 2026-08-31 — user-authorized partial Phase II/III wrap-up

Question: can later-phase evidence be extracted from the stopped Phase II run without misrepresenting incomplete coverage as confirmatory completion?

Findings:

- The user authorized proceeding with available results. `reports/PARTIAL_PHASE_AMENDMENT.md` freezes the change as exploratory and preserves the original Gate G2 rule.
- Phase II stopped at 8,338/13,440 checksum-valid jobs. The analyzer retained 8,240 jobs with their required identity baselines and excluded 98 unpaired completed jobs without imputation.
- Six prespecified figures and all tabular summaries were written to a separate `phase2_partial` directory with a `PARTIAL.json` marker; no confirmatory `DONE.json` was created.
- Current-TFM regression results corroborate posterior instability with small average loss movement for TabICLv2 and historical TabPFN-v2.5, while Mitra remains near its refit-noise baseline. LightGBM disagreement prevents a broad neural-versus-tree interpretation.
- A grouped held-out descriptor screen over five non-affine transform families produced six numerical threshold passes, all for TabICLv2. The strongest was single-estimator spline loss-gap prediction by ridge (`R²=0.662`, 46.0% MAE improvement). Random-PWL disagreement replicated across TabICL single/default (`R²≈0.20`, 20–23% improvement).
- TabPFN-v2.5 and Mitra did not clear both screen thresholds. Small dataset counts, model-dependent missingness, multiple testing, and absent permutation/stability controls keep Route D and Gate G2 unresolved.
- Focused tests after the partial-analysis changes: 3 passed.

Decision: preserve the partial audit as evidence for a TabICLv2-focused descriptor/synthetic follow-up. Do not claim completed Phase II, passed G2, a validated remedy, or ICLR readiness.
