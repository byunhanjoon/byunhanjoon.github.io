# Day-5 program completion matrix

Status: **complete**. This checklist distinguishes the new frozen remaining-work
panel from earlier exploratory/adaptive evidence. No missing configured cell was
replaced by another dataset or model.

| Program requirement | Final completion evidence | Status |
|---|---|---:|
| Exact feature/category/target schema nuisances plus seed | Full mixed products with separate initialization and dataloader factors; 16 exact-subset tensors and 144 repeated-split tensors | complete |
| MLP | 12 datasets × 3 splits, exact subset, matched control, menu and row-order controls | complete |
| ResNet | Same common panel and controls as MLP | complete |
| FT-Transformer | Official `rtdl_revisiting_models` backbone on the same panel and controls | complete |
| TabM | Installed official `tabm` backbone on the same panel and controls | complete |
| TabPFN internal/external study | 6 classification datasets × 3 splits; disabled/default/8-member internal settings and external IID/SRS/S1/S2 | complete |
| CatBoost / GBDT | CatBoost, HistGradientBoosting, XGBoost, and LightGBM on all 12 datasets | complete |
| Invariant negative control | One-hot logistic/ridge on all 12 datasets; risk near numerical zero | complete |
| Task-balanced breadth | 12 datasets: 6 classification and 6 regression | complete |
| Repeated partitions | Three frozen splits for every modern-neural dataset×model cell | complete |
| Matched initial function | 4 datasets × 4 neural families × ordinary/matched arms; telemetry rerun bit-identical | complete |
| Exact tensors and fANOVA | Five-factor fANOVA with schema-only, seed-only, and schema×seed terms | complete |
| IID/SRS/LHS/Sobol/strength 1/2/3 | B=1–64 with B16 comparisons and mathematically valid B64 strength-3 | complete |
| Negative-dependent packing | Disjoint pair32 and mutually disjoint pack64; exact closure when the finite product fits | complete |
| Ranking and model selection | Four modern candidates on 12 datasets × 3 splits, validation/test regret and fidelity | complete |
| Independent-cover cross-score | IID and strength-2 cross32 transport analysis | complete |
| Larger natural-menu approximation | 2 datasets × 4 neural models × 256 states; nested M=4/8/16/32/64 | complete |
| Row/order control | 4 datasets × 4 models with initialization/minibatch seed fixed | complete |
| Secondary metrics | Brier/MSE plus log loss, accuracy, AUROC, ECE, MAE, RMSE, and R² where applicable | complete |
| Runtime and memory | Fit telemetry for neural, matched, menu, row-order, TabPFN, and wall time for CPU classical runs | complete |
| Dataset/model metadata | Required dataset and model tables with environment/package capture | complete |
| Frozen protocol and deviations | Hashed completion protocol plus three explicit implementation-correction records | complete |
| Required figures/tables | Six tables and ten figure concepts in CSV/Markdown and PNG/PDF | complete |
| Integrity and tests | 587 tensors, 44,720 represented fits/calls, zero audit issues, 103 tests | complete |
| Final `results.md` | Consolidated success/failure report with provenance and limitations | complete |

Optional TabR and SAINT were not locally callable. Their absence is recorded in
`completion_environment.json`; they were strongly preferred but not mandatory,
and were not installed after the primary model panel was frozen.
