# Phase I kill-test analysis draft

This draft is generated only after the frozen job grid and all artifact integrity checks complete.

## Question

Do current tabular foundation models change predictions or performance under fully matched, invertible numerical reparameterizations beyond identity-refit noise and more than tree controls?

## Exact protocol

- Config: `/home/byunhanjoon/byunhanjoon.github.io/experiments/road-to-iclr-day-08/configs/audit/pilot.yaml` (`5c75900a6a3bfb607a24a679b9f8a8545d4f7126cea391e0ef301b85e994720f`)
- Code digest: `595cc6ae397a9a551882104812ecaf6e41b0e49fc87833685b747c467d009227`
- Complete jobs: 3276/3276
- Datasets: 12; seeds: [20260831, 20260832, 20260833]; context/query caps: 2048/1024
- Each job uses two fitted contexts: original for clean/query-only and transformed for matched/context-only.
- Primary aggregate unit is dataset. Confidence intervals are 10,000-draw paired dataset bootstraps.

## Result table

| Model | Task type | Datasets | Normalized loss gap (95% CI) | Excess disagreement (95% CI) | W/T/L |
|---|---:|---:|---:|---:|---:|
| catboost | binary | 4 | -6.6848e-05 [-0.00034737, 0.00014584] | 7.6932e-05 [2.7799e-05, 0.00012606] | 1/2/1 |
| catboost | multiclass | 4 | -7.4222e-05 [-0.000284, 0.00015915] | 5.7737e-05 [1.1182e-05, 0.00012195] | 2/1/1 |
| catboost | regression | 4 | 0.00054639 [2.4076e-05, 0.0010687] | 0.015449 [0.0044046, 0.026494] | 0/1/3 |
| mitra_default | binary | 4 | 4.2631e-06 [-8.6384e-05, 0.00012059] | 3.5978e-05 [1.8994e-05, 6.0563e-05] | 1/2/1 |
| mitra_default | multiclass | 4 | -0.00020795 [-0.00049481, 5.3339e-05] | 5.4297e-05 [-2.2321e-05, 0.00016255] | 2/1/1 |
| mitra_default | regression | 4 | -4.688e-05 [-0.00015242, 4.6421e-05] | 0.0029086 [0.00028435, 0.0051221] | 1/3/0 |
| tabicl_v2_default | binary | 4 | 0.0039672 [-3.5902e-05, 0.010879] | 0.00089861 [0.00024855, 0.0020763] | 1/0/3 |
| tabicl_v2_default | multiclass | 4 | 0.0022457 [0.0008637, 0.0032223] | 0.00060572 [0.00039144, 0.00078768] | 0/0/4 |
| tabicl_v2_default | regression | 4 | 0.00070673 [0.00046161, 0.00092909] | 0.027238 [0.014053, 0.04384] | 0/0/4 |
| tabicl_v2_single | binary | 4 | 0.039735 [0.00098191, 0.1068] | 0.007411 [0.00056671, 0.018883] | 1/0/3 |
| tabicl_v2_single | multiclass | 4 | 0.0044556 [0.0013866, 0.0086221] | 0.0011736 [0.00059782, 0.0018632] | 0/0/4 |
| tabicl_v2_single | regression | 4 | -4.7731e-05 [-0.0011077, 0.0010123] | 0.032142 [0.017561, 0.050037] | 2/0/2 |
| tabpfn_v25_default | binary | 4 | 0.0019164 [0.00028169, 0.0039259] | 0.00049004 [0.00015766, 0.00097641] | 1/0/3 |
| tabpfn_v25_default | multiclass | 4 | 0.0049643 [-0.00015095, 0.012385] | 0.0013669 [0.00042596, 0.0026656] | 1/0/3 |
| tabpfn_v25_default | regression | 4 | 0.0018248 [-0.00051513, 0.0046003] | 0.021974 [0.012711, 0.032066] | 1/0/3 |
| tabpfn_v25_single | binary | 4 | 0.0028278 [-0.00083843, 0.0069218] | 0.00090878 [0.00053448, 0.0014658] | 2/0/2 |
| tabpfn_v25_single | multiclass | 4 | 0.0090054 [-0.0012621, 0.021497] | 0.0027878 [0.00075833, 0.0054586] | 2/0/2 |
| tabpfn_v25_single | regression | 4 | 0.0040199 [0.00047671, 0.0077021] | 0.040751 [0.025344, 0.065057] | 1/0/3 |
| xgboost | binary | 4 | 0.00010231 [-1.8164e-05, 0.00022351] | 0.0001019 [2.7921e-05, 0.00017589] | 0/2/2 |
| xgboost | multiclass | 4 | 2.8938e-07 [-8.3457e-05, 6.5711e-05] | 2.7853e-05 [3.1082e-06, 5.3973e-05] | 1/3/0 |
| xgboost | regression | 4 | 0.000143 [-7.8684e-05, 0.00036468] | 0.006422 [0.0018101, 0.011996] | 1/1/2 |

## Plots

- `model_landscape.png`
- `matched_vs_mismatch.png`
- `severity_curves.png`

## Interpretation and alternative explanations

Interpretation and Gate G1 are intentionally left for evidence review. Check identity-refit noise, transform severity, default-vs-single ensembles, categorical preprocessing, optimization nondeterminism, and any model-specific preprocessing before attributing an effect to the learned prior.

## Raw results

- Manifest: `/home/byunhanjoon/byunhanjoon.github.io/experiments/road-to-iclr-day-08/results/MANIFEST.jsonl`
- Validated run paths and metrics: `/home/byunhanjoon/byunhanjoon.github.io/experiments/road-to-iclr-day-08/results/analysis/phase1/595cc6ae397a9a55/run_metrics.csv`
