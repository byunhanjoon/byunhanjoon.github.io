# E3 — M0–M5 method kill test

Status: **complete; G3 failed and M6 is prohibited**.

## Frozen primary design

M0–M5 use an 11-neighbor distance-weighted KNN as a cheap within-episode backbone. M0
uses context-fitted z-score coordinates, M1 median/IQR coordinates, M2 a train-only
tie-aware ECDF, and M3 averages identity and random monotone-PWL raw views. M4 and M5
reuse the exact same raw/rank predictions, so the 50/50, development-tuned fixed,
learned, and oracle mixtures have identical expert fit compute.

Gate inputs never include rho, mechanism/warp identity, the coupling bit, query rows, or
query labels. They contain context-only rank-label associations and label-free marginal
summaries. Query labels define training oracle targets and the diagnostic oracle only.
Fixed weights and gate calibration are chosen on development tasks; every reported test
has disjoint generator seeds and 420 tasks per rho/task type.

## Initial kill test

The initial run contained 10,920 episodes (43,680 expert fits). Its untouched 2,940-task
test per task type showed real oracle headroom: fixed-minus-oracle was 0.00462 log loss
[0.00442, 0.00483] and 0.00844 MSE [0.00791, 0.00900]. M5 improved over fixed by only
`3.1e-5` and `2.22e-4`, capturing 0.67% and 2.63% of that headroom.

This triggered the program's sequential 16E failure branch. Each iteration used a fresh
train/development/test seed family; no inspected test was reused:

| Iteration | Permitted change | Classification capture | Regression capture |
|---|---|---:|---:|
| Initial | aggregate quantile + association gate | 0.67% | 2.63% |
| More train | 1,680 → 5,880 training episodes/task | 0.20% | 2.67% |
| Featurewise | aligned shape/association pooling | 0.82% | 4.09% |
| Auxiliary | OOF mechanism/warp probabilities | 0.64% | 6.18% |
| Calibrated | development logit temperature/bias | 0.27% | 16.96% |

The auxiliary task was not information-starved: train-OOF warp accuracy was 94.3% and
mechanism accuracy 61.9%/74.1% in the final run. Classification calibration selected the
identity transform (`temperature=1`, `bias=0`, full gate), while regression selected
`temperature=.5`, `bias=0`, full gate.

## Final fresh-test result

| Method | Classification log loss | Regression MSE |
|---|---:|---:|
| M0 raw | 0.66711 | 0.72947 |
| M1 robust affine | 0.66949 | 0.74068 |
| M2 rank | 0.65857 | 0.69195 |
| M3 transform augmentation | 0.65787 | 0.73138 |
| M4 50/50 | 0.65115 | 0.69996 |
| M4 development-tuned fixed | 0.65040 | 0.69178 |
| M5 calibrated learned gate | 0.65038 | 0.69034 |
| Oracle gate | 0.64596 | 0.68331 |

For classification, fixed-minus-M5 was 0.000012 with a 95% task-bootstrap CI of
[-0.000035, 0.000058], while fixed-minus-oracle was 0.004436 [0.004243, 0.004634]. For
regression, the corresponding values were 0.001436 [0.001218, 0.001659] and 0.008469
[0.007930, 0.009007]. The gate beat fixed on only 48.9%/58.6% of individual tasks.

## Gate decision

G3 fails. Rank dominates the ordinary raw and robust experts on average, reproducing the
E1 negative rather than the required low/high-rho raw/rank crossover. Although the fixed
mixture is strong and the calibrated regression gate captures 17% of oracle headroom,
classification adaptation is indistinguishable from fixed and neither task reaches the
program's 20–30% guide robustly. A joint neural-expert step would be a materially more
expensive method, not evidence that this cheap opportunity exists. The execution-order
rule therefore stops method development before M6, method freeze, E5, or real-data
confirmation.

Artifacts:

- `results/processed/e3_failure_branch_sequence_v1.csv`
- `results/processed/e3_contrasts_calibrated_v1.csv`
- `results/processed/e3_phase_summary_calibrated_v1.csv`
- `results/processed/e3_integrity_calibrated_v1.json`
- `figures/e3_failure_branch_sequence_v1.png`
- `figures/e3_method_kill_calibrated_v1.png`
