# E0 — Day-8 reproduction

Status: **executed for available current families; TabPFN-3 access-incomplete**.

## Frozen scope

The panel deterministically selected the first regression (`wine_quality`) and first
binary (`churn`) tasks in the Day-8 frozen list, capped at 512 context and 256 query rows.
Identity, positive affine, random monotone PWL, and monotone spline were run at three
seeds. The original four-way paired-fit protocol was reused: clean/query-only share one
fit and matched/context-only share another.

TabICLv2 single/default and isolated official Mitra produced 72/72 complete jobs with no
failures. TabPFN-3 default/OOD produced 48 immutable unavailable records because neither
official client/checkpoints nor credentials are locally available; v2.5 was not
substituted. Therefore this report reproduces the TabICL/Mitra contrast but does not
satisfy the TabPFN-3 portion of E0.

## Integrity

All 72 prediction checksums, paired-fit IDs, and row alignments passed. All transform
audits preserve order and missing masks with zero strict-order violations. Worst relative
inverse reconstruction error was `4.14e-7`. Evidence is in
`results/processed/e0_integrity_v1.json` and the immutable raw directories under
`results/raw/e0_reproduction/`.

## Results

The reported intervals below are diagnostic bootstrap intervals across the nine
transform-seed cells, not dataset-generalization intervals; each task type has only one
dataset here.

| Model | Churn excess TV [95% cell CI] | Wine excess normalized disagreement [95% cell CI] |
|---|---:|---:|
| TabICLv2 single | 0.01874 [0.00955, 0.02685] | 0.04355 [0.02291, 0.05913] |
| TabICLv2 default | 0.01862 [0.00917, 0.02667] | 0.03765 [0.01940, 0.05397] |
| Mitra default | 0.00192 [0.00086, 0.00314] | -0.00019 [-0.01549, 0.01405] |

Mitra's identity/refit floor was 0.00148 TV on churn and 0.02158 normalized disagreement
on wine quality; TabICL identity disagreement was exactly zero. Matched normalized loss
did not materially worsen for Mitra. TabICLv2-default churn loss increased by 0.01268
[0.00614, 0.01919], while its wine loss interval included zero.

This reproduces a cross-family difference in *posterior/prediction stability*, not a
universal performance failure and not a causal architectural explanation. The result is
consistent with Day-8 but remains a two-dataset development audit.

Artifacts:

- `results/processed/e0_model_dataset_summary_v1.csv`
- `results/processed/e0_cells_v1.csv`
- `results/processed/e0_availability_v1.csv`
- `figures/e0_reproduction_v1.png`
