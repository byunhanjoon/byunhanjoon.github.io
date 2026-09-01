# Electricity 2x2 screen: no configuration advances

## Outcome

The frozen screen produced a clear negative result. Neither doubling mixture components nor replacing the flattened MLP with a channel-aware temporal Transformer closes any of the Electricity gap. The original calibrated four-component MLP remains the best analytic projective model tested.

| Configuration | Parameters | Mean CRPS | Change vs. MLP-K4 | Paired wins | Advances? |
|---|---:|---:|---:|---:|---:|
| MLP-K4 | 136,580 | **0.21312** | — | — | reference |
| MLP-K8 | 136,495 | 0.21833 | +2.45% | 0/3 | no |
| Attention-K4 | 134,494 | 0.21675 | +1.70% | 0/3 | no |
| Attention-K8 | 137,396 | 0.21978 | +3.13% | 1/3 | no |
| TACTiS-2 | 193,282 | **0.18703** | -12.24% vs. MLP-K4 | — | baseline |

The advancement threshold was 0.20007, representing closure of at least half the MLP-K4-to-TACTiS gap. No new configuration approached it.

## Factor diagnosis

Every controlled factor effect points in the wrong direction (positive means worse CRPS):

- K8 minus K4 with MLP: +0.00521.
- K8 minus K4 with attention: +0.00303.
- Attention minus MLP with K4: +0.00363.
- Attention minus MLP with K8: +0.00145.

All configurations passed capacity, finiteness, calibration, and speed conditions. They failed on predictive quality and paired wins, so this is not a compute, calibration, or threshold artifact.

## Updated diagnosis

The following straightforward explanations for Electricity have now been tested and rejected:

1. global dispersion error — calibration helps reliability but not the CRPS gap;
2. insufficient within-component covariance rank — rank-4 made Electricity worse;
3. too few mixture components — K8 made both backbones worse;
4. lack of basic temporal attention — the matched Transformer made both component counts worse.

The highest-value remaining hypothesis is the **training objective**. The projective models learn from one random scalar projection per example, whereas TACTiS learns a joint/copula objective over the future. The scalar objective may provide a noisy or weak signal for identifying an entire 32-dimensional conditional law, particularly on Electricity.

## Recommended next screen

Keep the winning MLP-K4 architecture and compare, on Electricity first:

1. current scalar-query NLL;
2. normalized full-joint mixture NLL;
3. a hybrid of scalar-query and normalized joint NLL.

All three retain exactly the same analytic projective family, capacity, and inference speed. This isolates supervision rather than architecture. A hybrid should advance only if it reaches CRPS at most 0.20007 and wins at least two seeds; otherwise the current analytic Gaussian-mixture family should be considered plateaued against TACTiS on Electricity.

## ICLR implication

The negative result narrows the story but lowers the probability of an oral-level forecasting paper based solely on architectural upgrades. The current positive claim remains two-of-three-dataset competitiveness, better average calibration after validation scaling, and an approximately three-orders-of-magnitude query-time advantage. A successful joint/hybrid objective is now the most plausible route to making the quality claim robust.

## Reproducibility

- Protocol SHA-256: `e37bf77277385647f228c64c0525e0b9a080e5787245851717cfb17351dc9cd5`
- Nine new training checkpoints and twelve calibrated evaluation cells.
- All results finite.
- Total screen wall time: 217.7 seconds.
- Machine-readable results: `representation_outputs/audit.json`, `evaluation_cells.csv`, `evaluation_summary.csv`, and `calibration_cells.csv`.
