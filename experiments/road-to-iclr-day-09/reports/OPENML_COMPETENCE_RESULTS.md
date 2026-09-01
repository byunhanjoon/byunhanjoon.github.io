# OpenML breadth and independent regression confirmation

Status: **scoped regression transfer passed and independently confirmed; classification
transfer rejected**.

## Unseen Day-8 OpenML identities

The first breadth run used all 13 compatible Day-8 identities not seen in the initial
real panel, with no real-data tuning.

| Task | Datasets | Fixed loss | Competence loss | Dataset-balanced gain (95% hierarchical CI) | Decision |
|---|---:|---:|---:|---:|---|
| Classification | 6 | 0.369194 | 0.371324 | **-0.002130 [-0.004652, -0.000256]** | rejected; fixed is better |
| Regression | 7 | 0.896852 | 0.605824 | **+0.291028 [0.025246, 0.946467]** | scoped transfer passed |

Regression also beats uniform by 0.141822 [0.012791, 0.465479]. Five individual
dataset intervals favor competence, Miami Housing is inconclusive with a positive point
estimate, and Physiochemical Protein is positive but highly variable. No identity was
removed. Classification soft weighting still beats hard selection, but not the stable
fixed mixture; Diabetes has a clear negative and the other five point estimates are near
zero or negative.

## Dataset-cross-fitted temperature/shrinkage

A leave-one-dataset-out diagnostic over the complete 9-classification/11-regression real
panel shrank classification weights halfway toward fixed and recovered a small positive
point estimate, +0.000824 [-0.000563, 0.002310], but not a positive interval. Regression
remained better than fixed, +0.241132 [0.016416, 0.656430], yet was 0.030248
[0.001407, 0.120410] worse than the original synthetic-tuned competence router. This
calibration variant is not promoted.

## Independent regression confirmation

A deterministic task-ID rule selected five further identities before outcomes. After a
structural query-size correction, all 300 episodes completed.

| Dataset | Competence gain vs fixed (episode-bootstrap CI) |
|---|---:|
| Abalone | +0.019705 [0.009909, 0.028971] |
| Auction Verification | **-0.006629 [-0.011584, -0.001815]** |
| Geographical Origin of Music | +0.394287 [0.334574, 0.455078] |
| Naval Propulsion | +0.094491 [0.083463, 0.105376] |
| Solar Flare | +0.000398 [0.000095, 0.000713] |

The dataset-balanced gain is **+0.100451 standardized MSE [0.001731, 0.250552]**,
passing the frozen confirmation gate. Four of five identities favor competence; the
Auction Verification failure is retained. The rule also has a favorable but uncertain
gain over uniform (+0.067763 [-0.001752, 0.176280]).

## Scope

The supported synthetic-only performance result is narrow but real: a context-only soft
competence router improves a fixed six-expert mixture on two disjoint real regression
panels totaling 12 datasets, with a positive dataset-hierarchical interval in each panel.
Full competence does not transfer as an improvement over fixed to binary classification.
A later real-development-tuned 10% adaptation step produced a small independent binary
gain, reported separately; it does not change this rejection. The algorithm is standard
exponential aggregation; novelty remains the controlled target-alignment story and the
synthetic-to-real task asymmetry, not the weighting rule.
