# Two-hour follow-up protocol: causal identification and schema semantics

Frozen before the extended runs on 2026-08-31 (Asia/Seoul).

## Objective

Decide whether the first sweep's two leading effects remain scientifically interesting after removing their largest shortcuts:

1. replace summary-vector regressors with a real pretrained tabular foundation model and a raw-row set encoder;
2. replace the manual source/destination oracle with frozen pretrained text embeddings.

The target wall-clock budget is approximately two hours. No result will be promoted merely because additional compute was spent.

## Direction 1: pretrained predictive model versus causal identification

Construct independently sampled pairs of binary SCMs with the same population observational law.

- Randomized causal world: `T ~ Bernoulli(0.5)`, `Y = T xor E_r`, true ATE `1 - 2r`.
- Hidden-confounding world: `U ~ Bernoulli(0.5)`, `T = U xor E_q`, `Y = U xor E_p`, true ATE `0`.
- Set `r = q + p - 2qp`. This makes the complete binary joint distribution of `(T,Y)` identical across worlds while keeping causal effects different.

Add four independent nuisance columns. For six `(q,p)` cells, three sample sizes, and at least 15 data seeds, fit:

- logistic regression;
- random forest;
- TabPFN 6.3 using the locally cached v2.5 classifier weights.

Evaluate held-out observational AUROC, accuracy, log loss, Brier score, confidence, and the plug-in intervention contrast obtained by setting `T=1` versus `T=0`. Preserve paired observational-summary distance and causal error. Run shuffled-label TabPFN controls on a prespecified subset.

A result is foundation-model-relevant if TabPFN is predictively strong and confident in both worlds but necessarily emits approximately the same intervention contrast for them. This does not mean TabPFN claims causal validity; it tests the common predictive-to-causal plug-in workflow.

Also train raw-row DeepSets ATE estimators over many independently sampled datasets. Compare observations only against observations plus an explicit randomized/hidden-confounding assumption bit. Test held-out noise cells and multiple model seeds.

### Direction 1 success gate

- TabPFN observational AUROC at least 0.75 in informative cells;
- confounded-world plug-in ATE error at least 0.30 while predictive confidence remains above 0.70;
- observation-only raw-set ATE error remains near its pairwise lower bound;
- explicit assumptions reduce raw-set MAE by at least 50% on held-out cells.

## Direction 2: real frozen language embeddings

Use locally cached BGE-base, E5-base, and GTE-base encoders without fine-tuning. Train a regularized linear orientation probe only on source schemas. Compare:

- structure-only position baseline;
- word/character TF-IDF;
- each frozen encoder;
- a zero-shot source/destination prototype score;
- oracle roles;
- permuted-description sanity control.

All role nouns and domains in the target split are disjoint from training. Evaluate name-only text, clean descriptions, paraphrases, opaque field names with informative descriptions, and ambiguous text. Then pass predicted orientations into the same downstream numeric classifier used for all methods. Use at least 10 numeric seeds and 20 held-out role pairs. Audit real workspace vocabulary for airline origin/destination, taxi pickup/dropoff, and bike start/end fields, but do not generalize from three examples.

### Direction 2 success gate

- a frozen encoder exceeds TF-IDF held-role orientation accuracy by at least 15 points;
- downstream held-schema AUROC improves at least 5 points over structure-only;
- the effect survives opaque names when descriptions remain informative;
- shuffled descriptions destroy the gain;
- no claim of semantic transfer if only the oracle clears these gates.

## Reporting and falsification

Raw per-cell records, model/environment versions, seeds, exceptions, and runtime are mandatory. The follow-up report must distinguish mathematical non-identification, behavior of an actual pretrained model, semantic role recovery, and downstream prediction. Failed models and conditions remain in the tables.
