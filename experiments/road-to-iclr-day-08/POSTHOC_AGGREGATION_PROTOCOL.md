# POST-HOC CORRECTIVE PROTOCOL — AGGREGATION-AWARE RETRIEVAL RISK

Written after the frozen candidate-wise reranker failed.  This status is
intentional and permanent: results from this experiment cannot retroactively
make the prospective primary screen pass.

## Motivation

The prospective study showed that mean top-k one-neighbor risk can decrease
without a prediction gain.  Proposition A3b explains why: the diagnostic drops
signed mismatch cancellation and squared-weight noise dilution.  This final
corrective experiment optimizes A1 itself rather than averaging A3.

## Frozen corrective estimator

For each already-trained TabR or ModernNCA model, retain its `k` nearest rows by
learned distance.  Let cross-fitted conditional-mean discrepancies be `d_i`
(vectors for classification) and candidate uncertainty be `u_i`.  Choose

```text
w* = argmin_{w in simplex} ||sum_i w_i d_i||^2 + sum_i w_i^2 u_i.
```

The convex problem is solved by projected gradient descent on GPU.  Prediction
is the weighted average of the observed candidate targets/classes.  The neural
model supplies only the shortlist; its prediction/value head is not reused.

Validation chooses `k` from `[16, 32, 64]`; ties choose the smaller shortlist.
No test target enters weights or selection.  The following are reported:

1. original learned-distance model prediction;
2. full plug-in aggregate-risk weights;
3. mismatch-only weights (candidate uncertainty held constant);
4. reliability-only inverse-variance weights within the learned shortlist;
5. the cross-fitted ExtraTrees conditional-mean proxy used to construct `d`.

The last baseline is mandatory: if using the proxy directly is as good or
better, retrieval adds no demonstrated value.

## Scope and controls

- Reuse all 12 prospective datasets, three splits, three model seeds, both key
  models, checkpoints, preprocessing, and OOF proxies without retraining.
- Synthetic: all four tasks, eight seeds, both key models; compare exact and
  estimated aggregate-risk weights.  S3 remains primary.
- Check simplex feasibility and objective descent numerically for every batch.
- Classification uses the Brier-risk vector extension; performance reporting
  remains accuracy/log loss. Regression uses standardized RMSE.
- All outputs are labeled post-hoc. No new dataset, representation, or model
  capacity is introduced.

## Stop rule

Stop the retrieval-risk method direction unless full plug-in weighting:

- beats the original model on at least 8/12 datasets for both key models;
- has positive dataset-balanced score gain for both;
- beats the direct cross-fitted proxy on at least 8/12 for both; and
- shows a clear exact and estimated S3 gain over the original model.

Regardless of outcome, do not launch a larger benchmark in this Day-8 run.
