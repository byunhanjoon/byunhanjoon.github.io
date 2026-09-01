# Independent classification-shrinkage confirmation

The real-development-tuned lambda=0.1 candidate passes its independent panel gate.

Across breast-w, credit-approval, credit-g, spambase, and electricity (400 fresh
episodes / 9,600 expert fits), shrinking 90% toward the fixed mixture improves
dataset-balanced log loss by **0.000600** with 95% hierarchical CI
**[0.000038, 0.001471]**. Fixed and candidate losses are 0.446659 and 0.446059, a 0.134%
relative reduction. Three of five datasets improve; the two negatives are -0.000023 on
credit-g and -0.000002 on electricity.

A post-result dependence sensitivity addresses the heavy query overlap in the two
smallest official test folds. Bootstrapping only the five dataset means gives
[0.000051, 0.001455], and every leave-one-dataset-out mean stays positive; the minimum is
+0.000181 after excluding Spambase. The conclusion therefore does not depend on treating
the 80 repeated episode samples as independent.

Brier score also improves by 0.000207 [0.000001, 0.000527]. AUC is unchanged
(+0.000101 [-0.000322, 0.000551]), and the NLL>2 rate does not improve. Full competence
has a favorable point gain (+0.002292) but a wide interval crossing zero and mixed
dataset signs. The confirmed result is therefore the small frozen shrinkage step, not
full routing or tail-risk elimination.

This is a modest performance result with an important provenance distinction: lambda
was selected on the earlier real panels after synthetic-only transfer failed, then
confirmed once on these deterministic CC18 identities. It does not restore the original
synthetic-only classification claim, cover categorical features, or constitute a novel
shrinkage algorithm.

Artifacts: `CLASSIFICATION_SHRINKAGE_CONFIRMATION_PROTOCOL.md`, config hash
`c41a629643...`, raw run `classification_shrinkage_confirmation_c41a629643`, and
`results/processed/classification_shrinkage_confirmation_audit_v1.json`.
Raw reconstruction matches parent log losses within `7.93e-9`; the raw bundle SHA-256
begins `1a2f79541ec1` and the parent cells hash begins `8b1f4c109acd`.
The dataset forest plot is `figures/classification_shrinkage_confirmation_v1.png`.
