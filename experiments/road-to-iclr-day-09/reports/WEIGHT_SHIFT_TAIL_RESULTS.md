# Weight shift versus tail risk

Moving farther from the fixed mixture has a consistent tail cost in classification but
no universal monotone benefit in regression.

On the six unseen binary identities, the equal-dataset mean within-dataset Spearman
correlation between `KL(w_competence || w_fixed)` and worst-decile-NLL gain is -0.3524
[-0.5157, -0.1923]. Every dataset correlation is negative. The highest-KL episode
quintile has 0.07868 worse tail gain than the lowest-KL quintile
[-0.12097, -0.03483]. Larger adaptive moves therefore track the rare-error failure.

On 16 regression identities, the mean within-dataset Spearman correlation is +0.0124
[-0.0505, +0.0798], so no general monotone association is supported. The high-minus-low
KL contrast is positive on average (+4.6138 [0.0884, 12.1464]) because a small set of
large-headroom datasets—especially Physiochemical Protein, House, Wine Quality, and
Geographical Origin of Music—have very large positive contrasts. Ten datasets instead
have small negative contrasts.

Mean weight movement is also larger in regression (KL 0.944, TV 0.519) than
classification (KL 0.327, TV 0.294), yet only classification shows a consistent
within-dataset tail penalty. The evidence rejects a universal adaptation-magnitude rule:
stronger movement is predictably risky for binary log loss, while regression value
depends on dataset-specific reducible tail headroom. No safe threshold is inferred.

Artifacts: `WEIGHT_SHIFT_TAIL_PROTOCOL.md`,
`results/processed/weight_shift_tail_detail_v1.csv`, and
`results/processed/weight_shift_tail_audit_v1.json`.
