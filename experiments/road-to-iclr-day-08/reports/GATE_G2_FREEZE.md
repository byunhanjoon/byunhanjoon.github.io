# Gate G2 prospective decision rule

Frozen: 2026-08-31, before Phase II completion or Phase III outcome inspection.

## Question

Can the matched reparameterization effect be connected to a scientifically interpretable property of feature marginals, learned representation/readout behavior, or a controlled task prior—rather than remaining an opaque preprocessing artifact?

## Minimal pass routes

At least one route below must pass with its prespecified controls. A paper-strength explanation should triangulate two routes; a single weak correlation is not enough to justify remedy/pretraining work.

### Route D — marginal descriptors

Pass only if a transparent model predicts held-out-dataset sensitivity for at least one current TFM with:

- grouped cross-dataset validation that keeps both splits of a dataset in the same fold;
- out-of-sample `R² >= 0.10` and at least 10% lower MAE than a training-fold-mean baseline;
- replication for at least two non-affine transform families or for both loss gap and posterior disagreement;
- directionally stable descriptor associations across the two split seeds;
- a target-permutation control centered near chance; and
- no feature derived from query rows, labels, model predictions, or dataset identity.

Random-forest importance alone does not pass. Ridge coefficients must be reported after standardization, and correlated descriptors must be discussed as associations rather than causal mechanisms.

### Route S — controlled prior behavior

Pass only if the frozen S1–S6 protocol shows both:

1. useful metadata: paired S2 performance is better than S3 at small contexts, with a hierarchical 95% interval excluding a negligible effect and attenuation as context grows; and
2. harmful conflict: S4 conflict is worse than S4 in-prior and S3 under aligned task/noise seeds, again beyond a negligible interval.

The same qualitative pattern must appear in TabICLv2 and at least one other current/learned family or explain clearly why a contrasting family is robust. S1 sensitivity without S2/S3/S4 structure does not pass.

### Route R — representation/readout mechanism

Pass only if an accessible current model shows:

- reproducible representation, neighborhood, attention, or readout change under matched warps;
- paired association between that internal change and prediction disagreement across datasets; and
- a causal restoration/intervention that moves both the internal statistic and prediction back toward the original task more than a matched sham intervention.

Hooks must not modify normal predictions, and all internal comparisons must preserve row/feature alignment. Correlation without restoration is supportive but does not pass this route alone.

## Automatic failure conditions

Gate G2 fails if any of the following holds after reasonable in-scope checks:

- sensitivity is explained entirely by lossy transforms, missingness changes, context/query mismatch, refit noise, or an adapter bug;
- descriptor predictability disappears under dataset-grouped validation or permutation controls;
- synthetic effects require post-outcome generator changes or occur only in a narrow unconnected family;
- internal changes are not associated with predictions or cannot survive a sham-control comparison; or
- the explanation applies only to historical TabPFN-v2.5 while current TabICLv2/Mitra evidence is absent.

## Decision consequences

- **Pass, paper-strength:** at least two routes pass, or one route passes strongly and explains cross-family heterogeneity. Proceed to the method ladder.
- **Pass, narrow:** exactly one route passes cleanly. Permit only cheap M1–M4 remedies first; do not start pretraining unless one improves the development tradeoff.
- **Fail:** write the negative explanation audit, update `ICLR_READINESS.md`, recommend a mechanistic/reliability pivot if defensible, and stop RSPF/pretraining.

All effect sizes, failed routes, worst datasets, and alternative explanations remain reportable. This gate cannot be relaxed after outcomes without a dated amendment that labels subsequent evidence exploratory.
