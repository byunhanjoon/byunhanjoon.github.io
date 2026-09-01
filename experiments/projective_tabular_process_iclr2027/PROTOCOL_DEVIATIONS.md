# Protocol deviations

1. CatBoost 1.2.10 rejects bootstrap samples whose labels are all equal. Four
   16-shot solar-flare episodes encountered such a member. The implementation
   now uses the mathematical fitted solution for that member---the observed
   constant function---while leaving all other bootstrap members, episodes,
   datasets, and scores unchanged. This is a degenerate-case implementation
   completion, not an outcome-dependent hyperparameter or dataset change.

2. Thirteen evaluation episodes exposed two edge cases in the pinned
   TabPFN-2.5 package. Twelve small-context `fps_benchmark` episodes had an
   object-typed feature with every context and query value missing; the package
   attempted to convert its internal missing-category token to float and
   failed. We deterministically drop columns that are entirely missing in the
   context. Such columns contain no context-conditional information, and the
   rule never uses query labels. One constant-label `solar_flare` context
   produced float64 distribution borders and float32 logits; we cast logits to
   the criterion's border dtype before computing its variance. Model weights,
   means, episodes, seeds, and hyperparameters are unchanged, and affected
   metadata records the dropped columns.

3. Some 16-shot `bike_sharing` application episodes made a categorical column
   constant in the context. TabPFN-2.5 then failed to infer it as categorical
   and attempted to cast string categories to floats. On precisely this package
   validation error, categorical/object columns are replaced by deterministic
   context-derived integer codes (unseen query categories map to -1) and the
   same model is rerun. This fallback uses no query labels and is recorded in
   episode metadata; it does not affect the 35-dataset primary benchmark.

4. The predeclared integrity audit found that batched TabICLv2 preprocessing
   is query-set dependent: removing half the query rows changed hidden states by
   as much as 0.071 and induced covariance differences up to 0.00924. Therefore
   the first batched lift is positive semidefinite but not projective and is
   retained only as a failed diagnostic. The corrected method evaluates every
   query row as a singleton after fitting the context once, making its mean,
   marginal scale, and feature map functions only of `(context, x)`. The head
   is retrained and all primary scores are rerun on these singleton features;
   no dataset, label, query, seed, success gate, or test-dependent hyperparameter
   is changed.

5. The frozen protocol permitted TabPFN-3 as an optional point-prediction
   baseline if official local weights were accessible. After acquiring and
   pinning the official v3 checkpoint, we verified that package 8.5.0 also
   exposes a predictive marginal variance. We therefore add a globally
   development-temperature-calibrated diagonal aggregate analysis as an
   explicitly post-protocol secondary baseline. It does not enter the primary
   ProjTabICL-versus-identical-diagonal gate or any model selection.
