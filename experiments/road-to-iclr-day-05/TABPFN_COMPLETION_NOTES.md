# TabPFN completion-study accounting

The completion panel uses installed TabPFN `6.3.0` with the cached v2.5
classifier checkpoint. The package source was inspected before analysis to
avoid describing TabPFN as unaware of permutations.

Default classification inference constructs an ensemble configuration for
each `n_estimators` member. In the installed version it applies the configured
preprocessor alternatives, ordinal-encodes declared/inferred categorical
columns, shuffles feature positions (`FEATURE_SHIFT_METHOD="shuffle"`),
shuffles class IDs (`CLASS_SHIFT_METHOD="shuffle"`), and enables the default
row-fingerprint feature. The experiment fixes `random_state=4201`, declares
categorical positions explicitly, and uses `fit_mode="fit_preprocessors"`.

The three stored internal settings are:

- `1:none`: one member with feature and class shifts set to `None`;
- `1:default`: one member under the package defaults;
- `8:default`: eight default members.

External actions independently vary the input feature order, valid category-ID
maps, and binary target IDs, always using the same mapping for train,
validation, and test and aligning output probabilities back to canonical
labels. External IID/SRS/strength-1/strength-2 estimates are computed from the
stored finite action tensor. They do not silently multiply the internal
ensemble: reports include both external TabPFN calls and the product of
external calls with internal forward ensemble members. A TabPFN call is never
reported as a trained/fitted neural model.

The external study measures residual variation after the installed package's
own mechanisms. It does not claim that TabPFN lacks feature or class
permutation handling.
