# UCI Bike Sharing confirmation freeze

Frozen before the Day 6 MPE runs.  This is a practical successor diagnostic,
not prospective evidence: Day 4 already used this dataset for a different
cyclic spectral representation.

Only the `hr` field changes.  All other numerical, binary, and categorical
features, the chronological 60/20/20 split, target `log1p(cnt)`, training
budget, and backbone are shared.  Every hour representation has 16 columns:

- quantile PLE;
- fixed eight-frequency periodic features;
- path/code-distance RBF partition;
- correct 24-hour ring MPE;
- multiscale ring MMPE;
- a permuted-ring corrupt control.

Run MLP and ResNet with three frozen seeds.  The single-scale MPE practical
gate requires positive mean test-loss reduction versus Q-PLE and code-RBF in
both backbones, plus at least 4/6 wins over the corrupted metric.  MMPE replaces
MPE only if it has lower mean test loss in both backbones.  Since the dataset is
not untouched, all test comparisons are descriptive and cannot promote the
method to a general empirical claim.
