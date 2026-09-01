# Multiclass model-selection scope addendum

Status: **frozen before model-selection outcomes**; the underlying multiclass
prediction tensors were already generated for the cover scope test.

Apply the unchanged equal-budget model-selection analysis to Vehicle (4
classes) and Segment (7 classes), with linear, forest, and Adam-MLP candidates.
Use 1,024 validation-only actions and semantically align class probabilities
before multiclass Brier scoring.

This is a small scope addendum, not an independent external confirmation.
Report winner agreement, validation quotient regret, selected realized test
loss, and validation-to-test winner alignment.  The addendum is favorable if
strength-2 has higher mean agreement and lower mean validation regret than
IID-16, and lower realized test loss on at least one of two datasets.  Retain
the result regardless of direction.
