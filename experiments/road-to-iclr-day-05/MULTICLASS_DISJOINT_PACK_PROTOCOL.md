# Multiclass disjoint-packing scope protocol

Status: frozen before inspecting packing outcomes on the existing multiclass
tensors.

On Vehicle (4 classes) and Segment (7 classes), compare a 32-fit disjoint
strength-2 pair with an independent two-cover mean, then evaluate a mutually
disjoint four-cover pack at 64 fits. Use 1,024 actions and vector Brier loss.

The gate passes if pair32 has lower score RMSE and direct prediction residual
for all 6/6 candidates and pack64 equals the exact quotient score to `1e-12`
for all 6/6. Report selection but expect the previously documented exact-winner
ceiling. This is a frozen vector-valued scope addendum, not new-source evidence.
