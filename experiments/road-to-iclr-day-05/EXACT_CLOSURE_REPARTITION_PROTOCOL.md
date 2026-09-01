# Frozen protocol: exact packed closure under evaluation repartitioning

Reuse the 1,024 paired stratified/random repartitions of the external and
task-balanced pooled validation+test sets. Restrict to datasets whose nuisance
product has at most 64 cells, where a 64-fit four-pack prediction—and therefore
the 128-fit independent-pack cross-score—is exactly the quotient for every
candidate.

Compare exact quotient validation selection with the complete U-statistic over
eight independent 16-fit covers (128 fits). Report validation quotient regret,
complement-test quotient regret, validation/test winner agreement, paired
source differences, and source-cluster intervals.

The method gate requires lower mean validation regret in both panels. A
separate transfer gate requires no higher mean complement-test regret in both
panels and both source-bootstrap upper endpoints below zero. If only the method
gate passes, label the result `validation_only_pass`; exact nuisance integration
does not remove evaluation-sample target shift.
