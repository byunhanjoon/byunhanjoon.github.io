# Limitations

The initial generator uses short, purely numerical programs and a restricted
operator library. The constrained polynomial baseline is only a pilot control;
a competitive symbolic-regression system remains mandatory before publication.

The heterogeneous synthetic generator and typed selector share an operator
library. The current result tests the value of a matched typed inductive bias,
not open-ended typed program discovery. Categories are low-cardinality and
timestamps have no missing or malformed values.

The neural residual is not OOD-safe. Penalization controls its in-distribution
usage but does not prevent learned contributions from exploding beyond the
training shell. A shift-aware gate is required before deployment.

Real temporal relevance is currently negative, not merely missing. The current
season router catastrophically extrapolates local elapsed-time trends on UCI
Bike Sharing. The study covers only one public temporal dataset and does not yet
include the full TabReD benchmark.
